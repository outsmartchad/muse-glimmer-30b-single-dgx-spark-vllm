# How Muse Glimmer reads text and images on one DGX Spark

## How the model works

A chart and the question “Which curve peaks first?” can occupy the same Muse Glimmer prompt. The chart starts as pixels and the question starts as text. Two separate input paths turn them into vectors of the same width. One language decoder then reads that combined sequence and writes a text answer. Meta describes Glimmer as a dense causal model distilled from Muse Spark, with about **29.6 billion parameters including the perception encoder**. The benchmarks in this repository exercise text. The chart example below follows the published architecture and the serving code. [Meta model card][meta] · [Pinned vLLM implementation][impl] · [Benchmark inputs](../benchmark.py)

### How the two inputs become a sequence

For the text path, a chat template first supplies the conversation markers the model expects around the user's words. A byte-pair tokenizer represents frequent combinations of text as single units and splits other strings into smaller units. The vocabulary has 200,000 regular tokens and 2,048 special entries. A token can be punctuation, part of a word, a whole word with leading space, or a special marker. Its integer ID is an address into an embedding table. The row it selects contains 6,656 learned numbers. The implementation normalizes that row by its root-mean-square magnitude before it enters the decoder. [Tokenizer and dimensions][meta] · [Embedding implementation][impl]

Those table entries are **weights**. Training learned them along with the projection matrices in the rest of the network. The vectors produced as the network processes this particular question are **activations**. As weights stay fixed during inference, two requests containing the same word start with the same text embedding. The later layer activations can differ because each request supplies its own history. Distillation used a larger teacher to train the released model. Serving the trained student requires its weights and the user's input. The teacher has no part in this recipe's request path. [Meta model card][meta] · [Target launcher](../serve.sh)

An image has no vocabulary ID containing its pixels. The processor prepares a patch grid and reserves a block of visual positions in the prompt, bounded by image markers. In Glimmer's frozen ViT-G/14 vision encoder, a spatial patch is 14 by 14 pixels. A learned projection maps patches into vectors of width 1,536. Fifty vision layers combine patch information, using local window attention in many layers and full-image attention in others. Two-dimensional position information helps a patch represent its place on the chart. “Frozen” describes the pretrained perception weights held fixed during training of the surrounding system. Their forward computation still runs when an image is supplied. [Vision configuration][cfg] · [Perception encoder](https://arxiv.org/abs/2504.13181) · [Frozen encoder release][meta]

The encoder groups neighboring two-by-two patch features by rearranging their values into one wider vector. Four vectors of width 1,536 yield one of width 6,144 and require one downstream visual position. A two-layer adapter using GELU activations, followed by a projection, converts these features to the language decoder's 6,656 coordinates. The runtime inserts each projected feature at its corresponding image placeholder. As a shape example, a prepared 28-by-28 patch grid has 784 patches and becomes 196 visual features after merging. Actual image preparation determines the grid. Meta gives a maximum of 4,096 visual tokens per image. Images therefore consume context and prefill work before the first word of the answer is generated. [Vision encoder and adapter][impl] · [Model limits][meta]

```text
text pieces → vocabulary IDs → normalized 6,656-wide embeddings ─┐
                                                                ├→ interleaved prompt
image pixels → 14 × 14 patches → vision encoder → 2 × 2 merge   │
             → adapter → 6,656-wide visual features ────────────┘
        → [local → local → local → global] × 13 decoder layers
        → final text scores → sampled ID → text continuation
```

### How a layer locates the relevant evidence

Every decoder position has a vector. Learned projections create several **queries** describing information it seeks, **keys** advertising what each position contains and **values** carrying the information that can be returned. A query scores the keys it may see, a softmax turns the scores into relative weights, and summing the corresponding weighted values gives a head's result. A query at the end of the chart question can therefore read a visual feature as well as an earlier word. A causal mask permits the current and preceding positions. This rule lets training predict missing next tokens without revealing them through later inputs. [Decoder and multimodal input code][impl] · [Transformers architecture reference][ref]

Muse Glimmer has 32 query heads and two key/value heads, each of width 128. Each set of 16 queries shares a key/value head, a design called **grouped-query attention**. Sharing reduces the amount of information the server must remember per position. The individual query heads can still assign different scores to those shared keys. Glimmer normalizes the query and key features, applies the model's query scaling, and uses a learned sigmoid output gate to regulate what the attention update contributes. The gate depends on the current input. [Attention geometry][cfg] · [Pinned attention code][impl]

Of the 52 layers, **39 local layers see a 2,048-token window** and 13 global layers can read the full preceding context. At position 10,000, numbered from one, local attention can directly address positions 7,953 through 10,000. The fourth layer of each group can directly address positions 1 through 10,000. In a long multimodal conversation, a recent question could therefore be thousands of visual tokens away from its chart, yet still reach it in a global layer. A visible position only becomes important when its query–key score makes it so. [Layer schedule and window size][cfg]

Position encoding also follows that schedule. The local layers apply rotary position embeddings, rotating pairs of query and key features by a position-dependent angle. The dot product can then express relative distance. The global layers omit that rotation. They still receive contextualized vectors from earlier layers and obey the causal mask. The ordered sequence and the permitted attention region continue to matter at every layer. A correct engine must follow both the mask and the rotary convention when it loads these weights. [Per-layer rotary settings][cfg] · [Pinned vLLM implementation][impl]

### Transforming the features into words

After mixing positions, each layer transforms features with the same **dense feed-forward network** for every token. Two learned projections expand 6,656 inputs to 19,968 features. A SiLU activation on one branch gates the other by elementwise multiplication, and a third projection returns to the original width. This is a SwiGLU network. Glimmer applies RMS normalizations around the attention and feed-forward sublayers and adds their updates through residual connections. An earlier representation can persist while each successive layer changes a portion of it. The full set of dense decoder layers runs for each generated position. [Feed-forward and normalization code][impl] · [Model dimensions][cfg]

At the final position, a normalized vector enters an output matrix separate from the embedding table. It gives a numerical score, or **logit**, to each vocabulary entry. Glimmer scales these scores and applies a smooth tanh cap with magnitude 20, limiting the influence of extremely large raw values. The sampler then decides which token to emit. In the prose benchmark temperature 0.7 sharpens the distribution, top-k 20 retains the twenty strongest candidates and top-p 1 adds no further cumulative-probability cutoff. The chosen ID extends the sequence. Once the tokenizer decodes it, the user may see a word, part of a word, or a punctuation mark. [Scoring code][impl] · [Quantized checkpoint config][cfg] · [Request construction](../benchmark.py)

## How this recipe runs it on one DGX Spark

### From downloaded weights to a localhost API

```bash
./serve.sh download
./serve.sh start
docker logs -f spark-catalog-muse-target-only
# Check readiness from a second terminal on the Spark
curl -fsS http://127.0.0.1:18896/v1/models
```

The first command downloads Red Hat's pinned NVFP4 export and builds an ARM64 container. The Dockerfile pins the NVIDIA PyTorch base by digest, the Muse-enabled vLLM fork at commit `99a10304`, and the CUDA 13 precompiled wheel source revision. That combination supplies native kernels along with the model loader, image processor and Muse-specific parsers. The second command checks the snapshot, mounts it read-only, disables online Hub access, and starts one model API on loopback. Allow approximately 22 GiB of disk for weights plus room for the runtime image. [Dockerfile](../Dockerfile) · [Launcher](../serve.sh) · [Pinned upstream deployment][deploy]

Red Hat calibrated the decoder's linear operators using LLM Compressor. In NVFP4 a group of 16 low-precision floating-point weights shares a finer scale, letting a compact group represent the range that actually occurred. The export also specifies locally scaled four-bit activations for the affected operators. Its exclusions cover the vision tower, adapter, vision projection and output head. The server separately stores attention history in FP8 E4M3. Quantized weight bytes are a cost incurred when loading the model. Cache bytes depend on the runtime's allocation and on what the active requests retain. [Red Hat compression recipe][redhat] · [Pinned group and exclusion rules][cfg]

A client sends the served name `Muse-Glimmer-30B` and a message list to `/v1/chat/completions`. The template accepts `reasoning_strength`. The test used `high` for prose decode and `low` for key retrieval. The parser splits streamed reasoning and answer fields, and the launcher enables automatic tool choice with the Muse tool-call parser. This recipe runs **target-only decode**. The setup record reports an HTTP 403 from the model host for the regional download of the upstream DFlash drafter. Each new output position in the published run came from the target. A DFlash design can verify several proposals per target pass, but there is no measured speculative profile here. [Serving options](../serve.sh) · [Benchmark](../benchmark.py) · [Setup record](../README.md) · [DFlash paper](https://arxiv.org/abs/2602.06036)

### Why a long prompt and five short streams behave differently

In **prefill**, vLLM knows every input position and can evaluate many of them together under the causal masks. It saves keys and values for future use. The global layers need history reaching back to the first retained position. Each local layer only needs a moving window for future outputs. With two KV heads and one byte per FP8 scalar, a single fully retained position in one layer has a logical payload of `2 × 2 × 128 = 512` bytes for K and V together. Thirteen global layers alone account for 6,656 such bytes per position. Actual engine allocations include local caches, blocks, scaling metadata and alignment. The arithmetic explains how the mix of global and local layers changes long-context cost. [Layer dimensions][cfg] · [vLLM memory documentation][mem]

The launcher allows 131,072 tokens per sequence, up to eight scheduled sequences and a GPU-memory utilization target of 0.42. vLLM reported 22.77 GiB reserved for KV and a capacity figure of 3,219,558 token positions. That figure describes the engine's cache pool and its accounting for the hybrid layer layout. A single chat still fits within the configured context including its template, visual positions and desired answer. The sampled GPU-process peak was 46.6 GiB with no fixed Docker memory cap. The Spark's Grace CPU and Blackwell GPU share 128 GB of physical memory, so the configured memory fraction, the reserved KV pool and the observed process peak answer different questions. [Startup figures](../README.md#runtime) · [GPU samples][raw] · [DGX Spark specifications][spark]

**Decode** normally has only one new position per active conversation. Batching five short conversations lets one pass through the dense weights advance five different replies. On November 21 the fastest of three 400-token waves measured **12.4 tok/s** at one stream and **61.2 aggregate tok/s** at five. The median individual stream at five still delivered 12.3 tok/s. These are short text requests to the local API, with output timing beginning at the first nonempty reasoning or answer chunk. [Benchmark protocol](../README.md#benchmarks) · [Raw waves][raw]

The two fresh synthetic-key requests measure a different part of the wait. At **32,742 tokens**, first output came after **10.64 seconds**, equivalent to 3,079 input tok/s. At **124,985 tokens**, it came after **55.15 seconds**, or 2,266 input tok/s. Both retrieved their exact keys. The longer request's first *answer* chunk arrived at 61.61 seconds because the model had already streamed reasoning. Cache-hit counts were unavailable. These results establish two individual text retrievals and their client-visible latencies. Image quality, image prefill time and eight simultaneous full-length requests remain outside the measured workload. [Raw retrieval records][raw] · [Timing definitions](../benchmark.py)

[meta]: https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/1297fd7dfeac4598948a5d15886968435c81c2b7/README.md
[cfg]: https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4/blob/403f780a4d35194b7230bc2d809effa3c147f407/config.json
[redhat]: https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4/blob/403f780a4d35194b7230bc2d809effa3c147f407/README.md
[impl]: https://github.com/xianbaoqian/vllm/blob/99a10304dce8945119bd0b1a072297803c52a749/vllm/model_executor/models/muse_glimmer.py
[ref]: https://huggingface.co/docs/transformers/main/model_doc/muse_glimmer
[mem]: https://docs.vllm.ai/en/latest/configuration/optimization.html
[deploy]: https://github.com/MiaAI-Lab/Muse-Glimmer-30B-DGX-Spark-RTX-5090-6000-PRO/tree/1b71dc011010a3e2c373eafd8a41f12c7c7913fc
[spark]: https://www.nvidia.com/en-us/products/workstations/dgx-spark/#specifications
[raw]: ../results/benchmark-2026-11-21.json
