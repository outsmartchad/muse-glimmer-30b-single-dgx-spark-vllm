# How Muse Glimmer 30B works on one DGX Spark

## How the model works

Muse Glimmer is a dense 29.6B causal Transformer with a frozen ViT-G/14 perception encoder. Its text decoder has 52 layers, hidden size 6,656, 32 attention heads, two KV heads, and a 2,048-token sliding window on three of every four layers. Every fourth layer is full attention with rotary embeddings disabled, which preserves long-range access without making every layer global. The model accepts interleaved text and images, with up to 4,096 visual tokens per image. These fields are documented in the [publisher model card](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/README.md) and the [Transformers architecture reference](https://huggingface.co/docs/transformers/main/model_doc/muse_glimmer).

Muse Glimmer 30B is chosen for a target-only recipe with substantial memory headroom. The pinned RedHatAI checkpoint is [Muse-Glimmer-30B-NVFP4 at `403f780a`](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4/tree/403f780a4d35194b7230bc2d809effa3c147f407). NVFP4 describes the stored model weights. The serving command selects FP8 E4M3 for the KV cache, while vLLM executes the model and parser code from the pinned image built by the Dockerfile.

This recipe deliberately disables speculation. The repository's launch record reports a regional HTTP 403 from Hugging Face while fetching the upstream DFlash draft. That failure is the evidence available here, so draft validation remains open for this clone. The target-only choice leaves room for the long-context KV cache. Muse's chat template exposes `reasoning_strength`, and the benchmark uses `high` for prose decode and `low` for retrieval. Consult the [Muse model card](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4), [DFlash paper](https://arxiv.org/abs/2602.06036), [vLLM KV-cache documentation](https://docs.vllm.ai/en/latest/configuration/optimization.html), and the [pinned upstream deployment](https://github.com/MiaAI-Lab/Muse-Glimmer-30B-DGX-Spark-RTX-6000-PRO/tree/1b71dc011010a3e2c373eafd8a41f12c7c7913fc) for primary references.

## How this recipe runs it on one DGX Spark

`serve.sh download` retrieves the immutable model snapshot and builds the ARM64 image from the local Dockerfile. `serve.sh start` mounts the snapshot read-only and runs vLLM on `127.0.0.1:18896`. A client posts chat requests to `/v1/chat/completions`. vLLM applies the Muse parser, schedules requests, manages the FP8 KV cache, and streams the target model's reasoning or answer as SSE.

The profile requests a 131,072-token model window, eight maximum sequences, FP8 E4M3 KV, and `0.42` GPU-memory utilization. vLLM reported 22.77 GiB of KV-cache budget and 3,219,558 token positions. These reported positions are an engine capacity figure under the selected profile, with actual admission depending on request shape. Measured decode was 12.4 tok/s for one stream and 61.2 tok/s aggregate for five. Fresh retrieval prompts reached their exact keys at 3,079 input tok/s for 32,742 tokens and 2,266 input tok/s for 124,985 tokens. Peak sampled GPU-process memory was 46.6 GiB. See [`serve.sh`](../serve.sh), [`benchmark.py`](../benchmark.py), and the [raw results](../results/benchmark-2026-11-21.json).

The request path is

```text
client on Spark
  -> localhost:18896 OpenAI API
  -> vLLM Muse parser, scheduler, and FP8 KV cache
  -> Muse Glimmer NVFP4 target-only decode
  -> streamed reasoning or answer
```

The benchmark uses 400-token decode waves with thinking enabled, temperature `0.7`, top-p `1`, and top-k `20`. The two retrieval runs use lower reasoning strength and different prompt lengths. They establish observed behavior for those requests. Docker reported no fixed container memory limit.
