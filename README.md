# Muse Glimmer 30B on one DGX Spark

A pinned vLLM fork serves the RedHatAI NVFP4 checkpoint with speculation disabled. Hugging Face returned a regional HTTP 403 for the upstream DFlash draft. The model API binds to `127.0.0.1:18896`.

[How it works](docs/how-it-works.md)

[Benchmarks](#benchmarks) · [Runtime](#runtime) · [Quick start](#quick-start) · [Raw results](results/benchmark-2026-11-21.json)

## Benchmarks

Measured November 21, 2026 on one NVIDIA GB10 DGX Spark. The benchmark sent direct streaming requests to the localhost model API. Other model servers were paused.

### Decode, prose

SparkDash hash-map prompt with thinking on, temperature 0.7, top-p 1, top-k 20 and 400 generated tokens per request. Each row shows the fastest aggregate decode wave across three repetitions. Per-request decode and TTFT are medians from the same wave. TTFT counts the first streamed reasoning or answer token.

| Streams | Best aggregate tok/s | Per request tok/s | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 12.4 | 12.4 | 341 ms |
| 2 | 24.6 | 12.3 | 386 ms |
| 3 | 36.8 | 12.3 | 435 ms |
| 4 | 49.1 | 12.3 | 433 ms |
| 5 | 61.2 | 12.3 | 440 ms |

### Fresh prompt prefill

The input rate divides prompt tokens, minus any reported cache hits, by client TTFT. Cache-hit counts were unavailable in the streaming response. The benchmark uses an independent synthetic key for each context size and compares the response with that key.

| Prompt tokens | Input tok/s to first token | TTFT | Retrieval |
| ---: | ---: | ---: | --- |
| 32,742 | 3,079 | 10.64 s | Exact key |
| 124,985 | 2,266 | 55.15 s | Exact key |

## Runtime

- RedHatAI NVFP4 target at `403f780a4d35194b7230bc2d809effa3c147f407`.
- Native 131,072-token context, 8 maximum sequences, FP8 E4M3 KV cache and a 0.42 GPU memory target.
- The Dockerfile pins the Muse parser fork and CUDA 13 base wheel. Prose runs use `reasoning_strength=high`. Fresh-prompt runs use `low`.
- vLLM reported 22.77 GiB for the FP8 KV cache and 3,219,558 token positions.
- Peak GPU memory was 46.6 GiB across sampled inference-container processes. Docker reported no fixed container memory limit.

## Quick start

Use a single DGX Spark with Docker, NVIDIA Container Toolkit, the Hugging Face CLI, Python 3 and curl. Allow about 22 GiB for model files plus room for the runtime image. Stop any existing model server before running this recipe.

```bash
./serve.sh download
./serve.sh start
docker logs -f spark-catalog-muse-target-only
```

When the localhost API is ready, run the direct benchmark on the Spark.

```bash
curl -fsS http://127.0.0.1:18896/v1/models
export SPARK_BENCHMARK_URL=http://127.0.0.1:18896
export SPARK_BENCHMARK_MODEL=Muse-Glimmer-30B
python3 benchmark.py > decode-local.json
python3 benchmark.py 32768 2026144868 > prefill-32768-local.json
python3 benchmark.py 125000 2026237100 > prefill-125000-local.json
./serve.sh stop
```

This recipe builds on [Mia AI Lab's pinned deployment](https://github.com/MiaAI-Lab/Muse-Glimmer-30B-DGX-Spark-RTX-5090-6000-PRO/tree/1b71dc011010a3e2c373eafd8a41f12c7c7913fc). Our scripts, documentation and synthetic fixtures use the [MIT license](LICENSE). Upstream models and runtime components retain their own terms.
