#!/bin/bash
# Start vLLM server for Gemma 4 26B-A4B MoE (SM 12.1 compatible)
#
# MODEL: google/gemma-4-26B-A4B-it
#   - Mixture of Experts: 26B total params, only ~4B active per token
#   - Supports text + image input, text output; 140+ languages
#   - Context window: up to 256K tokens; limited below for KV cache control
#   - MoE delivers ~31B-class quality at ~4B-class inference cost
#
# MEMORY PROFILE (bf16, no quantization):
#   Weights:  26B × 2 bytes        = ~52 GB  (all experts loaded, sparse activation)
#   KV cache: shaped by 4B-active attention layers — smaller than a full 26B dense
#   At 32K ctx, fp8 KV, 1 seq:    = ~2 GB
#   At 32K ctx, fp8 KV, 64 seqs:  = ~128 GB
#   Total (64 seqs, 32K, fp8 KV): = ~52 + ~128 GB = ~180 GB — fits on B200 (192 GB)
#
# KV CACHE LEVERS (in order of impact):
#   1. --max-model-len          : hard cap on sequence length; KV scales linearly
#   2. --kv-cache-dtype fp8     : halves KV memory vs bf16 (negligible quality loss)
#   3. --max-num-seqs           : MoE supports many more concurrent seqs than dense
#   4. --gpu-memory-utilization : fraction of VRAM reserved for model + KV cache

set -e

# Use green vllm Python environment
export PATH=/home/ohsono/green/vllm/.venv/bin:$PATH

# Set CUDA environment (CUDA 13.1 to match PyTorch)
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/targets/sbsa-linux/lib:$LD_LIBRARY_PATH

# Add PyTorch CUDA libraries to LD_LIBRARY_PATH
TORCH_LIB=$(/home/ohsono/green/vllm/.venv/bin/python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || true)
export LD_LIBRARY_PATH=${TORCH_LIB}:${LD_LIBRARY_PATH}

# Reduce memory fragmentation — helps with expert weight loading in MoE
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Set vLLM environment
# VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1 requires ENABLE_DSV3_FUSED_A_GEMM compiled in;
# not available in this build — leave unset to fall back to standard Triton MoE kernel.
export TIKTOKEN_RS_CACHE_DIR="/home/ohsono/vllm/tiktoken_encodings"

# FlashInfer settings
export FLASHINFER_LOGLEVEL=0       # suppress verbose flashinfer logs
export FLASHINFER_JIT_VERBOSE=0    # suppress JIT compilation output
export FLASHINFER_NVCC_THREADS=4   # parallel NVCC threads for JIT kernel compilation

# API configuration
export VLLM_API_KEY="${VLLM_API_KEY:-vllm-1vb742C4hgV9K6nLyBpfxRGGq}"

PORT="${1:-8000}"

# Prometheus metrics exporter
export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/vllm_prometheus_gemma4_moe_$$}"
mkdir -p "${PROMETHEUS_MULTIPROC_DIR}"
trap 'rm -rf "${PROMETHEUS_MULTIPROC_DIR}"' EXIT

echo "========================================================================"
echo "Starting vLLM - Gemma 4 26B-A4B MoE (SM 12.1 compatible)"
echo "========================================================================"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Port: $PORT"
echo "Metrics: http://0.0.0.0:${PORT}/metrics"
echo "========================================================================"

exec /home/ohsono/green/vllm/.venv/bin/python /home/ohsono/green/vllm/.venv/bin/vllm serve google/gemma-4-26B-A4B-it \
  --host 0.0.0.0 \
  --port "${PORT}" \
  \
  `# Expose both the short alias and the full HF path so clients can use either` \
  --served-model-name gemma-4-26B-A4B-it google/gemma-4-26B-A4B-it \
  \
  `# mxfp4 weight quantization (4-bit); reduces MoE 26B from ~52 GB (bf16) to ~13 GB.` \
  `# Frees ~26 GB VRAM on GB10 for KV concurrency.` \
  `# MoE + mxfp4 + fp8 KV = 128+ concurrent seqs at 32K context possible.` \
  --quantization mxfp4 \
  --dtype bfloat16 \
  \
  `# Triton attention backend: required because Gemma4 full-attention layers use` \
  `# global_head_dim=512, which FlashInfer does not support (max 256).` \
  --attention-backend TRITON_ATTN \
  \
  `# FP8 KV cache halves KV memory vs bf16. Critical for maximizing concurrency.` \
  `# MoE KV cache is shaped by active attention heads (~4B effective), not total 26B.` \
  --kv-cache-dtype fp8 \
  \
  `# Gemma 4 supports up to 256K context. 32K covers most real workloads.` \
  `# With 64 seqs at 32K fp8 KV: ~128 GB + 52 GB weights = ~180 GB on B200.` \
  `# Reduce max-model-len or max-num-seqs if you see OOM at startup.` \
  --max-model-len 32768 \
  \
  `# 0.90 — MoE weights (~52 GB) are smaller than dense 31B (~62 GB),` \
  `# leaving more VRAM headroom for KV cache. Raise only if OOM is not a concern.` \
  --gpu-memory-utilization 0.90 \
  \
  `# Prefix caching reuses KV cache across requests with shared prefixes.` \
  `# Required to enable the FlashInfer attention backend on SM 12.1.` \
  --enable-prefix-caching \
  \
  `# Chunked prefill: interleaves prefill chunks with decode steps for lower latency.` \
  `# Especially valuable for MoE where expert routing adds per-token overhead.` \
  --enable-chunked-prefill \
  \
  `# 64 concurrent sequences with nvfp4 (13 GB weights) + fp8 KV on GB10.` \
  `# Free VRAM: 121 - 13 = 108 GB. At 32K ctx, MoE KV ≈ ~1.5 GB per seq.` \
  `# 64 seqs × 1.5 GB = 96 GB + 12 GB headroom — excellent utilization.` \
  --max-num-seqs 64 \
  \
  `# 32768 matches max-model-len; allows a full sequence worth of tokens per scheduling step.` \
  --max-num-batched-tokens 32768 \
  \
  `# fastsafetensors: parallel tensor loading via mmap — faster cold start.` \
  --load-format fastsafetensors \
  \
  `# Tool calling: gemma4 parser extracts function calls from model output.` \
  --tool-call-parser gemma4 \
  \
  `# Enable auto tool choice so the model can decide when to call tools.` \
  --enable-auto-tool-choice \
  \
  `# Reasoning parser extracts <think>...</think> blocks into reasoning_content.` \
  --reasoning-parser gemma4 \
  \
  --api-key "${VLLM_API_KEY}" 2>&1 | tee -a /home/ohsono/vllm_gemma4_26b_moe.log

  # Optional flags — uncomment by removing the leading # and adding \ to the line above:
  # --quantization mxfp4           # reduce weights to ~13 GB (mxfp4); aggressive but effective
  # --max-num-seqs 128             # raise concurrency after confirming KV headroom
  # --max-model-len 65536          # raise context if 32K isn't enough; watch KV memory
  # --tensor-parallel-size 2       # shard across 2 GPUs for expert parallelism
  # --enforce-eager                # disable CUDA graph capture; saves ~1-2 GB VRAM
  # --cpu-offload-gb 8             # offload KV cache to Grace CPU memory (GB10/B200 unified pool)
