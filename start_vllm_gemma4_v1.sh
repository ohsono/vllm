#!/bin/bash
# Start vLLM server for Gemma 4 31B Dense (SM 12.1 compatible)
#
# MODEL: google/gemma-4-31b-it
#   - Dense multimodal transformer (all 31B params active per token)
#   - Supports text + image input, text output; 140+ languages
#   - Context window: up to 256K tokens; limited below for KV cache control
#
# MEMORY PROFILE (bf16, no quantization):
#   Weights:  31B × 2 bytes            = ~62 GB
#   KV cache: scales with max-model-len and max-num-seqs
#   At 32K ctx, fp8 KV, 1 seq:        = ~4 GB
#   At 32K ctx, fp8 KV, 32 seqs:      = ~128 GB
#   Total (32 seqs, 32K, fp8 KV):     = ~190 GB — fits on B200 (192 GB), tight
#   Recommended: 16 seqs or reduce max-model-len for comfortable headroom
#
# KV CACHE LEVERS (in order of impact):
#   1. --max-model-len          : hard cap on sequence length; KV scales linearly
#   2. --kv-cache-dtype fp8     : halves KV memory vs bf16 (negligible quality loss)
#   3. --max-num-seqs           : caps concurrent sequences; reduces KV pressure
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

# Reduce memory fragmentation — helps when weights nearly fill GPU VRAM
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Set vLLM environment
export TIKTOKEN_RS_CACHE_DIR="/home/ohsono/vllm/tiktoken_encodings"

# FlashInfer settings
export FLASHINFER_LOGLEVEL=0       # suppress verbose flashinfer logs
export FLASHINFER_JIT_VERBOSE=0    # suppress JIT compilation output
export FLASHINFER_NVCC_THREADS=4   # parallel NVCC threads for JIT kernel compilation

# API configuration
export VLLM_API_KEY="${VLLM_API_KEY:-vllm-1vb742C4hgV9K6nLyBpfxRGGq}"

PORT="${1:-8000}"

# Prometheus metrics exporter
export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/vllm_prometheus_gemma4_dense_$$}"
mkdir -p "${PROMETHEUS_MULTIPROC_DIR}"
trap 'rm -rf "${PROMETHEUS_MULTIPROC_DIR}"' EXIT

echo "========================================================================"
echo "Starting vLLM - Gemma 4 31B Dense (SM 12.1 compatible)"
echo "========================================================================"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Port: $PORT"
echo "Metrics: http://0.0.0.0:${PORT}/metrics"
echo "========================================================================"

exec /home/ohsono/green/vllm/.venv/bin/python /home/ohsono/green/vllm/.venv/bin/vllm serve google/gemma-4-31b-it \
  --host 0.0.0.0 \
  --port "${PORT}" \
  \
  `# Expose both the short alias and the full HF path so clients can use either` \
  --served-model-name gemma-4-31b-it google/gemma-4-31b-it \
  \
  `# mxfp4 weight quantization (4-bit); reduces 31B from ~62 GB (bf16) to ~15-16 GB.` \
  `# Frees ~46 GB VRAM for KV cache on GB10 — enables 32-64 seqs at 65K context!` \
  `# Quality: minimal loss on instruction-following; excellent for dense transformers.` \
  --quantization mxfp4 \
  --dtype bfloat16 \
  \
  `# FlashInfer attention backend: required for FP8 KV cache and prefix caching on SM 12.1.` \
  `# Triton backend has PTX compilation issues on SM 12.1 — use FlashInfer instead.` \
  --attention-backend TRITON_ATTN \
  \
  `# FP8 KV cache halves KV memory vs bf16 with negligible quality loss.` \
  `# At 32K ctx: bf16 = ~8 GB, fp8 = ~4 GB per concurrent sequence.` \
  --kv-cache-dtype fp8 \
  \
  `# Gemma 4 supports up to 256K context natively. 32K covers most real workloads.` \
  `# Raise to 65536+ for long-context tasks; watch KV cache memory growth.` \
  --max-model-len 65536 \
  \
  `# 0.85 leaves 15% headroom for CUDA kernels and OS. Lower if you see OOM.` \
  --gpu-memory-utilization 0.80 \
  \
  `# Prefix caching reuses KV cache across requests with shared prefixes (e.g. system prompts).` \
  `# Required to enable the FlashInfer attention backend on SM 12.1.` \
  --enable-prefix-caching \
  \
  `# Chunked prefill: allows large prompts to be processed in chunks,` \
  `# preventing a single long prefill from blocking the entire batch.` \
  --enable-chunked-prefill \
  \
  `# 32 concurrent sequences with nvfp4 (15-16 GB weights) + fp8 KV.` \
  `# At 65K ctx: ~8 GB per seq × 32 = ~256 GB total — cap at 16 seqs for GB10 safety.` \
  `# Formula: free_vram = 121 - 16 = 105 GB; max_seqs ≈ 105 / (ctx_bytes_per_seq + overhead).` \
  `# Increase conservatively as you observe runtime memory usage.` \
  --max-num-seqs 16 \
  \
  --reasoning-parser gemma4 \
  \
  --tool-call-parser gemma4 \
  \
  --enable-auto-tool-choice \
  \
  `# 65536 is a good ceiling for 31B dense without quantization.` \
  --max-num-batched-tokens 65536 \
  \
  `# fastsafetensors: parallel tensor loading via mmap — faster cold start.` \
  --load-format fastsafetensors \
  \
  --api-key "${VLLM_API_KEY}" 2>&1 | tee -a /home/ohsono/vllm_gemma4_31b.log

  # Optional flags — uncomment by removing the leading # and adding \ to the line above:
  # --quantization mxfp4           # reduce weights to ~16 GB (mxfp4); use if VRAM is tight
  # --max-num-seqs 32              # raise concurrency after confirming KV headroom
  # --max-model-len 65536          # raise context if 32K isn't enough
  # --tensor-parallel-size 2       # shard model across 2 GPUs if single GPU is insufficient
  # --enforce-eager                # disable CUDA graph capture; saves ~1-2 GB VRAM
  # --cpu-offload-gb 8             # offload KV cache to Grace CPU memory (GB10/B200 unified pool)
