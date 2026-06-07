#!/bin/bash
# Start vLLM server for Gemma 4 31B Dense (SM 12.1 compatible)
#
# MODEL: nvidia/Gemma-4-31B-IT-NVFP4  (pre-quantized from google/gemma-4-31b-it)
#   - Dense multimodal transformer (all 31B params active per token)
#   - Supports text + image input, text output; 140+ languages
#   - Context window: up to 256K tokens; limited below for KV cache control
#
# MEMORY PROFILE (NVFP4, --quantization modelopt):
#   Weights:  ~15-16 GB (NVFP4 vs ~62 GB bf16)
#   KV cache: scales with max-model-len and max-num-seqs
#   At 49K ctx, fp8 KV, 1 seq:        = ~6 GB
#   At 49K ctx, fp8 KV, 8 seqs:       = ~48 GB
#   Total (8 seqs, 49K, fp8 KV):      = ~64 GB — fits on GB10 (87 GB usable) with headroom
#   Recommended: 8-16 seqs at 0.70 gpu-memory-utilization
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

exec /home/ohsono/green/vllm/.venv/bin/python /home/ohsono/green/vllm/.venv/bin/vllm serve nvidia/Gemma-4-31B-IT-NVFP4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  \
  `# Expose both the short alias and the full HF path so clients can use either` \
  --served-model-name gemma-4-31b-it nvidia/Gemma-4-31B-IT-NVFP4 \
  \
  `# NVFP4 pre-quantized checkpoint (nvidia/Gemma-4-31B-IT-NVFP4, ~15-16 GB).` \
  `# Using pre-quantized NVFP4 instead of runtime --quantization mxfp4 for reliability.` \
  `# prithivMLmods/gemma-4-31B-it-MXFP4 is a community MXFP4 alternative if needed.` \
  --quantization modelopt \
  --dtype bfloat16 \
  \
  `# Triton attention backend: Gemma4 full-attention layers use global_head_dim=512,` \
  `# which FlashInfer does not support (max head_dim 256) — must use TRITON_ATTN.` \
  --attention-backend TRITON_ATTN \
  \
  `# FP8 KV cache halves KV memory vs bf16 with negligible quality loss.` \
  `# At 32K ctx: bf16 = ~8 GB, fp8 = ~4 GB per concurrent sequence.` \
  --kv-cache-dtype fp8 \
  \
  `# Gemma 4 supports up to 256K context natively. Capped at 49152 (48K) here:` \
  `# at gpu_memory_utilization=0.80 the engine reports max supported = 51936 tokens.` \
  --max-model-len 49152 \
  \
  `# 0.70 = 84.7 GiB vLLM budget on 121 GiB unified memory (OS hard floor ~34 GiB).` \
  `# 0.80 targeted 96.8 GiB but only 87 GiB is truly available → OOM.` \
  --gpu-memory-utilization 0.70 \
  \
  `# Prefix caching reuses KV cache across requests with shared prefixes (e.g. system prompts).` \
  --enable-prefix-caching \
  \
  `# Chunked prefill: allows large prompts to be processed in chunks,` \
  `# preventing a single long prefill from blocking the entire batch.` \
  --enable-chunked-prefill \
  \
  `# 8 concurrent sequences at ctx=49152 with NVFP4 + fp8 KV.` \
  `# KV budget 65.2 GiB ÷ 7.89 GiB/seq (10 full×49152×16kv×512dim + 50 slide×1024×16kv×256dim) = 8.` \
  `# Increase to 12 if you reduce max-model-len to 32768 (5.39 GiB/seq → 12 seqs).` \
  --max-num-seqs 8 \
  \
  --reasoning-parser gemma4 \
  \
  --tool-call-parser gemma4 \
  \
  --enable-auto-tool-choice \
  \
  `# Match max-num-batched-tokens to max-model-len.` \
  --max-num-batched-tokens 49152 \
  \
  `# fastsafetensors: parallel tensor loading via mmap — faster cold start.` \
  --load-format fastsafetensors \
  \
  --api-key "${VLLM_API_KEY}" 2>&1 | tee -a /home/ohsono/vllm_gemma4_31b.log

  # Optional flags — uncomment by removing the leading # and adding \ to the line above:
  # --max-num-seqs 32              # raise concurrency after confirming KV headroom
  # --max-model-len 65536          # raise context; watch KV memory
  # --tensor-parallel-size 2       # shard model across 2 GPUs if single GPU is insufficient
  # --enforce-eager                # disable CUDA graph capture; saves ~1-2 GB VRAM
  # --cpu-offload-gb 8             # offload KV cache to Grace CPU memory (GB10/B200 unified pool)
  # NOTE: to use base model instead: swap model ID to google/gemma-4-31b-it
  #       and --quantization modelopt → --quantization mxfp4 (runtime quantize; less tested)
