#!/bin/bash
# Start vLLM server for Gemma 4 26B-A4B MoE (SM 12.1 compatible)
#
# MODEL: nvidia/Gemma-4-26B-A4B-NVFP4  (pre-quantized from google/gemma-4-26B-A4B-it)
#   - Mixture of Experts: 26B total params, only ~4B active per token
#   - Supports text + image input, text output; 140+ languages
#   - Context window: up to 256K tokens; limited below for KV cache control
#   - MoE delivers ~31B-class quality at ~4B-class inference cost
#   - NOTE: runtime --quantization mxfp4 CRASHES on this MoE (vllm#39000)
#
# MEMORY PROFILE (NVFP4, --quantization modelopt):
#   Weights:  ~16.5 GB (NVFP4 vs ~52 GB bf16; 52 tok/s on DGX Spark)
#   KV cache: shaped by 4B-active attention layers — smaller than a full 26B dense
#   At 32K ctx, fp8 KV, 1 seq:    = ~2 GB
#   At 32K ctx, fp8 KV, 32 seqs:  = ~64 GB
#   Total (32 seqs, 32K, fp8 KV): = ~16.5 + ~64 GB = ~80.5 GB — fits on GB10 (87 GB usable)
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

exec /home/ohsono/green/vllm/.venv/bin/python /home/ohsono/green/vllm/.venv/bin/vllm serve nvidia/Gemma-4-26B-A4B-NVFP4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  \
  `# Expose both the short alias and the full HF path so clients can use either` \
  --served-model-name gemma-4-26B-A4B-it nvidia/Gemma-4-26B-A4B-NVFP4 \
  \
  `# NVFP4 pre-quantized checkpoint (nvidia/Gemma-4-26B-A4B-NVFP4, ~16.5 GB).` \
  `# Runtime --quantization mxfp4 on this MoE is BROKEN (vllm#39000): 2D weight tensor` \
  `# vs MXFP4 expecting 3D (num_experts, out, in) → crashes during weight loading.` \
  `# modelopt loads the pre-quantized NVFP4 weights directly without re-quantizing.` \
  --quantization modelopt \
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
  `# With 32 seqs at 32K fp8 KV: ~64 GB + 16.5 GB weights = ~80.5 GB — fits on GB10.` \
  `# Reduce max-model-len or max-num-seqs if you see OOM at startup.` \
  --max-model-len 32768 \
  \
  `# 0.70 = 84.7 GiB vLLM budget on 121 GiB unified memory (OS hard floor ~34 GiB).` \
  `# NVFP4 weights ~16.5 GiB; 0.90 targeted 108.9 GiB → 21.9 GiB overcommit → OOM.` \
  --gpu-memory-utilization 0.70 \
  \
  `# Prefix caching reuses KV cache across requests with shared prefixes.` \
  `# Required to enable the FlashInfer attention backend on SM 12.1.` \
  --enable-prefix-caching \
  \
  `# Chunked prefill: interleaves prefill chunks with decode steps for lower latency.` \
  `# Especially valuable for MoE where expert routing adds per-token overhead.` \
  --enable-chunked-prefill \
  \
  `# 32 concurrent sequences with NVFP4 (~16.5 GB weights) + fp8 KV on GB10.` \
  `# KV budget: 84.7 - 16.5 - 3.5 = 64.7 GB. At 32K ctx, MoE KV ≈ ~2 GB per seq.` \
  `# 32 seqs × 2 GB = 64 GB — fits within budget.` \
  --max-num-seqs 32 \
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
  # --max-num-seqs 64              # raise after confirming KV headroom at runtime
  # --max-model-len 65536          # raise context if 32K isn't enough; watch KV memory
  # --tensor-parallel-size 2       # shard across 2 GPUs for expert parallelism
  # --enforce-eager                # disable CUDA graph capture; saves ~1-2 GB VRAM
  # --cpu-offload-gb 8             # offload KV cache to Grace CPU memory (GB10/B200 unified pool)
  # WARNING: --quantization mxfp4 on google/gemma-4-26B-A4B-it CRASHES (vllm#39000).
  #          Use nvidia/Gemma-4-26B-A4B-NVFP4 + --quantization modelopt (this script).
