# ==============================================================================
# Ultra-Optimized BS-RoFormer Serverless Worker
# Architecture: Band-Split RoPE Transformer (ByteDance / UVR)
# SOTA Models: BS-RoFormer ViperX 12.97 SDR & Mel-Band RoFormer Kim
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    TORCHAUDIO_USE_SOUND_FILE=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# 1. Install system tools, ffmpeg, build packages, and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# 2. Install PyTorch with CUDA 12.4 explicitly (prevents CPU-only wheel fallback)
RUN python3 -m pip install --no-cache-dir \
    torch==2.4.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. Install worker dependencies
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# 4. Pre-download SOTA BS-RoFormer models into image layer
# Baking models directly eliminates cold-start model downloads at runtime (Zero Cold-Start!)
RUN mkdir -p /app/models && \
    python3 -c "from audio_separator.separator import Separator; s = Separator(model_file_dir='/app/models'); s.load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt'); s.load_model('mel_band_roformer_kim_ft_unfrozen.ckpt')"

# 5. Copy handler and assets
COPY handler.py /app/handler.py
COPY README.md /app/README.md

CMD ["python3", "-u", "/app/handler.py"]
