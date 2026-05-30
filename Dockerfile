ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

ARG TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6;8.9;9.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    MAX_JOBS=8 \
    MPLBACKEND=Agg \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/medgs
COPY . /opt/medgs

RUN python -m pip install --upgrade pip setuptools wheel && \
    grep -v -E '^(torch|torchvision)$' requirements.txt > /tmp/requirements-docker.txt && \
    python -m pip install -r /tmp/requirements-docker.txt && \
    python -m pip install --no-build-isolation \
        ./submodules/diff-gaussian-rasterization \
        ./submodules/simple-knn

COPY docker/entrypoint.sh /usr/local/bin/medgs-entrypoint
RUN chmod +x /usr/local/bin/medgs-entrypoint

ENV PYTHONPATH=/workspace:/opt/medgs
WORKDIR /opt/medgs
ENTRYPOINT ["medgs-entrypoint"]
CMD ["bash"]
