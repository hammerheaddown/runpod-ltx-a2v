FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TORCH_COMPILE_DISABLE=1 \
    TORCHDYNAMO_DISABLE=1 \
    LTX_MODELS_ROOT=/runpod-volume/models \
    LTX_LORAS_DIR=/runpod-volume/loras

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 curl ca-certificates build-essential \
        python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Torch — cu128 trio matching Lightricks HF Space.
RUN pip install --no-cache-dir \
        torch==2.8.0 torchvision torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128

# HF Space requirements (verbatim).
RUN pip install --no-cache-dir \
        transformers==4.57.6 \
        accelerate einops scipy av \
        scikit-image>=0.25.2 \
        flashpack==0.1.2 \
        matplotlib \
        huggingface_hub[cli]

# xformers needs --no-build-isolation to see the installed torch.
RUN pip install --no-cache-dir xformers==0.0.32.post2 --no-build-isolation

# Pin LTX-2 to the same commit the Lightricks Space uses, install editable
# with --force-reinstall --no-deps so torch/transformers stay intact.
ARG LTX_COMMIT_SHA=ae855f8538843825f9015a419cf4ba5edaf5eec2
RUN mkdir -p /opt/LTX-2 \
    && cd /opt/LTX-2 \
    && git init \
    && git remote add origin https://github.com/Lightricks/LTX-2.git \
    && git fetch --depth 1 origin ${LTX_COMMIT_SHA} \
    && git checkout ${LTX_COMMIT_SHA} \
    && pip install --no-cache-dir --force-reinstall --no-deps -e \
        /opt/LTX-2/packages/ltx-core \
        -e /opt/LTX-2/packages/ltx-pipelines

# RunPod SDK + handler deps.
RUN pip install --no-cache-dir runpod requests pillow

WORKDIR /workspace
COPY handler.py /workspace/handler.py

CMD ["python", "-u", "/workspace/handler.py"]
