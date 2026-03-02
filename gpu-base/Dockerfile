# CUDA 12.x + cuDNN: даёт libcublas.so.12 и libcudnn для faster-whisper (пакеты собраны под 12, не под 13)
# Берём последний 12.x с cuDNN для Ubuntu 22.04
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# Базовый образ с Python и общими либами для GPU-сервисов (stt, tts)
RUN apt update && apt install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg git \
 && rm -rf /var/lib/apt/lists/* /tmp/*

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install flask soundfile \
 && rm -rf /tmp/*
