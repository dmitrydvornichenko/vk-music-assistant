FROM python:3.10

# parec/paplay для Pulse TCP; libportaudio2 нужен, чтобы lwake (sounddevice) мог импортироваться
# procps для pkill (остановка paplay)
RUN apt-get update && apt-get install -y \
    libpulse0 \
    libportaudio2 \
    pulseaudio-utils \
    procps \
    && rm -rf /var/lib/apt/lists/* /tmp/*

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install requests numpy local-wake silero-vad \
 && rm -rf /tmp/*

WORKDIR /app
COPY main.py openwebui_adapter.py /app/
CMD ["python","/app/main.py"]

