FROM ai-gpu-base:latest

ENV PATH="/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install faster-whisper \
 && rm -rf /tmp/*

WORKDIR /app
COPY stt_server.py /app/
CMD ["python","/app/stt_server.py"]

