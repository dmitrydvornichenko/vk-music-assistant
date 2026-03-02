FROM ai-gpu-base:latest

ENV PATH="/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchaudio omegaconf pydub soundfile scipy \
 && rm -rf /tmp/*

WORKDIR /app
COPY tts_server.py /app/
CMD ["python","/app/tts_server.py"]

