FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpulse0 \
    pulseaudio-utils \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && rm -rf /root/.cache/pip

COPY music_mcp.py .

CMD ["mcpo", "--port", "8001", "--", "python", "/app/music_mcp.py"]
