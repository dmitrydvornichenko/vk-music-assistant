FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && playwright install --with-deps chromium \
 && rm -rf /root/.cache/pip

COPY auth_server.py .

EXPOSE 8080

CMD ["python", "/app/auth_server.py"]
