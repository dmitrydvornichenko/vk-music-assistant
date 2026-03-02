FROM vasyan-ai-gpu-base:latest

ENV PATH="/venv/bin:$PATH"

# Зависимости для RUNorm (нормализация текста для TTS)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install runorm torch \
 && rm -rf /tmp/*

# Патч для бага в runorm: token_type_ids может отсутствовать в новых версиях transformers
RUN sed -i 's/data\.pop("token_type_ids")/data.pop("token_type_ids", None)/' \
    /venv/lib/python3.10/site-packages/runorm/runorm.py

WORKDIR /app
COPY preprocessor_server.py /app/
CMD ["python", "/app/preprocessor_server.py"]
