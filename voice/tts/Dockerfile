FROM vasyan-ai-gpu-base:latest

ENV PATH="/venv/bin:$PATH"

# Узкоспециализированные зависимости только для TTS
# ВАЖНО: также измени tts_server.py — там два варианта кода (XTTS и Silero)
# Роллбек: см. ROLLBACK_TTS.md

# === ВАРИАНТ 1: XTTS v2 (закомментирован, медленный но естественный голос) ===
# PyTorch <2.6: в 2.6+ torch.load по умолчанию weights_only=True, чекпоинты Coqui TTS не грузятся
# transformers <4.40: в 4.50+ GPT2InferenceModel потерял GenerationMixin (нет метода .generate)
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install TTS "transformers<4.40" "torch>=2.0,<2.6" \
#  && rm -rf /tmp/*

# === ВАРИАНТ 2: Silero TTS (активен, быстрый и правильный русский) ===
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchaudio omegaconf pydub soundfile scipy \
 && rm -rf /tmp/*

WORKDIR /app
COPY tts_server.py /app/
CMD ["python","/app/tts_server.py"]

