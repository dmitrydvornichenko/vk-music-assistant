import os
import tempfile
import logging
from flask import Flask, request, send_file, jsonify, Response

# Настройка уровня логирования из переменной окружения
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("tts")

# Приглушаем шумные библиотеки
for lib in ("httpx", "httpcore", "urllib3", "torch", "omegaconf"):
    logging.getLogger(lib).setLevel(logging.WARNING)

# ==================== ВАРИАНТ 1: XTTS v2 (закомментирован, медленный) ====================
# # Принять лицензию Coqui без интерактива
# os.environ["COQUI_TOS_AGREED"] = "1"
# from TTS.api import TTS
# 
# log.info("Loading XTTS v2...")
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
# log.info("XTTS loaded")
# 
# DEFAULT_SPEAKER = "Ana Florence"
# DEFAULT_LANGUAGE = "ru"
# ==================== КОНЕЦ ВАРИАНТА 1 ====================

# ==================== ВАРИАНТ 2: Silero TTS (активен, быстрый) ====================
import torch
import torchaudio
import soundfile as sf

log.info("Loading Silero TTS...")

device = torch.device('cuda')
torch.set_num_threads(4)

# Загружаем модель через torch.hub (v5 - новая версия с лучшим качеством)
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v5_ru'
)
model.to(device)

# Доступные голоса v5: xenia (женский, естественный), aidar (мужской), baya, kseniya, eugene
# v5 поддерживает SSML, автоматические ударения и омографы
DEFAULT_SPEAKER = os.environ.get("TTS_SPEAKER", "xenia")
SAMPLE_RATE = 48000

log.info("Silero TTS v5 loaded, speaker=%s", DEFAULT_SPEAKER)
# ==================== КОНЕЦ ВАРИАНТА 2 ====================

app = Flask(__name__)

@app.post("/tts")
def speak():
    output_path = None
    try:
        log.debug("TTS request: json keys=%s", list(request.json.keys()) if request.json else None)
        if not request.json or "text" not in request.json:
            log.warning("TTS: no text in request")
            return jsonify({"error": "No text provided"}), 400
        
        text = request.json["text"]
        if not text or not text.strip():
            log.warning("TTS: empty text")
            return jsonify({"error": "Empty text"}), 400
        
        # ==================== ВАРИАНТ 1: XTTS (закомментирован) ====================
        # speaker = request.json.get("speaker") or DEFAULT_SPEAKER
        # language = request.json.get("language") or DEFAULT_LANGUAGE
        # log.info("TTS: text=%d chars, speaker=%s, language=%s", len(text), speaker, language)
        # 
        # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        #     output_path = tmp_file.name
        # 
        # kwargs = {"text": text, "file_path": output_path, "language": language}
        # if request.json.get("speaker_wav"):
        #     kwargs["speaker_wav"] = request.json["speaker_wav"]
        # else:
        #     kwargs["speaker"] = speaker
        # 
        # log.info("TTS: synthesizing to %s...", output_path)
        # tts.tts_to_file(**kwargs)
        # size = os.path.getsize(output_path)
        # log.info("TTS: synthesized %d bytes", size)
        # 
        # def generate():
        #     try:
        #         with open(output_path, 'rb') as f:
        #             data = f.read(1024)
        #             while data:
        #                 yield data
        #                 data = f.read(1024)
        #     finally:
        #         if os.path.exists(output_path):
        #             os.unlink(output_path)
        # 
        # return Response(generate(), mimetype="audio/wav", headers={
        #     "Content-Disposition": "inline; filename=output.wav"
        # })
        # ==================== КОНЕЦ ВАРИАНТА 1 ====================
        
        # ==================== ВАРИАНТ 2: Silero TTS (активен) ====================
        speaker = request.json.get("speaker") or DEFAULT_SPEAKER
        log.debug("TTS: text=%d chars, speaker=%s", len(text), speaker)
        
        # Текст должен быть уже обработан через text_preprocessor
        # Здесь только чистый синтез речи
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
        
        log.debug("TTS: synthesizing with Silero v5...")
        
        # Синтез (поддерживает SSML из коробки в v5)
        audio = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=SAMPLE_RATE
        )
        
        # Сохраняем в WAV (используем soundfile вместо torchcodec)
        audio_np = audio.cpu().numpy()
        sf.write(output_path, audio_np, SAMPLE_RATE)
        
        size = os.path.getsize(output_path)
        log.info("TTS: synthesized %d bytes", size)
        
        def generate():
            try:
                with open(output_path, 'rb') as f:
                    data = f.read(1024)
                    while data:
                        yield data
                        data = f.read(1024)
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)
        
        return Response(generate(), mimetype="audio/wav", headers={
            "Content-Disposition": "inline; filename=output.wav"
        })
        # ==================== КОНЕЦ ВАРИАНТА 2 ====================
        
    except Exception as e:
        log.exception("TTS error: %s", e)
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    log.info("Starting TTS server on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)

