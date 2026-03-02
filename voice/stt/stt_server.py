from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os
import tempfile
import logging

# Настройка уровня логирования из переменной окружения
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("stt")

# Приглушаем шумные библиотеки
for lib in ("httpx", "httpcore", "urllib3", "transformers", "faster_whisper"):
    logging.getLogger(lib).setLevel(logging.WARNING)

log.info("Loading Whisper large-v3...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
log.info("Whisper large-v3 loaded (~6GB VRAM)")

# Параметры распознавания из env
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "ru")
WHISPER_NO_SPEECH_THRESHOLD = float(os.environ.get("WHISPER_NO_SPEECH_THRESHOLD", "0.7"))
WHISPER_VAD_FILTER = os.environ.get("WHISPER_VAD_FILTER", "true").lower() in ("1", "true", "yes")

log.info("Whisper config: language=%s, no_speech_threshold=%.2f, vad_filter=%s", 
         WHISPER_LANGUAGE, WHISPER_NO_SPEECH_THRESHOLD, WHISPER_VAD_FILTER)

app = Flask(__name__)

@app.post("/stt")
def stt():
    try:
        log.debug("STT request: files=%s", list(request.files.keys()))
        if "audio" not in request.files:
            log.warning("STT: no audio file in request")
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files["audio"]
        if audio_file.filename == "":
            log.warning("STT: empty filename")
            return jsonify({"error": "Empty audio file"}), 400
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        size = os.path.getsize(tmp_path)
        log.debug("STT: saved %d bytes to %s, transcribing...", size, tmp_path)
        
        try:
            segments, info = model.transcribe(
                tmp_path,
                language=WHISPER_LANGUAGE,
                no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
                condition_on_previous_text=False,
                vad_filter=WHISPER_VAD_FILTER,
                vad_parameters={"threshold": 0.5} if WHISPER_VAD_FILTER else None
            )
            segments = list(segments)
            text = " ".join([s.text for s in segments]).strip()
            log.info("STT: language=%s, %d segments, text=%r", getattr(info, "language", "?"), len(segments), text[:200] if text else "(empty)")
            return jsonify({"text": text})
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        log.exception("STT error: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    log.info("Starting STT server on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)

