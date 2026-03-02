from flask import Flask, request, jsonify
import os
import tempfile
import logging
import gigaam

# Настройка логирования
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("stt")

# Приглушаем шумные библиотеки
for lib in ("httpx", "httpcore", "urllib3", "transformers", "gigaam"):
    logging.getLogger(lib).setLevel(logging.WARNING)

# Читаем параметры из окружения
MODEL_NAME = os.environ.get("MODEL_NAME", "v3_e2e_rnnt")
DEVICE = os.environ.get("DEVICE", None)  # может быть "cuda", "cpu" или None (авто)

log.info(f"Loading GigaAM model {MODEL_NAME} on device {DEVICE or 'auto'}...")
model = gigaam.load_model(MODEL_NAME, device=DEVICE)
log.info("GigaAM model loaded successfully.")

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
            # Вызов GigaAM transcribe
            text = model.transcribe(tmp_path)
            log.info("STT: text=%r", text[:200] if text else "(empty)")
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