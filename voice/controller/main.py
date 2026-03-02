"""
Wake word по WAV-эталонам, всё в Docker.
Микрофон и колонки — через Pulse TCP (parec/paplay), без sounddevice/PortAudio.

Запись эталонов на хосте:
  pip install local-wake
  lwake record wake_ref/1.wav --duration 2
  lwake record wake_ref/2.wav --duration 2
  lwake record wake_ref/3.wav --duration 2
Папку wake_ref монтируем в WAKE_REFERENCE_DIR.
"""
import os
import sys
import subprocess
import tempfile
import requests
import wave
import logging
import sqlite3
import time
import json
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps
from openwebui_adapter import OpenWebUIAdapter

# Настройка уровня логирования из переменной окружения
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("controller")
# Убираем поток DEBUG от numba/librosa (local-wake)
for _name in ("numba", "numba.core", "numba.core.ssa", "numba.core.ir", "librosa"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Аудио параметры (из env с дефолтами)
RATE = int(os.environ.get("AUDIO_RATE", "16000"))
BUFFER_SEC = float(os.environ.get("WAKE_BUFFER_SEC", "2.0"))  # для wake word
COMMAND_BUFFER_SEC = float(os.environ.get("COMMAND_BUFFER_SEC", "1.0"))  # для stop/wait (короче!)
SLIDE_SEC = float(os.environ.get("WAKE_SLIDE_SEC", "0.25"))
CHUNK_VAD = int(os.environ.get("VAD_CHUNK_SIZE", "512"))
VAD_WINDOW = int(os.environ.get("VAD_WINDOW_SIZE", "8000"))  # 0.5 сек для Silero VAD
SILENCE_CHUNKS = int(os.environ.get("VAD_SILENCE_CHUNKS", "30"))  # сколько чанков тишины = конец фразы
MIN_RECORDING_DURATION = float(os.environ.get("MIN_RECORDING_DURATION", "0.2"))  # мин длина записи
GRACE_PERIOD = float(os.environ.get("RECORDING_GRACE_PERIOD", "1.5"))  # не считать тишину сразу после wake word

# Wake word и команды
ref_dir = os.environ.get("WAKE_REFERENCE_DIR", "/app/wake_ref")
wake_threshold = float(os.environ.get("WAKE_THRESHOLD", "0.1"))
stop_ref_dir = os.environ.get("STOP_REF_DIR", "/app/stop_ref")  # "стоп" - выход из follow-up
wait_ref_dir = os.environ.get("WAIT_REF_DIR", "/app/wait_ref")  # "подожди" - пауза в follow-up
command_threshold = float(os.environ.get("COMMAND_THRESHOLD", "0.15"))  # порог для stop/wait
pulse_server = os.environ.get("PULSE_SERVER", "")

# Звуковые индикаторы
sounds_dir = os.environ.get("SOUNDS_DIR", "/app/sounds")
sound_start = os.path.join(sounds_dir, "start.wav")     # после wake word
sound_end = os.path.join(sounds_dir, "end.wav")         # после записи фразы
sound_exit = os.path.join(sounds_dir, "exit.wav")       # выход из follow-up

# LLM параметры
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M")
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "medium")  # low, medium, high (для gpt-oss)

# Ключевые слова для форсирования HIGH reasoning (если первое слово в запросе)
# Можно задать через env как список через запятую: "думай,подумай,размышляй"
HIGH_REASONING_KEYWORDS = os.environ.get("HIGH_REASONING_KEYWORDS", "думай,подумай").lower().split(",")
HIGH_REASONING_KEYWORDS = [w.strip() for w in HIGH_REASONING_KEYWORDS if w.strip()]  # убираем пробелы

# Follow-up mode (продолжение диалога без wake word)
FOLLOW_UP_ENABLED = os.environ.get("FOLLOW_UP_ENABLED", "true").lower() in ("1", "true", "yes")
FOLLOW_UP_TIMEOUT = float(os.environ.get("FOLLOW_UP_TIMEOUT", "7.0"))  # секунд после ответа

USE_TEXT_PREPROCESSOR = os.environ.get("USE_TEXT_PREPROCESSOR", "true").lower() in ("1", "true", "yes")

# SYSTEM_PROMPT: LLM сама готовит текст для TTS (препроцессор отключен)
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", """Ты русский голосовой ассистент. Отвечай кратко (1-2 абзаца) и только на русском языке.

КРИТИЧЕСКИ ВАЖНО:
1. Названия исполнителей и песен в JSON-действиях пиши ТОЧНО КАК СЛЫШАЛ (не транслитерируй!)
2. Без markdown (**, __, `), эмодзи, спецсимволов
3. Пиши как для устной речи

МУЗЫКА:
Когда пользователь просит включить музыку, верни JSON в формате:
{"action": "play_music", "artist": "имя исполнителя", "song": "название песни"}

В artist/song пиши названия ТОЧНО КАК УСЛЫШАЛ (латиницей если так сказали, кириллицей если так сказали).

Примеры:
- "включи кино" → {"action": "play_music", "artist": "кино", "song": ""}
- "включи Pink Floyd High Hopes" → {"action": "play_music", "artist": "Pink Floyd", "song": "High Hopes"}
- "включи музыку машина времени" → {"action": "play_music", "artist": "машина времени", "song": ""}

Если не уверен в исполнителе/песне - угадай по контексту. Всегда заполняй хотя бы artist или song.
""")

# Фразы для сброса контекста (новый разговор)
RESET_PHRASES = [
    "новый разговор", "забудь всё", "начали сначала", "сброс", "новый диалог",
    "clear", "new conversation", "reset",
]

def parse_llm_action(text: str):
    """Парсит JSON action из ответа LLM."""
    try:
        # Ищем JSON в ответе
        start = text.find('{"action":')
        if start == -1:
            return None
        
        # Ищем закрывающую скобку
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    json_str = text[start:i+1]
                    return json.loads(json_str)
        return None
    except:
        return None

# Open WebUI БД для истории чатов
WEBUI_DB_PATH = os.environ.get("WEBUI_DB_PATH", "/app/webui_data/webui.db")
USER_ID = os.environ.get("VOICE_USER_ID", "ec3ddb7c-ea1e-4672-a94d-7c92c9eab21e")  # elestrin

# Инициализация адаптера Open WebUI (автоматически загружает последний голосовой чат)
webui_adapter = OpenWebUIAdapter(WEBUI_DB_PATH, USER_ID)
log.info("Open WebUI DB: %s, user_id=%s, chat_id=%s", WEBUI_DB_PATH, USER_ID, webui_adapter.get_current_chat_id())

if not pulse_server:
    print("Задай PULSE_SERVER (например tcp:host.docker.internal:4713)", file=sys.stderr)
    sys.exit(1)

if not os.path.isdir(ref_dir) or not [f for f in os.listdir(ref_dir) if f.endswith(".wav")]:
    print(
        f"Папка с эталонами пуста или не найдена: {ref_dir}\n"
        "Запиши 3–4 WAV (lwake record wake_ref/1.wav) и смонтируй в WAKE_REFERENCE_DIR.",
        file=sys.stderr,
    )
    sys.exit(1)

# Загружаем эталоны (local-wake)
from lwake.listen import load_support_set
from lwake.features import extract_embedding_features, dtw_cosine_normalized_distance

support_set = load_support_set(ref_dir, method="embedding")
if not support_set:
    log.error("Не удалось загрузить эталоны из %s", ref_dir)
    sys.exit(1)

# Загружаем команды stop/wait (опционально)
stop_set = []
wait_set = []
if os.path.isdir(stop_ref_dir) and [f for f in os.listdir(stop_ref_dir) if f.endswith(".wav")]:
    stop_set = load_support_set(stop_ref_dir, method="embedding")
    log.info("Loaded %d stop command(s) from %s", len(stop_set), stop_ref_dir)
if os.path.isdir(wait_ref_dir) and [f for f in os.listdir(wait_ref_dir) if f.endswith(".wav")]:
    wait_set = load_support_set(wait_ref_dir, method="embedding")
    log.info("Loaded %d wait command(s) from %s", len(wait_set), wait_ref_dir)

vad_model = load_silero_vad()
env = {**os.environ, "PULSE_SERVER": pulse_server}

buffer_samples = int(BUFFER_SEC * RATE)  # для wake word
command_buffer_samples = int(COMMAND_BUFFER_SEC * RATE)  # для stop/wait
slide_samples = int(SLIDE_SEC * RATE)
slide_bytes = slide_samples * 2  # s16le

log.info("PULSE_SERVER=%s", pulse_server)
log.info("Wake ref dir=%s, threshold=%s", ref_dir, wake_threshold)
log.info("Loaded %d reference(s): %s", len(support_set), [f for f, _ in support_set])
log.info("Buffer=%d samples (%.2fs), slide=%d samples (%.2fs)", buffer_samples, BUFFER_SEC, slide_samples, SLIDE_SEC)
log.info("Chat mode: history kept, say one of %s to reset", RESET_PHRASES[:4])
log.info("Listening for wake word (distance < %s = trigger)...", wake_threshold)


def read_from_parec_until_silence(proc=None, on_ready_callback=None, timeout=None, use_grace_period=True):
    """Читает из parec кусками, пока VAD не зафиксирует тишину.
    Читаем маленькими порциями (CHUNK_VAD), но VAD проверяем на большем окне (VAD_WINDOW).
    
    Grace period: в первые GRACE_PERIOD секунд не считаем тишину (даём время начать говорить).
    
    Если proc=None, запускает parec сам. Если передан процесс - использует его.
    on_ready_callback: вызывается после того как parec инициализировался (прочитан первый чанк).
    timeout: если задан - возвращает пустой массив если за это время не было речи.
    use_grace_period: если False, grace period не используется (для follow-up режима).
    """
    import select
    
    if proc is None:
        proc = subprocess.Popen(
            ["parec", "--format=s16le", "--rate=%d" % RATE, "--channels=1", "--raw"],
            stdout=subprocess.PIPE,
            env=env,
        )
    audio_chunks = []
    silence = 0
    chunk_bytes = CHUNK_VAD * 2  # 512 сэмплов для чтения
    vad_buffer = np.array([], dtype=np.int16)  # Буфер для VAD окна
    total_samples = 0  # Счётчик записанных сэмплов
    grace_samples = int(GRACE_PERIOD * RATE)  # Сколько сэмплов в grace period
    ready_callback_called = False
    start_time = time.time()
    speech_detected = False  # Флаг что была хоть какая-то речь
    
    try:
        while True:
            # Проверка таймаута (для follow-up режима) - КАЖДЫЕ 0.1 сек!
            # Если 7 сек без речи → таймаут
            if timeout and (time.time() - start_time) > timeout:
                if not speech_detected:
                    log.debug("Recording timeout (%.1fs) without speech", timeout)
                    return np.array([], dtype=np.int16)  # Возвращаем пустой массив
                # Если речь была - продолжаем до тишины
            
            # Неблокирующий read с timeout 0.1 сек (чтобы проверять таймаут часто!)
            ready, _, _ = select.select([proc.stdout], [], [], 0.1)
            if not ready:
                continue  # Нет данных, идём проверять таймаут снова
            
            raw = proc.stdout.read(chunk_bytes)
            if len(raw) < chunk_bytes:
                break
            arr = np.frombuffer(raw, dtype=np.int16).copy()
            audio_chunks.append(arr)
            total_samples += len(arr)
            
            # После первого успешного чтения - parec инициализирован, можно играть звук
            if not ready_callback_called and on_ready_callback:
                on_ready_callback()
                ready_callback_called = True
            
            # Добавляем в VAD буфер
            vad_buffer = np.concatenate([vad_buffer, arr])
            
            # Проверяем VAD только если накопилось достаточно для окна
            if len(vad_buffer) >= VAD_WINDOW:
                # Берём последние VAD_WINDOW сэмплов для проверки
                vad_chunk = vad_buffer[-VAD_WINDOW:]
                speech = get_speech_timestamps(vad_chunk, vad_model, sampling_rate=RATE)
                
                # Отмечаем если была речь
                if speech:
                    if not speech_detected:
                        log.debug("VAD: speech detected at %.2fs (first detection)", time.time() - start_time)
                    speech_detected = True
                
                # Grace period: не считаем тишину в начале записи (только если включен)
                if use_grace_period and total_samples < grace_samples:
                    silence = 0  # Игнорируем тишину в grace period
                else:
                    # Считаем тишину ТОЛЬКО ЕСЛИ была речь (иначе таймаут сработает)
                    if speech_detected:
                        silence = 0 if speech else silence + 1
                        if silence > SILENCE_CHUNKS:
                            log.debug("VAD: %d chunks of silence after speech, ending recording", silence)
                            break
                
                # Оставляем только последние VAD_WINDOW сэмплов в буфере
                if len(vad_buffer) > VAD_WINDOW * 2:
                    vad_buffer = vad_buffer[-VAD_WINDOW:]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    out = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.int16)
    log.info("Recording done: %d samples (%.2fs), speech_detected=%s", len(out), len(out) / RATE, speech_detected)
    return out


def save_wav(data: np.ndarray, path: str) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data.tobytes())


def play_wav(path: str) -> None:
    """Синхронное воспроизведение WAV (для коротких звуков)."""
    subprocess.run(["paplay", path], env=env, check=True)

def play_wav_interruptible(path: str) -> str:
    """
    Воспроизведение WAV с возможностью прерывания командами stop/wait.
    Возвращает:
    - "finished": доиграло до конца
    - "stopped": прервано командой stop
    - "wait": прервано командой wait (пауза)
    """
    import select
    
    # Запускаем paplay в фоне
    log.debug("Starting paplay for file: %s", path)
    proc = subprocess.Popen(["paplay", path], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log.debug("paplay started with PID: %s", proc.pid)
    
    # Параллельно слушаем команды
    parec_proc = subprocess.Popen(
        ["parec", "--format=s16le", "--rate=%d" % RATE, "--channels=1", "--raw"],
        stdout=subprocess.PIPE,
        env=env,
    )
    
    chunk_bytes = CHUNK_VAD * 2
    command_buffer = np.zeros(command_buffer_samples, dtype=np.float32)
    last_command_time = 0
    loop_count = 0  # Счетчик итераций для дебага
    start_time = time.time()
    max_playback_time = 300  # 5 минут максимум (защита от зависаний)
    
    try:
        log.debug("Interruptible playback: starting parec listener loop, paplay pid=%s", proc.pid)
        while proc.poll() is None:  # Пока paplay работает
            loop_count += 1
            
            # Защита от бесконечного цикла
            elapsed = time.time() - start_time
            if elapsed > max_playback_time:
                log.error("Playback timeout after %.1fs (max %ds), killing paplay (pid %s)", elapsed, max_playback_time, proc.pid)
                proc.kill()
                break
            
            # Дебаг если слишком долго
            if loop_count == 100:  # ~10 сек
                poll_result = proc.poll()
                log.warning("Playback running for 10s, proc.poll()=%s, paplay pid=%s", poll_result, proc.pid)
            # Неблокирующий read с timeout
            ready, _, _ = select.select([parec_proc.stdout], [], [], 0.1)
            if not ready:
                if loop_count % 10 == 0:  # Каждую секунду
                    log.debug("Playback loop: no audio data (iteration %d)", loop_count)
                continue  # Нет данных, проверяем proc.poll() снова
            
            # Читаем аудио
            raw = parec_proc.stdout.read(chunk_bytes)
            if len(raw) < chunk_bytes:
                log.warning("Parec read incomplete: got %d bytes, expected %d", len(raw), chunk_bytes)
                break
            
            if loop_count % 50 == 0:  # Каждые ~5 сек при 0.1s timeout
                log.debug("Playback loop: processing audio (iteration %d)", loop_count)
            
            arr = np.frombuffer(raw, dtype=np.int16).copy()
            float_chunk = arr.astype(np.float32) / 32768.0
            command_buffer = np.roll(command_buffer, -len(float_chunk))
            command_buffer[-len(float_chunk):] = float_chunk
            
            # Проверяем команды (с cooldown 0.15s чтобы короткие команды не упустить)
            current_time = time.time()
            if (stop_set or wait_set) and (current_time - last_command_time > 0.15):
                try:
                    feats = extract_embedding_features(y=command_buffer, sample_rate=RATE)
                    if feats is not None:
                        # Stop
                        for filename, ref_feats in stop_set:
                            try:
                                d = dtw_cosine_normalized_distance(feats, ref_feats)
                                # Логируем если близко к порогу (в пределах 50% от порога)
                                if d < command_threshold * 1.5:
                                    log.debug("STOP check: %s distance %.4f (threshold %.4f, match=%s)", 
                                             filename, d, command_threshold, d < command_threshold)
                                if d < command_threshold:
                                    log.info("✓ STOP TRIGGERED: %s distance %.4f < %.4f", filename, d, command_threshold)
                                    proc.kill()
                                    parec_proc.kill()
                                    return "stopped"
                            except Exception as e:
                                log.debug("STOP check failed: %s", e)
                        # Wait
                        for filename, ref_feats in wait_set:
                            try:
                                d = dtw_cosine_normalized_distance(feats, ref_feats)
                                # Логируем если близко к порогу (в пределах 50% от порога)
                                if d < command_threshold * 1.5:
                                    log.debug("WAIT check: %s distance %.4f (threshold %.4f, match=%s)", 
                                             filename, d, command_threshold, d < command_threshold)
                                if d < command_threshold:
                                    log.info("✓ WAIT TRIGGERED: %s distance %.4f < %.4f", filename, d, command_threshold)
                                    proc.kill()
                                    parec_proc.kill()
                                    return "wait"
                            except Exception as e:
                                log.debug("WAIT check failed: %s", e)
                except Exception as e:
                    log.debug("Feature extraction failed: %s", e)
        
        final_poll = proc.poll()
        log.debug("Playback finished normally: proc.poll()=%s, iterations=%d, elapsed=%.1fs", 
                 final_poll, loop_count, time.time() - start_time)
        return "finished"
    finally:
        # Чистим процессы
        try:
            proc.kill()
        except:
            pass
        try:
            parec_proc.kill()
        except:
            pass
        log.debug("Interruptible playback: cleanup complete")

def play_notification(sound_type: str) -> None:
    """Воспроизведение звукового уведомления (если файл существует)."""
    sound_map = {
        "start": sound_start,
        "end": sound_end,
        "exit": sound_exit,
    }
    sound_path = sound_map.get(sound_type)
    if sound_path and os.path.exists(sound_path):
        try:
            # Играем в фоне, не блокируем
            log.info("🔊 Playing sound: %s (%s)", sound_type, sound_path)
            proc = subprocess.Popen(["paplay", sound_path], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log.debug("play_notification(%s): paplay started, pid=%s", sound_type, proc.pid)
        except Exception as e:
            log.error("play_notification(%s): failed: %s", sound_type, e)
    else:
        log.debug("play_notification(%s): sound file not found: %s", sound_type, sound_path)

def stop_audio_playback() -> None:
    """Останавливает все запущенные paplay процессы."""
    try:
        subprocess.run(["pkill", "-9", "paplay"], check=False)
        log.info("Stopped audio playback (killed paplay)")
    except:
        pass


def is_reset_phrase(text: str) -> bool:
    """Сброс только если вся фраза — одна из команд сброса (без лишних слов)."""
    t = text.lower().strip().rstrip(".,!?")
    return t in RESET_PHRASES


def get_adaptive_reasoning(text: str) -> str:
    """
    Определяет оптимальный уровень reasoning в зависимости от количества слов.
    
    Правила:
    - Первое слово "думай"/"подумай" (или из HIGH_REASONING_KEYWORDS) → HIGH (независимо от длины)
    - <= 5 слов → LOW
    - 5 < слов < 15 → MEDIUM
    - >= 15 слов → HIGH
    """
    # Проверяем первое слово на ключевые слова для HIGH reasoning
    words = text.split()
    if words:
        first_word = words[0].lower().rstrip(".,!?:;")  # убираем знаки препинания
        if first_word in HIGH_REASONING_KEYWORDS:
            log.info("Adaptive reasoning: keyword '%s' detected → HIGH (forced)", first_word)
            return "high"
    
    # Подсчёт слов
    word_count = len(words)
    
    if word_count <= 5:
        log.debug("Adaptive reasoning: %d words → LOW", word_count)
        return "low"
    elif word_count < 15:
        log.debug("Adaptive reasoning: %d words → MEDIUM", word_count)
        return "medium"
    else:
        log.debug("Adaptive reasoning: %d words → HIGH", word_count)
        return "high"


def get_chat_history():
    """Возвращает историю текущего чата."""
    return webui_adapter.get_history()


def on_wake_detected(skip_wake_detection=False, follow_up_timeout=None):
    """
    Обработка команды после wake word или в follow-up режиме.
    skip_wake_detection=True означает что мы в follow-up, wake word уже был.
    follow_up_timeout: таймаут для follow-up режима (если нет речи - вернёт "timeout").
    """
    if not skip_wake_detection:
        log.info("Wake word DETECTED — starting command recording")
        
        # СРАЗУ останавливаем музыку (если играет)
        try:
            requests.post("http://music:5003/stop", timeout=1)
            log.debug("Music stopped for new wake word")
        except:
            pass
        
        # Колбэк: играем звук ТОЛЬКО после инициализации parec
        audio = read_from_parec_until_silence(on_ready_callback=lambda: play_notification("start"))
    else:
        log.info("Follow-up mode — waiting for speech (timeout %.1fs)", follow_up_timeout or 0)
        # В follow-up режиме с таймаутом: пилик после инициализации
        # БЕЗ grace period! (иначе VAD детектирует эхо/шум и ждёт 4.26 сек вместо таймаута)
        audio = read_from_parec_until_silence(
            on_ready_callback=lambda: play_notification("start"),
            timeout=follow_up_timeout,
            use_grace_period=False
        )
    
    if audio.size == 0:
        if follow_up_timeout:
            log.info("Follow-up timeout - no speech detected")
            return "timeout"
        else:
            log.warning("No audio recorded, skipping")
            return "empty"
    
    # Проверка минимальной длины (фильтр ложных срабатываний)
    duration = len(audio) / RATE
    if duration < MIN_RECORDING_DURATION:
        log.warning("Recording too short (%.2fs < %.2fs), probably false wake or cut off too early, skipping", 
                    duration, MIN_RECORDING_DURATION)
        return "empty"
    
    wav_path = tempfile.mktemp(suffix=".wav")
    reply_path = tempfile.mktemp(suffix=".wav")
    try:
        save_wav(audio, wav_path)
        play_notification("end")  # Звук "записали, обрабатываем"
        log.info("STT: sending %d bytes to http://stt:5000/stt", os.path.getsize(wav_path))
        with open(wav_path, "rb") as f:
            r = requests.post("http://stt:5000/stt", files={"audio": f}, timeout=30)
        r.raise_for_status()
        text = r.json().get("text", "").strip()
        log.info("STT (user): %r", text or "(empty)")

        if not text:
            log.warning("Empty text from STT, skipping LLM/TTS")
            return "empty"
        

        # Сброс контекста по фразе: создаём новую сессию
        if is_reset_phrase(text):
            old_count = webui_adapter.get_message_count()
            webui_adapter.create_new_chat()
            log.info("Chat reset: old chat had %d msgs, new chat_id=%s", old_count, webui_adapter.get_current_chat_id())
            resp = "Окей, новый разговор."
        else:
            # Добавляем реплику пользователя в БД
            webui_adapter.add_message("user", text)
            # Загружаем историю из БД и шлём в /api/chat
            history = get_chat_history()
            messages = []
            if SYSTEM_PROMPT.strip():
                messages.append({"role": "system", "content": SYSTEM_PROMPT.strip()})
            messages.extend(history)
            log.info("LLM chat: %d history msgs + system, sending to http://llm:11434/api/chat", len(history))
            
            # Формируем payload для API запроса
            payload = {"model": LLM_MODEL, "messages": messages, "stream": False}
            # Для gpt-oss моделей добавляем reasoning effort
            if "gpt-oss" in LLM_MODEL.lower():
                # Используем адаптивный reasoning вместо фиксированного
                reasoning_level = get_adaptive_reasoning(text)
                payload["think"] = reasoning_level
                log.info("GPT-OSS adaptive reasoning: %s (text: %d words)", reasoning_level, len(text.split()))
            
            r = requests.post(
                "http://llm:11434/api/chat",
                json=payload,
                timeout=300,  # 5 минут: первая загрузка модели может занять 1-2 мин
            )
            r.raise_for_status()
            
            # Получаем ответ и reasoning (если есть)
            message = r.json().get("message") or {}
            resp = message.get("content", "")
            thinking = message.get("thinking", "")
            if thinking:
                log.debug("LLM reasoning trace (%d chars): %s...", len(thinking), thinking[:100])
            log.info("LLM (assistant): %d chars", len(resp))
            
            # Проверка на action (например play_music) - НЕ ЗАПУСКАЕМ музыку сразу!
            # Сначала ищем что включить и формируем ответ, но музыку запустим ПОСЛЕ TTS
            music_action = None
            action = parse_llm_action(resp)
            if action and action.get("action") == "play_music":
                log.info("Music action detected: %s", action)
                music_action = action  # Сохраняем для запуска ПОСЛЕ TTS
                try:
                    # Ищем трек НО НЕ ЗАПУСКАЕМ (только для формирования ответа)
                    music_r = requests.post(
                        "http://music:5003/search",
                        json={"artist": action.get("artist", ""), "song": action.get("song", "")},
                        timeout=15
                    )
                    if music_r.status_code == 200:
                        music_data = music_r.json()
                        # Используем правильные названия из VK (как они там хранятся)
                        resp = f"Включаю {music_data.get('artist', '')} {music_data.get('title', '')}"
                        log.info("Music found (will play after TTS): %s", resp)
                    else:
                        resp = "Не могу найти эту песню"
                        music_action = None  # Отменяем воспроизведение
                        log.warning("Music search failed: %s", music_r.text)
                except Exception as e:
                    log.exception("Music service unavailable: %s", e)
                    resp = "Музыкальный сервис недоступен"
                    music_action = None

        if not resp:
            log.warning("Empty LLM response, skipping TTS")
            return "empty"

        # Препроцессинг текста перед TTS (опционально)
        # ВАЖНО: финальный текст (после препроцессора) сохраняем в БД для дебага
        tts_text = resp
        if USE_TEXT_PREPROCESSOR:
            log.info("Text Preprocessor: sending %d chars to http://text_preprocessor:5000/preprocess", len(resp))
            try:
                r = requests.post(
                    "http://text_preprocessor:5000/preprocess",
                    json={"text": resp, "add_ssml": True},
                    timeout=60
                )
                r.raise_for_status()
                tts_text = r.json().get("processed_text", resp)
                log.info("Text Preprocessor: processed %d → %d chars", len(resp), len(tts_text))
            except Exception as e:
                log.warning("Text Preprocessor failed: %s, using original text", e)
                tts_text = resp
        
        # Сохраняем в БД финальный текст (который реально идет в TTS)
        webui_adapter.add_message("assistant", tts_text)
        total_msgs = webui_adapter.get_message_count()
        log.info("Saved to DB: %d chars, total %d msgs in chat", len(tts_text), total_msgs)

        log.info("TTS: sending %d chars to http://tts:5000/tts", len(tts_text))
        # TTS синтез занимает ~1 сек на 10 символов, +запас. Для 1000 символов = 100+ сек
        tts_timeout = max(120, len(tts_text) // 5)  # минимум 2 мин, или 1 сек на 5 символов
        r = requests.post("http://tts:5000/tts", json={"text": tts_text}, timeout=tts_timeout)
        r.raise_for_status()
        with open(reply_path, "wb") as f:
            f.write(r.content)
        log.info("TTS: %d bytes, playing (interruptible)", len(r.content))
        playback_result = play_wav_interruptible(reply_path)
        
        # ПОСЛЕ проигрывания ответа запускаем музыку (если была команда play_music)
        music_started = False
        if music_action and playback_result == "finished":
            log.info("Starting music playback after TTS finished")
            try:
                # Останавливаем предыдущую музыку (если играет)
                try:
                    requests.post("http://music:5003/stop", timeout=2)
                except:
                    pass
                
                music_r = requests.post(
                    "http://music:5003/play",
                    json={"artist": music_action.get("artist", ""), "song": music_action.get("song", "")},
                    timeout=15
                )
                if music_r.status_code == 200:
                    log.info("Music started successfully")
                    music_started = True
                else:
                    log.warning("Music playback failed: %s", music_r.text)
            except Exception as e:
                log.exception("Failed to start music: %s", e)
        
        if playback_result == "stopped":
            log.info("Playback stopped by user")
            return "stopped"
        elif playback_result == "wait":
            log.info("Playback paused by user")
            return "success"  # Пауза = продолжаем в follow-up
        elif music_started:
            log.info("Music playing - skipping follow-up")
            return "music_playing"  # Музыка играет - НЕ входим в follow-up
        else:
            log.info("Playback finished")
            return "success"  # Успешная обработка — входим в follow-up
    except requests.exceptions.RequestException as e:
        log.exception("API error: %s", e)
        return "error"
    except Exception as e:
        log.exception("Error: %s", e)
        return "error"
    finally:
        for p in (wav_path, reply_path):
            if os.path.exists(p):
                os.unlink(p)
def main():
    listen_cycle = 0
    while True:
        listen_cycle += 1
        log.info("Listen cycle %d: starting parec (wake word stream)", listen_cycle)
        proc = subprocess.Popen(
            ["parec", "--format=s16le", "--rate=%d" % RATE, "--channels=1", "--raw"],
            stdout=subprocess.PIPE,
            env=env,
        )
        audio_buffer = np.zeros(buffer_samples, dtype=np.float32)
        slide_count = 0

        try:
            while True:
                raw = proc.stdout.read(slide_bytes)
                if len(raw) < slide_bytes:
                    log.warning("Listen: short read %d < %d, restarting parec", len(raw), slide_bytes)
                    break
                slide_count += 1
                chunk = np.frombuffer(raw, dtype=np.int16)
                float_chunk = chunk.astype(np.float32) / 32768.0
                audio_buffer = np.roll(audio_buffer, -len(float_chunk))
                audio_buffer[-len(float_chunk) :] = float_chunk

                try:
                    feats = extract_embedding_features(y=audio_buffer, sample_rate=RATE)
                except Exception:
                    continue
                if feats is None:
                    continue

                for filename, ref_feats in support_set:
                    try:
                        d = dtw_cosine_normalized_distance(feats, ref_feats)
                        if d < wake_threshold:
                            log.info("Wake: %s distance %.4f < %.4f", filename, d, wake_threshold)
                            proc.terminate()
                            try:
                                proc.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                            
                            # Обработка команды
                            result = on_wake_detected(skip_wake_detection=False)
                            
                            # Follow-up mode после ответа (только если был успешный ответ или stopped)
                            # Если запустилась музыка (music_playing) - НЕ входим в follow-up
                            if result == "music_playing":
                                log.info("Music playing - returning to wake word mode without follow-up")
                                # Без звуков - музыка уже играет
                            elif FOLLOW_UP_ENABLED and result in ("success", "stopped"):
                                while True:
                                    # Сразу запускаем запись с таймаутом (пилик сыграет после инициализации parec)
                                    result2 = on_wake_detected(skip_wake_detection=True, follow_up_timeout=FOLLOW_UP_TIMEOUT)
                                    
                                    if result2 == "success":
                                        # Успешная обработка, продолжаем follow-up
                                        continue
                                    
                                    elif result2 == "music_playing":
                                        # Музыка играет - выход БЕЗ звуков
                                        log.info("Music playing - exiting follow-up without sound")
                                        break
                                    
                                    elif result2 == "timeout":
                                        # Таймаут - выход из follow-up
                                        log.info("Follow-up timeout → returning to wake word mode")
                                        play_notification("exit")
                                        break
                                    
                                    else:
                                        # Другие случаи (empty, error) - выход
                                        log.info("Follow-up ended: %s", result2)
                                        play_notification("exit")
                                        break
                            
                            break
                    except Exception:
                        continue
                else:
                    continue
                break
        except KeyboardInterrupt:
            log.info("Stopping...")
            break
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                proc.kill()


if __name__ == "__main__":
    main()
