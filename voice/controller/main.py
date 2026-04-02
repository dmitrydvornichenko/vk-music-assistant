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
import time
import json
import numpy as np
from vad_onnx import load_silero_vad, get_speech_timestamps
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

# LLM: OpenAI-совместимый API (llama.cpp, local server и т.д.)
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss-20b-Q4_K_M.gguf")

# MCP (music-mcp): те же инструменты что и cli.py
MCPO_URL = os.environ.get("MCPO_URL", "http://music-mcp:8001")
MAX_LLM_ITERS = int(os.environ.get("MAX_LLM_ITERS", "6"))

# URLs голосовых сервисов
STT_URL  = os.environ.get("STT_URL",  "http://stt:5000")
TTS_URL  = os.environ.get("TTS_URL",  "http://tts:5000")
PREP_URL = os.environ.get("PREP_URL", "http://text_preprocessor:5000")

# Follow-up mode (продолжение диалога без wake word)
FOLLOW_UP_ENABLED = os.environ.get("FOLLOW_UP_ENABLED", "true").lower() in ("1", "true", "yes")
FOLLOW_UP_TIMEOUT = float(os.environ.get("FOLLOW_UP_TIMEOUT", "7.0"))  # секунд после ответа

USE_TEXT_PREPROCESSOR = os.environ.get("USE_TEXT_PREPROCESSOR", "true").lower() in ("1", "true", "yes")

# System prompt: тот же что в cli.py — инструменты MCP (search_tracks, play_music, stop_music, ...)
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", """You are a voice assistant. Treat all input like it was spoken.
There might be mistakes in speech-to-text, mind it.
You have tools for vk.com (vkontakte) social network.
Use tools to handle all music-related requests.
Do NOT ask for confirmation, do NOT show intermediate results — just call the right tool. Reply in Russian, briefly (1-2 sentences for TTS).

## Playing a specific song or track
SONG/TRACK request → search_tracks(query), then play_music with the single best result. Always pass access_key to play_music when search returned it.
Prefer results where main_artists matches the requested artist.

## Playing an album
ALBUM request → search_album(artist + album name). From the first result use owner_id, album_id, access_key and call play_album(owner_id, album_id, access_key). Add shuffle=True if user wants shuffled.

## User's music library
- Own music (включи мою музыку) → play_user_audio()
- Shuffled → play_user_audio(shuffle=True)
- Another user → find_user("Name") then play_user_audio(owner_id=...)

## Playback control
- Pause (пауза) → pause_music()
- Resume (продолжи) → resume_music()
- Stop (стоп) → stop_music()
- Volume up/down → volume_up() / volume_down()

## Queue
- Show queue (очередь) → get_queue()
- Skip (следующий, пропусти) → skip_tracks(count=1)

## Account
- Who am I → whoami()
- Find user → find_user("Name Surname")

No markdown, no emoji. Reply only in Russian, short phrase for voice.""")

RESET_PHRASES = [
    "новый разговор", "забудь всё", "начали сначала", "сброс", "новый диалог",
    "clear", "new conversation", "reset",
]

# Open WebUI БД для истории чатов
WEBUI_DB_PATH = os.environ.get("WEBUI_DB_PATH", "/app/webui_data/webui.db")
USER_ID = os.environ.get("VOICE_USER_ID", "ec3ddb7c-ea1e-4672-a94d-7c92c9eab21e")

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

# Инициализация адаптера Open WebUI — после всех проверок окружения
# Если БД недоступна (отдельный compose), работает в памяти без персистентности
webui_adapter = OpenWebUIAdapter(WEBUI_DB_PATH, USER_ID)
log.info("Chat history: db=%s available=%s chat_id=%s", WEBUI_DB_PATH, webui_adapter._db_available, webui_adapter.get_current_chat_id())

# Схема инструментов MCP — загружается один раз при старте ниже после объявления функций
_mcp_tools: list = []


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


def get_chat_history():
    """Возвращает историю текущего чата."""
    return webui_adapter.get_history()


# ---------------------------------------------------------------------------
# MCP tools (как в cli.py)
# ---------------------------------------------------------------------------

def fetch_tools() -> list:
    """Загружает инструменты из mcpo OpenAPI и кэширует в _mcp_tools."""
    global _mcp_tools
    resp = requests.get(f"{MCPO_URL}/openapi.json", timeout=10)
    resp.raise_for_status()
    schema = resp.json()
    tools = []
    for path, path_item in schema.get("paths", {}).items():
        op = path_item.get("post")
        if not op:
            continue
        name = path.lstrip("/")
        description = op.get("description") or op.get("summary") or ""
        body = op.get("requestBody", {})
        json_schema = (
            body.get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
        if "$ref" in json_schema:
            ref = json_schema["$ref"].lstrip("#/").split("/")
            node = schema
            for part in ref:
                node = node[part]
            json_schema = node
        parameters = {
            "type": "object",
            "properties": json_schema.get("properties", {}),
        }
        if json_schema.get("required"):
            parameters["required"] = json_schema["required"]
        tools.append({
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        })
    _mcp_tools = tools
    log.info("MCP tools loaded: %s", [t["function"]["name"] for t in tools])
    return tools


def get_tools() -> list:
    """Возвращает кэшированный список инструментов. Загружает при первом вызове."""
    if _mcp_tools:
        return _mcp_tools
    return fetch_tools()


def tool_request(name: str, arguments: dict) -> str:
    """Вызов инструмента MCP."""
    resp = requests.post(f"{MCPO_URL}/{name}", json=arguments, timeout=60)
    resp.raise_for_status()
    return json.dumps(resp.json(), ensure_ascii=False)


def llm_request(messages: list, tools: list) -> dict:
    """OpenAI-совместимый запрос к LLM (llama.cpp и др.)."""
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    resp = requests.post(
        f"{LLM_URL}/chat/completions",
        headers=headers,
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# Инструменты MCP, которые запускают воспроизведение музыки
_MUSIC_PLAY_TOOLS = {"play_music", "play_album", "play_user_audio"}


def run_llm_with_tools() -> tuple:
    """
    Цикл: LLM + вызов инструментов (как cli.py).
    Возвращает (response_text: str, music_started: bool).
    История уже содержит последнее сообщение пользователя (добавлено перед вызовом).
    """
    tools = get_tools()
    history = get_chat_history()
    messages = []
    if SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": SYSTEM_PROMPT.strip()})
    messages.extend(history)

    music_started = False

    for i in range(MAX_LLM_ITERS):
        log.info("LLM request %d/%d (%d messages)", i + 1, MAX_LLM_ITERS, len(messages))
        data = llm_request(messages, tools)
        choice = data["choices"][0]
        msg = choice["message"]
        reason = choice.get("finish_reason", "?")

        assistant_turn = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            assistant_turn["tool_calls"] = msg["tool_calls"]
        messages.append(assistant_turn)

        if reason != "tool_calls" or not msg.get("tool_calls"):
            return (msg.get("content") or "").strip(), music_started

        for tc in msg["tool_calls"]:
            func = tc["function"]
            name = func["name"]
            args = json.loads(func["arguments"])
            log.info("Tool call: %s(%s)", name, args)
            if name in _MUSIC_PLAY_TOOLS:
                music_started = True
            try:
                result = tool_request(name, args)
            except Exception as e:
                result = json.dumps({"error": str(e)})
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
    log.warning("Max LLM iterations reached")
    return (messages[-1].get("content") or "").strip(), music_started


def on_wake_detected(skip_wake_detection=False, follow_up_timeout=None):
    """
    Обработка команды после wake word или в follow-up режиме.
    skip_wake_detection=True означает что мы в follow-up, wake word уже был.
    follow_up_timeout: таймаут для follow-up режима (если нет речи - вернёт "timeout").
    """
    if not skip_wake_detection:
        log.info("Wake word DETECTED — starting command recording")
        # Останавливаем музыку через MCP (если играет)
        try:
            tool_request("stop_music", {})
            log.debug("Music stopped for new wake word")
        except Exception:
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
        log.info("STT: sending %d bytes to %s/stt", os.path.getsize(wav_path), STT_URL)
        with open(wav_path, "rb") as f:
            r = requests.post(f"{STT_URL}/stt", files={"audio": f}, timeout=30)
        r.raise_for_status()
        text = r.json().get("text", "").strip()
        log.info("STT (user): %r", text or "(empty)")

        if not text:
            log.warning("Empty text from STT, skipping LLM/TTS")
            return "empty"

        # Сброс контекста по фразе
        if is_reset_phrase(text):
            old_count = webui_adapter.get_message_count()
            webui_adapter.create_new_chat()
            log.info("Chat reset: old chat had %d msgs, new chat_id=%s", old_count, webui_adapter.get_current_chat_id())
            resp = "Окей, новый разговор."
            music_started = False
        else:
            webui_adapter.add_message("user", text)
            resp, music_started = run_llm_with_tools()
            log.info("LLM (assistant): %d chars, music_started=%s", len(resp or ""), music_started)

        # Музыка запущена — не озвучиваем и не пищим, просто играет
        if music_started:
            log.info("Music started — skipping TTS and notifications")
            webui_adapter.add_message("assistant", resp or "")
            return "music_playing"

        if not resp:
            log.warning("Empty LLM response, skipping TTS")
            return "empty"

        # Звук "записали, обрабатываем" — только для разговорных ответов
        play_notification("end")

        # Препроцессинг текста перед TTS
        tts_text = resp
        if USE_TEXT_PREPROCESSOR:
            log.info("Text Preprocessor: sending %d chars to %s", len(resp), PREP_URL)
            try:
                r = requests.post(
                    f"{PREP_URL}/preprocess",
                    json={"text": resp, "add_ssml": True},
                    timeout=60
                )
                r.raise_for_status()
                tts_text = r.json().get("processed_text", resp)
                log.info("Text Preprocessor: %d → %d chars", len(resp), len(tts_text))
            except Exception as e:
                log.warning("Text Preprocessor failed: %s, using original text", e)
                tts_text = resp

        # Сохраняем оригинальный ответ LLM в историю, а не препроцессированный
        webui_adapter.add_message("assistant", resp)
        log.info("Saved to DB: %d msgs in chat", webui_adapter.get_message_count())

        log.info("TTS: sending %d chars to %s", len(tts_text), TTS_URL)
        tts_timeout = max(120, len(tts_text) // 5)
        r = requests.post(f"{TTS_URL}/tts", json={"text": tts_text}, timeout=tts_timeout)
        r.raise_for_status()
        with open(reply_path, "wb") as f:
            f.write(r.content)
        log.info("TTS: %d bytes, playing (interruptible)", len(r.content))
        playback_result = play_wav_interruptible(reply_path)
        if playback_result == "stopped":
            log.info("Playback stopped by user")
            return "stopped"
        if playback_result == "wait":
            log.info("Playback paused by user")
            return "success"
        log.info("Playback finished")
        return "success"
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
                            
                            result = on_wake_detected(skip_wake_detection=False)
                            # follow-up только для разговорных ответов; музыка — без follow-up
                            if FOLLOW_UP_ENABLED and result in ("success", "stopped"):
                                while True:
                                    result2 = on_wake_detected(skip_wake_detection=True, follow_up_timeout=FOLLOW_UP_TIMEOUT)
                                    if result2 == "success":
                                        continue
                                    if result2 == "music_playing":
                                        log.info("Music started in follow-up — returning to wake word mode")
                                        break
                                    if result2 == "timeout":
                                        log.info("Follow-up timeout → returning to wake word mode")
                                        play_notification("exit")
                                        break
                                    else:
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
    log.info("Loading MCP tools from %s ...", MCPO_URL)
    try:
        fetch_tools()
    except Exception as e:
        log.warning("Could not load MCP tools at startup: %s (will retry on first request)", e)
    main()
