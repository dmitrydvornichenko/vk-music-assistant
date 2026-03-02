import os
import logging
import re
from flask import Flask, request, jsonify
from runorm import RUNorm

# Настройка уровня логирования из переменной окружения
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("text_preprocessor")

# Приглушаем шумные библиотеки
for lib in ("httpx", "httpcore", "urllib3", "transformers", "transformers.tokenization_utils", "transformers.modeling_utils"):
    logging.getLogger(lib).setLevel(logging.WARNING)

# Отключаем прогресс-бары и предупреждения от transformers
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # critical, error, warning, info, debug
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ==================== Загрузка RUNorm для нормализации текста ====================
MODEL_SIZE = os.environ.get("PREPROCESSOR_MODEL_SIZE", "small")  # small, medium, big
DEVICE = "cuda:0"  # Первая видяха (8GB)
# Путь к кешу HuggingFace моделей (стандартный путь для transformers)
HF_CACHE_DIR = os.environ.get("HF_HOME", "/root/.cache/huggingface")

log.info(f"Loading RUNorm normalizer: {MODEL_SIZE} on {DEVICE}")
log.info(f"HuggingFace cache: {HF_CACHE_DIR}")

normalizer = RUNorm()
normalizer.load(model_size=MODEL_SIZE, device=DEVICE, workdir=HF_CACHE_DIR)

log.info(f"RUNorm loaded successfully")

# ==================== Post-processing: дочистка после RUNorm ====================

# Словарь цифр для озвучивания (RUNorm иногда не трогает цифры)
DIGIT_TO_TEXT = {
    '0': 'ноль', '1': 'один', '2': 'два', '3': 'три', '4': 'четыре',
    '5': 'пять', '6': 'шесть', '7': 'семь', '8': 'восемь', '9': 'девять'
}

def replace_remaining_digits(text: str) -> str:
    """
    Заменяет оставшиеся цифры на текст (RUNorm может пропустить).
    Пример: "GDDR6x" → "GDDRшестьx"
    """
    return ''.join(DIGIT_TO_TEXT.get(char, char) for char in text)

# Словарь латинских букв → русские названия
# Словарь для дочистки оставшихся латинских букв (RUNorm иногда пропускает)
LATIN_TO_RUSSIAN = {
    'A': 'эй', 'B': 'би', 'C': 'си', 'D': 'ди', 'E': 'и',
    'F': 'эф', 'G': 'джи', 'H': 'эйч', 'I': 'ай', 'J': 'джей',
    'K': 'кей', 'L': 'эл', 'M': 'эм', 'N': 'эн', 'O': 'оу',
    'P': 'пи', 'Q': 'кью', 'R': 'ар', 'S': 'эс', 'T': 'ти',
    'U': 'ю', 'V': 'ви', 'W': 'дабл-ю', 'X': 'икс', 'Y': 'уай', 'Z': 'зет',
    'a': 'эй', 'b': 'би', 'c': 'си', 'd': 'ди', 'e': 'и',
    'f': 'эф', 'g': 'джи', 'h': 'эйч', 'i': 'ай', 'j': 'джей',
    'k': 'кей', 'l': 'эл', 'm': 'эм', 'n': 'эн', 'o': 'оу',
    'p': 'пи', 'q': 'кью', 'r': 'ар', 's': 'эс', 't': 'ти',
    'u': 'ю', 'v': 'ви', 'w': 'дабл-ю', 'x': 'икс', 'y': 'уай', 'z': 'зет'
}

def transliterate_remaining_latin(text: str) -> str:
    """
    Заменяет оставшиеся латинские буквы на их русские названия.
    Работает по словам - если слово содержит латиницу, заменяем побуквенно через дефис.
    Пример: "RTX" → "ар-ти-икс", "Ti" → "ти-ай", "4060 Ti" → "4060 ти-ай"
    """
    def process_word(word):
        # Собираем латинские буквы подряд и заменяем их
        letters = []
        result = []
        
        for char in word:
            if char.isalpha() and char.isascii():  # Латинская буква
                letters.append(LATIN_TO_RUSSIAN.get(char, char))
            else:
                # Не буква - сбрасываем накопленные буквы
                if letters:
                    result.append('-'.join(letters))
                    letters = []
                result.append(char)
        
        # Не забыть про буквы в конце
        if letters:
            result.append('-'.join(letters))
        
        return ''.join(result)
    
    # Обрабатываем каждое слово
    result = []
    for word in re.findall(r'\S+|\s+', text):
        if re.search(r'[a-zA-Z]', word):
            result.append(process_word(word))
        else:
            result.append(word)
    
    return ''.join(result)

# ==================== Flask API ====================
app = Flask(__name__)

@app.post("/preprocess")
def preprocess_text():
    try:
        if not request.json or "text" not in request.json:
            return jsonify({"error": "No text provided"}), 400
        
        original_text = request.json["text"]
        
        log.debug(f"Preprocessing text: {len(original_text)} chars")
        
        # Удаляем markdown ДО нормализации
        text_no_markdown = re.sub(r'\*\*(.+?)\*\*', r'\1', original_text)  # **bold**
        text_no_markdown = re.sub(r'__(.+?)__', r'\1', text_no_markdown)   # __bold__
        text_no_markdown = re.sub(r'\*(.+?)\*', r'\1', text_no_markdown)   # *italic*
        text_no_markdown = re.sub(r'_(.+?)_', r'\1', text_no_markdown)     # _italic_
        text_no_markdown = re.sub(r'`(.+?)`', r'\1', text_no_markdown)     # `code`
        
        # RUNorm делает всё: нормализует числа, кириллизирует латиницу, озвучивает аббревиатуры
        processed_text = normalizer.norm(text_no_markdown)
        
        # Дочищаем оставшиеся цифры (которые RUNorm не озвучил)
        processed_text = replace_remaining_digits(processed_text)
        
        # Дочищаем оставшуюся латиницу (которую RUNorm пропустил)
        processed_text = transliterate_remaining_latin(processed_text)
        
        log.debug(f"Processed: {len(processed_text)} chars")
        log.debug(f"Original: {original_text[:100]}...")
        log.debug(f"Processed: {processed_text[:100]}...")
        
        return jsonify({
            "original_text": original_text,
            "processed_text": processed_text
        })
        
    except Exception as e:
        log.exception(f"Preprocessing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": f"RUNorm-{MODEL_SIZE}"})

if __name__ == "__main__":
    log.info("Starting Text Preprocessor server on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
