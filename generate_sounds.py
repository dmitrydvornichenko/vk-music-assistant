#!/usr/bin/env python3
"""
Генератор звуковых уведомлений для голосового ассистента.
Создаёт мягкие "ding" звуки в папке sounds/

Запуск: python generate_sounds.py
"""
import numpy as np
import wave
import os

RATE = 48000  # Частота дискретизации
sounds_dir = "sounds"

def generate_sine_tone(frequency, duration, fade_ms=50):
    """Генерирует синусоидальный тон с fade in/out."""
    samples = int(RATE * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    
    # Синус
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Fade in/out для мягкости
    fade_samples = int(RATE * fade_ms / 1000)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    
    # Нормализация и конвертация в int16
    audio = (audio * 0.3 * 32767).astype(np.int16)  # 30% громкости
    
    return audio

def save_wav(audio, filename):
    """Сохраняет аудио в WAV файл."""
    filepath = os.path.join(sounds_dir, filename)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio.tobytes())
    print(f"Created: {filepath}")

def main():
    os.makedirs(sounds_dir, exist_ok=True)
    
    # start.wav - начало записи (короткий восходящий тон)
    freq1 = generate_sine_tone(800, 0.08, fade_ms=20)
    freq2 = generate_sine_tone(1200, 0.08, fade_ms=20)
    start_sound = np.concatenate([freq1, freq2])
    save_wav(start_sound, "start.wav")
    
    # end.wav - конец записи (короткий нисходящий тон)
    freq1 = generate_sine_tone(1200, 0.08, fade_ms=20)
    freq2 = generate_sine_tone(800, 0.08, fade_ms=20)
    end_sound = np.concatenate([freq1, freq2])
    save_wav(end_sound, "end.wav")
    
    # exit.wav - выход из follow-up (одиночный низкий тон)
    exit_sound = generate_sine_tone(600, 0.15, fade_ms=30)
    save_wav(exit_sound, "exit.wav")
    
    print("\nЗвуковые файлы созданы в папке sounds/")
    print("Примонтируй эту папку в контейнер controller:")
    print("  volumes:")
    print("    - ./sounds:/app/sounds:ro")

if __name__ == "__main__":
    main()
