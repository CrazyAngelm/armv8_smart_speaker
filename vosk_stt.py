import asyncio
import os
import json
import websockets
from vosk import Model, KaldiRecognizer
from typing import Optional
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Настройки WebSocket сервера
STT_WS_HOST = os.getenv("STT_WS_HOST", "0.0.0.0")
STT_WS_PORT = int(os.getenv("STT_WS_PORT", 8778))

# Путь к модели Vosk
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "models/vosk-model-small-ru-0.22")

# Энергетический порог и минимальная длительность речи (сек)
ENERGY_THRESHOLD = float(os.getenv("ENERGY_THRESHOLD", "0.005"))
MIN_SPEECH_DURATION = float(os.getenv("MIN_SPEECH_DURATION", "0.3"))
PCM_SAMPLE_RATE = int(os.getenv("PCM_SAMPLE_RATE", 16000))

# Класс для аудиосообщений
class AudioMsg:
    def __init__(self, raw: bytes, sr: int = PCM_SAMPLE_RATE):
        self.raw = raw
        self.sr = sr

# --- VAD: Проверка наличия речи по энергии ---
def detect_speech(audio_bytes: bytes, sample_rate: int) -> bool:
    # Преобразуем байты в numpy array
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    energy = np.mean(np.square(audio_array))
    frames_over_threshold = np.sum(np.square(audio_array) > ENERGY_THRESHOLD)
    min_speech_samples = int(MIN_SPEECH_DURATION * sample_rate)
    has_speech = energy > ENERGY_THRESHOLD and frames_over_threshold > min_speech_samples
    print(f"[VAD] Audio energy: {energy:.6f}, Threshold: {ENERGY_THRESHOLD}, Has speech: {has_speech}")
    return has_speech

# Функция распознавания речи через Vosk
async def stt_vosk(audio: AudioMsg) -> str:
    """
    Асинхронная функция распознавания речи через Vosk.
    Принимает AudioMsg (raw PCM 16kHz LE mono), возвращает строку.
    """
    if not os.path.exists(VOSK_MODEL_PATH):
        raise RuntimeError(f"Модель Vosk не найдена по пути: {VOSK_MODEL_PATH}")
    
    # VAD: Проверяем, содержит ли аудио речь
    if not detect_speech(audio.raw, audio.sr):
        return "Не удалось распознать речь"
    
    # Загружаем модель и создаем распознаватель
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, audio.sr)
    
    # Обрабатываем аудиоданные
    rec.AcceptWaveform(audio.raw)
    result = rec.FinalResult()
    
    # Парсим результат
    result_json = json.loads(result)
    recognized_text = result_json.get("text", "")
    
    # Если текст пустой, возвращаем сообщение об ошибке
    if not recognized_text:
        return "Не удалось распознать речь"
    
    return recognized_text

# Обработчик WebSocket для сервера STT
async def stt_ws_handler(ws):
    try:
        async for message in ws:
            if isinstance(message, bytes):
                audio = AudioMsg(message)
                try:
                    text = await stt_vosk(audio)
                    await ws.send(text)
                except Exception as e:
                    await ws.send(f"ERROR: {e}")
            else:
                await ws.send("ERROR: Only binary PCM messages supported")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"[STT WS] Connection closed: {e}")
    except Exception as e:
        print(f"[STT WS] Unexpected error: {e}")

# Основная функция для запуска WebSocket сервера
async def main_ws():
    print(f"[STT WS] Serving on ws://{STT_WS_HOST}:{STT_WS_PORT}")
    print(f"[STT WS] Using Vosk model: {VOSK_MODEL_PATH}")
    async with websockets.serve(stt_ws_handler, STT_WS_HOST, STT_WS_PORT, max_size=8*2**20, ping_interval=30, ping_timeout=30):
        await asyncio.Future()  # run forever

# Тестовая функция для прямого использования модуля
async def test_stt(pcm_file_path: str):
    with open(pcm_file_path, "rb") as f:
        raw = f.read()
    audio = AudioMsg(raw)
    try:
        text = await stt_vosk(audio)
        print("Распознанный текст:", text)
    except Exception as e:
        print(f"Ошибка: {e}")

# Запуск как основной скрипт
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "ws":
            # Запуск WebSocket сервера
            asyncio.run(main_ws())
        else:
            # Тестирование с указанным файлом
            asyncio.run(test_stt(sys.argv[1]))
    else:
        print("Использование:")
        print("  python vosk_stt.py ws - запуск WebSocket сервера")
        print("  python vosk_stt.py [файл.pcm] - тест распознавания из файла") 