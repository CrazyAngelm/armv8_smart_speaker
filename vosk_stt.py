import asyncio
import os
import json
import websockets
from vosk import Model, KaldiRecognizer
from typing import Optional
from dotenv import load_dotenv
import numpy as np
import webrtcvad

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

# --- Глобальная загрузка модели Vosk (один раз) ---
model = Model(VOSK_MODEL_PATH)

# --- WebRTC-VAD ---
vad = webrtcvad.Vad(2)  # 0-3, где 3 — самая агрессивная фильтрация
VAD_FRAME_MS = 30  # длина одного фрейма для VAD (10, 20 или 30 мс)

# Класс для аудиосообщений
class AudioMsg:
    def __init__(self, raw: bytes, sr: int = PCM_SAMPLE_RATE):
        self.raw = raw
        self.sr = sr

# --- VAD: Проверка наличия речи через WebRTC-VAD ---
def detect_speech(audio_bytes: bytes, sample_rate: int) -> bool:
    # WebRTC-VAD работает с 16-бит PCM, 8/16/32 кГц, моно
    frame_size = int(sample_rate * VAD_FRAME_MS / 1000) * 2  # 2 байта на сэмпл
    num_frames = len(audio_bytes) // frame_size
    speech_frames = 0
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        frame = audio_bytes[start:end]
        if len(frame) < frame_size:
            break
        if vad.is_speech(frame, sample_rate):
            speech_frames += 1
    min_speech_frames = max(1, int(0.3 * 1000 / VAD_FRAME_MS))  # минимум 0.3 сек речи
    has_speech = speech_frames >= min_speech_frames
    print(f"[VAD] Speech frames: {speech_frames}, Min required: {min_speech_frames}, Has speech: {has_speech}")
    return has_speech

# Функция распознавания речи через Vosk
async def stt_vosk(audio: AudioMsg) -> str:
    """
    Асинхронная функция распознавания речи через Vosk.
    Принимает AudioMsg (raw PCM 16kHz LE mono), возвращает строку.
    """
    # VAD: Проверяем, содержит ли аудио речь
    if not detect_speech(audio.raw, audio.sr):
        return "Не удалось распознать речь"
    # Используем глобальную модель
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
    async with websockets.serve(
        stt_ws_handler, 
        STT_WS_HOST, 
        STT_WS_PORT, 
        max_size=8*2**20, 
        ping_interval=300,   # 5 минут
        ping_timeout=None):  # Без таймаута
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