#!/usr/bin/env python3
"""
Сервис распознавания речи для ARMv8 (Debian).
Получает аудио через WebSocket и преобразует в текст с помощью Vosk.
"""
import asyncio
import websockets
import json
import os
import argparse
from vosk import Model, KaldiRecognizer

# Конфигурация
HOST = "0.0.0.0"
STT_WS_PORT = 8778
MIC_WS_URI = "ws://localhost:8765"

# Путь к модели Vosk
VOSK_MODEL_PATH = os.path.join("models", "vosk-model-small-ru-0.22")

# Глобальное состояние
text_buffer = []
is_processing = False

# Обработка аудио с помощью Vosk
async def process_audio(audio_data):
    """Обработка аудио данных и распознавание речи"""
    global is_processing, text_buffer
    
    if is_processing:
        return None
    
    try:
        is_processing = True
        
        # Проверяем наличие модели
        if not os.path.exists(VOSK_MODEL_PATH):
            print(f"[ОШИБКА] Модель Vosk не найдена: {VOSK_MODEL_PATH}")
            return None
        
        # Загружаем модель и создаем распознаватель
        model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(model, 16000)
        
        # Обрабатываем аудио
        recognizer.AcceptWaveform(audio_data)
        result = recognizer.FinalResult()
        
        # Разбор результата
        result_json = json.loads(result)
        recognized_text = result_json.get("text", "")
        
        if recognized_text:
            print(f"[РАСПОЗНАНО] {recognized_text}")
            text_buffer.append(recognized_text)
            # Хранить только последние 5 текстов
            if len(text_buffer) > 5:
                text_buffer.pop(0)
            return recognized_text
        
        return None
    except Exception as e:
        print(f"[ОШИБКА] Ошибка распознавания: {e}")
        return None
    finally:
        is_processing = False

# Подключение к микрофонному WebSocket
async def connect_to_mic():
    """Подключение к микрофонному сервису и обработка входящих аудио данных"""
    reconnect_delay = 5  # секунд между попытками переподключения
    
    while True:
        try:
            print(f"[ИНФО] Подключение к микрофонному сервису: {MIC_WS_URI}")
            async with websockets.connect(MIC_WS_URI, max_size=2**22) as websocket:
                print("[ИНФО] Подключено к микрофонному сервису")
                
                async for message in websocket:
                    if isinstance(message, bytes):
                        # Обработка аудио данных
                        await process_audio(message)
                    else:
                        print(f"[ИНФО] Получено текстовое сообщение: {message}")
        
        except Exception as e:
            print(f"[ОШИБКА] Ошибка подключения к микрофону: {e}")
            print(f"[ИНФО] Повторная попытка через {reconnect_delay} секунд...")
            await asyncio.sleep(reconnect_delay)

# Внешний STT WebSocket сервер
async def stt_handler(websocket, path):
    """Обработчик внешних WebSocket подключений для STT"""
    client_ip = websocket.remote_address[0]
    print(f"[ИНФО] STT клиент подключен: {client_ip}")
    
    try:
        # Отправляем начальный статус
        status = {
            "status": "ready",
            "model": VOSK_MODEL_PATH
        }
        await websocket.send(json.dumps(status))
        
        # Обрабатываем входящие сообщения
        async for message in websocket:
            if isinstance(message, bytes):
                # Обработка аудио данных
                recognized_text = await process_audio(message)
                result = {"text": recognized_text} if recognized_text else {"text": ""}
                await websocket.send(json.dumps(result))
            else:
                # Обработка команд
                try:
                    cmd = json.loads(message)
                    if cmd.get("command") == "get_last_texts":
                        await websocket.send(json.dumps({"texts": text_buffer}))
                    else:
                        await websocket.send(json.dumps({"error": "Неизвестная команда"}))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "Неверный формат JSON"}))
    
    except websockets.exceptions.ConnectionClosedError:
        print(f"[ИНФО] STT клиент отключен: {client_ip}")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка WebSocket: {e}")

async def main():
    global VOSK_MODEL_PATH, MIC_WS_URI
    # Разбор аргументов командной строки
    parser = argparse.ArgumentParser(description="Сервис распознавания речи для ARMv8")
    parser.add_argument("--model", type=str, default=VOSK_MODEL_PATH, help="Путь к модели Vosk")
    parser.add_argument("--mic-uri", type=str, default=MIC_WS_URI, help="URI WebSocket микрофонного сервиса")
    parser.add_argument("--port", type=int, default=STT_WS_PORT, help="Порт STT WebSocket сервера")
    args = parser.parse_args()
    
    # Обновляем глобальные настройки
    VOSK_MODEL_PATH = args.model
    MIC_WS_URI = args.mic_uri
    
    print(f"[ИНФО] Используется модель Vosk: {VOSK_MODEL_PATH}")
    print(f"[ИНФО] Микрофонный сервис: {MIC_WS_URI}")
    
    # Проверка наличия модели
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Модель Vosk не найдена: {VOSK_MODEL_PATH}")
        print("[ПРЕДУПРЕЖДЕНИЕ] Распознавание речи не будет работать!")
    
    # Запуск STT WebSocket сервера
    stt_server = await websockets.serve(stt_handler, HOST, args.port, max_size=2**22)
    print(f"[ИНФО] STT WebSocket сервер запущен на ws://{HOST}:{args.port}")
    
    # Подключение к микрофонному сервису
    mic_task = asyncio.create_task(connect_to_mic())
    
    # Продолжать выполнение
    try:
        await asyncio.gather(mic_task)
    finally:
        stt_server.close()
        await stt_server.wait_closed()

if __name__ == "__main__":
    print("[ИНФО] Запуск сервиса распознавания речи...")
    asyncio.run(main()) 