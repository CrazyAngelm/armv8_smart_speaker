#!/usr/bin/env python3
"""
Микрофонный сервис для ARMv8 (Debian).
Захватывает аудио и отправляет через WebSocket.
"""
import asyncio
import websockets
import numpy as np
import sounddevice as sd
import argparse

# Конфигурация
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
HOST = "0.0.0.0"
WS_PORT = 8765

# Глобальное состояние
audio_buffer = []
clients = set()
is_recording = True  # По умолчанию запись включена

# Функция обратного вызова для аудио
def audio_callback(indata, frames, time, status):
    if status:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] {status}")
    
    if is_recording and clients:  # Отправляем аудио только если есть подключенные клиенты
        audio_data = indata.copy()
        audio_buffer.append(audio_data)
        if len(audio_buffer) > 10:  # Ограничиваем размер буфера
            asyncio.run_coroutine_threadsafe(broadcast_audio(audio_buffer.pop(0)), asyncio.get_event_loop())

# WebSocket функции
async def ws_handler(websocket, path):
    """Обработчик WebSocket подключений"""
    clients.add(websocket)
    client_ip = websocket.remote_address[0]
    print(f"[ИНФО] Клиент подключен: {client_ip}")
    
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)
        print(f"[ИНФО] Клиент отключен: {client_ip}")

async def broadcast_audio(audio_data):
    """Отправка аудио данных всем подключенным клиентам"""
    if not clients:
        return
    
    # Конвертируем в байты для передачи
    data_bytes = audio_data.tobytes()
    
    # Отправляем всем подключенным клиентам
    disconnected = set()
    for client in clients:
        try:
            await client.send(data_bytes)
        except websockets.exceptions.ConnectionClosed:
            disconnected.add(client)
    
    # Удаляем отключенных клиентов
    for client in disconnected:
        clients.remove(client)

# Настройка и запуск микрофона
async def start_microphone(device=None):
    """Запуск захвата аудио с микрофона"""
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=audio_callback,
            blocksize=CHUNK,
            dtype='float32',
            device=device
        )
        with stream:
            print(f"[ИНФО] Микрофон запущен на устройстве {device if device else 'по умолчанию'}")
            print(f"[ИНФО] Частота дискретизации: {SAMPLE_RATE} Гц, каналов: {CHANNELS}")
            while True:
                await asyncio.sleep(1)
    except Exception as e:
        print(f"[ОШИБКА] Ошибка микрофона: {e}")
        print("[ИНФО] Доступные аудио устройства:")
        print(sd.query_devices())
        raise

def list_audio_devices():
    """Вывести список доступных аудио устройств ввода"""
    print("\nДоступные аудио устройства ввода:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"Индекс {i}: {dev['name']} (входов: {dev['max_input_channels']})")

async def main():
    # Разбор аргументов командной строки
    parser = argparse.ArgumentParser(description="Микрофонный сервис для ARMv8")
    parser.add_argument("--device", type=int, help="Индекс аудио устройства ввода")
    parser.add_argument("--list-devices", action="store_true", help="Показать список доступных аудио устройств")
    parser.add_argument("--port", type=int, default=WS_PORT, help="Порт WebSocket сервера")
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    # Запуск WebSocket сервера
    ws_server = await websockets.serve(ws_handler, HOST, args.port)
    print(f"[ИНФО] WebSocket сервер запущен на ws://{HOST}:{args.port}")
    
    # Запуск микрофона
    mic_task = asyncio.create_task(start_microphone(args.device))
    
    # Продолжать выполнение
    try:
        await asyncio.gather(mic_task)
    finally:
        ws_server.close()
        await ws_server.wait_closed()

if __name__ == "__main__":
    print("[ИНФО] Запуск микрофонного сервиса...")
    asyncio.run(main()) 