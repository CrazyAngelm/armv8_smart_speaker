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
import logging
import threading
import time
import webrtcvad
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env если существует
env_path = Path('.env')
if env_path.exists():
    load_dotenv()

# Конфигурация
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 480  # Оптимальный размер для WebRTCVAD (30 мс)
HOST = "0.0.0.0"
WS_PORT = 8765
MAX_BUFFER_SIZE = 10  # Максимальный размер буфера
VAD_MODE = 3  # Агрессивность VAD (0-3, где 3 - максимальная)
SPEECH_START_THRESHOLD = 3  # Количество фреймов речи для начала передачи
SILENCE_THRESHOLD_FRAMES = 20  # ~0.6 секунды тишины для завершения передачи

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

audio_buffer = []
clients = set()
buffer_lock = threading.Lock()  # Блокировка для безопасного доступа к буферу
is_recording = True  # По умолчанию запись включена
is_sending = False   # Флаг отправки для избежания одновременных отправок
vad = None           # Инициализация VAD

# Статистика VAD
in_speech = False
speech_frames = 0
silence_frames = 0
speaking_buffer = []

def init_vad():
    """Инициализация VAD (Voice Activity Detection)"""
    global vad
    vad = webrtcvad.Vad()
    vad.set_mode(VAD_MODE)
    logging.info(f"VAD инициализирован с агрессивностью {VAD_MODE}")

# WebSocket функции
async def ws_handler(websocket):
    """Обработчик WebSocket подключений"""
    clients.add(websocket)
    client_ip = websocket.remote_address[0]
    logging.info(f"Клиент подключен: {client_ip}")
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)
        logging.info(f"Клиент отключен: {client_ip}")

async def broadcast_audio(audio_data):
    """Отправка аудио данных всем подключенным клиентам"""
    global is_sending
    
    if not clients:
        return
    
    is_sending = True
    try:
        data_bytes = audio_data.tobytes()
        
        # Отправляем данные всем клиентам
        disconnected = set()
        for client in clients:
            try:
                await client.send(data_bytes)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logging.error(f"Ошибка отправки аудио: {e}")
                disconnected.add(client)
        
        # Удаляем отключенных клиентов
        for client in disconnected:
            clients.remove(client)
        
        if disconnected:
            logging.info(f"Отключено клиентов: {len(disconnected)}, осталось: {len(clients)}")
    finally:
        is_sending = False

def list_audio_devices():
    """Вывести список доступных аудио устройств ввода"""
    logging.info("Доступные аудио устройства ввода:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            logging.info(f"Индекс {i}: {dev['name']} (входов: {dev['max_input_channels']})")

# Функция обработки речи с VAD
def process_vad(audio_frame):
    """Обработка аудио с помощью VAD для определения речи"""
    global in_speech, speech_frames, silence_frames, speaking_buffer
    
    try:
        # Преобразуем float32 в int16 для VAD
        frame_bytes = audio_frame.tobytes()
        
        is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
        
        if not in_speech:
            if is_speech:
                speech_frames += 1
                speaking_buffer.append(audio_frame)
                if speech_frames >= SPEECH_START_THRESHOLD:
                    in_speech = True
                    silence_frames = 0
                    logging.debug("Речь обнаружена, начало передачи")
                    # Выводим статус обнаружения речи
                    print("\r🔊 Речь обнаружена... ", end="", flush=True)
            else:
                speech_frames = 0
                speaking_buffer = []
        else:
            if is_speech:
                silence_frames = 0
                speaking_buffer.append(audio_frame)
            else:
                silence_frames += 1
                if silence_frames < SILENCE_THRESHOLD_FRAMES:
                    speaking_buffer.append(audio_frame)
                else:
                    in_speech = False
                    speech_frames = 0
                    logging.debug(f"Конец речи, отправка {len(speaking_buffer)} фреймов")
                    # Выводим статус окончания речи
                    print("\r🔍 Обработка речи...         ", end="", flush=True)
                    
                    # Добавляем буфер речи в общий буфер для отправки
                    if speaking_buffer:
                        combined_speech = np.vstack(speaking_buffer)
                        with buffer_lock:
                            audio_buffer.append(combined_speech)
                        speaking_buffer = []
                        
                        # После добавления в буфер
                        time.sleep(0.1)  # Небольшая пауза
                        print("\r💬 Ожидание распознавания...", end="", flush=True)
        
        return is_speech
    except Exception as e:
        logging.error(f"Ошибка VAD: {e}")
        return False

# Функция для периодической отправки аудио из буфера
async def process_audio_buffer(loop):
    """Обрабатывает аудио буфер и отправляет данные клиентам"""
    global audio_buffer
    
    while True:
        if len(audio_buffer) > 0 and not is_sending and clients:
            with buffer_lock:
                # Берем первый фрейм из буфера
                if audio_buffer:
                    audio_data = audio_buffer.pop(0)
                    logging.debug(f"Отправка аудио данных, осталось в буфере: {len(audio_buffer)}")
                    
                    # Создаем задачу для отправки аудио
                    asyncio.create_task(broadcast_audio(audio_data))
        
        # Регулируем скорость отправки
        await asyncio.sleep(0.02)  # 20 мс - примерно размер чанка

async def start_microphone(device=None, loop=None):
    """Запуск захвата аудио с микрофона"""
    global audio_buffer
    
    # Функция обратного вызова для аудио
    def audio_callback(indata, frames, time, status):
        global audio_buffer
        
        if status:
            logging.warning(f"Статус аудио: {status}")
        
        if is_recording:
            # Копируем и конвертируем аудио данные в int16
            audio_data = np.int16(indata * 32767).copy()
            
            # Обрабатываем через VAD
            if process_vad(audio_data):
                logging.debug("Фрейм речи")
            else:
                logging.debug("Фрейм тишины")
            
            # Логгируем размер буфера раз в секунду
            if frames % (SAMPLE_RATE // CHUNK) == 0:
                logging.debug(f"Размер буфера аудио: {len(audio_buffer)}")
            
            # Если буфер стал слишком большим, удаляем старые данные
            if len(audio_buffer) > MAX_BUFFER_SIZE * 3:
                with buffer_lock:
                    # Оставляем только последние MAX_BUFFER_SIZE фреймов
                    audio_buffer = audio_buffer[-MAX_BUFFER_SIZE:]
                logging.warning(f"Буфер переполнен, удалены старые данные. Новый размер: {len(audio_buffer)}")
    
    try:
        # Инициализируем VAD
        init_vad()
        
        # Запускаем обработчик буфера в отдельной задаче
        buffer_processor = asyncio.create_task(process_audio_buffer(loop))
        
        # Запускаем поток ввода аудио
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=audio_callback,
            blocksize=CHUNK,
            dtype='float32',  # Сохраняем float32 для sounddevice, конвертируем в int16 в callback
            device=device
        )
        
        with stream:
            logging.info(f"Микрофон запущен на устройстве {device if device is not None else 'по умолчанию'}")
            logging.info(f"Частота дискретизации: {SAMPLE_RATE} Гц, каналов: {CHANNELS}")
            logging.info(f"VAD: режим {VAD_MODE}, размер фрейма {CHUNK} (примерно {CHUNK/SAMPLE_RATE*1000:.1f} мс)")
            
            # Здесь мы просто держим микрофон активным
            while True:
                await asyncio.sleep(1)
                logging.debug(f"Статус микрофона: активен, клиентов: {len(clients)}, размер буфера: {len(audio_buffer)}")
    
    except Exception as e:
        logging.error(f"Ошибка микрофона: {e}")
        logging.info("Доступные аудио устройства:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            logging.info(f"  {i} {dev['name']}, ALSA ({dev['max_input_channels']} in, {dev['max_output_channels']} out)")
        raise

async def main():
    global VAD_MODE  # Moved the global declaration to the top of the function
    
    # Разбор аргументов командной строки
    parser = argparse.ArgumentParser(description="Микрофонный сервис для ARMv8")
    parser.add_argument("--device", type=int, help="Индекс аудио устройства ввода")
    parser.add_argument("--list-devices", action="store_true", help="Показать список доступных аудио устройств")
    parser.add_argument("--port", type=int, default=WS_PORT, help="Порт WebSocket сервера")
    parser.add_argument("--debug", action="store_true", help="Включить отладочные сообщения")
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=VAD_MODE, 
                        help="Режим агрессивности VAD (0-3)")
    args = parser.parse_args()
    
    # Применяем аргументы
    VAD_MODE = args.vad_mode
    
    # Настройка уровня логирования
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.list_devices:
        list_audio_devices()
        return
    
    # Запуск WebSocket сервера
    ws_server = await websockets.serve(ws_handler, HOST, args.port)
    logging.info(f"WebSocket сервер запущен на ws://{HOST}:{args.port}")
    
    # Получаем event loop
    loop = asyncio.get_running_loop()
    
    # Запуск микрофона
    mic_task = asyncio.create_task(start_microphone(args.device, loop=loop))
    
    # Продолжать выполнение
    try:
        await asyncio.gather(mic_task)
    finally:
        ws_server.close()
        await ws_server.wait_closed()

if __name__ == "__main__":
    print("[ИНФО] Запуск микрофонного сервиса...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[ИНФО] Завершение работы...")
    except Exception as e:
        print(f"[ОШИБКА] Критическая ошибка: {e}")
        import sys
        sys.exit(1) 