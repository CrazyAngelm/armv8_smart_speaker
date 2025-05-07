#!/usr/bin/env python3
"""
Сервис распознавания речи для ARMv8 (Debian).
Получает аудио через WebSocket и преобразует в текст с помощью Vosk.
"""
import asyncio
import websockets
import json
import os
import sys
import argparse
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env если существует
env_path = Path('.env')
if env_path.exists():
    load_dotenv()

# Отключаем логи Kaldi/Vosk разными способами
os.environ['VOSK_LOGLEVEL'] = '-1'  # Пробуем другое имя
os.environ['VOSK_LOG_LEVEL'] = '-1'  # Более агрессивный уровень
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Если используется TensorFlow
import warnings
warnings.filterwarnings('ignore')

# Импортируем Vosk только после настройки логов
from vosk import Model, KaldiRecognizer, SetLogLevel

# Принудительно отключаем логи Vosk
SetLogLevel(-1)

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# Конфигурация
HOST = "0.0.0.0"
STT_WS_PORT = 8778
MIC_WS_URI = "ws://localhost:8765"
RECONNECT_BASE_DELAY = 1  # Начальная задержка для переподключения (секунды)
RECONNECT_MAX_DELAY = 30  # Максимальная задержка для переподключения (секунды)
RECONNECT_MAX_RETRIES = 0  # 0 - бесконечное количество попыток

# Путь к модели Vosk
VOSK_MODEL_PATH = os.path.join("models", "vosk-model-small-ru-0.22")

# Глобальное состояние
text_buffer = []
is_processing = False
model = None
recognizer = None
reconnect_delay = RECONNECT_BASE_DELAY
reconnect_attempts = 0

def load_model():
    """Загружаем модель один раз при запуске"""
    global model, recognizer
    
    if not os.path.exists(VOSK_MODEL_PATH):
        logging.error(f"Модель Vosk не найдена: {VOSK_MODEL_PATH}")
        return False
    
    try:
        model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(model, 16000)
        logging.info(f"Модель Vosk загружена: {VOSK_MODEL_PATH}")
        return True
    except Exception as e:
        logging.error(f"Ошибка загрузки модели: {e}")
        return False

# Обработка аудио с помощью Vosk
async def process_audio(audio_data):
    global is_processing, text_buffer, recognizer
    
    if is_processing:
        return None
    
    if not recognizer:
        logging.error("Распознаватель не инициализирован")
        return None
    
    try:
        is_processing = True
        
        # Помечаем начало обработки для отладки
        logging.debug(f"Начало обработки аудио данных размером {len(audio_data)} байт")
        
        # Обрабатываем аудио
        if recognizer.AcceptWaveform(audio_data):
            result = recognizer.Result()
            logging.debug(f"Получен результат: {result}")
            
            # Разбор результата
            result_json = json.loads(result)
            recognized_text = result_json.get("text", "")
            
            if recognized_text:
                # Логируем распознанный текст
                logging.info(f"Распознано: {recognized_text}")
                
                # Выводим текст в консоль с заметным форматированием
                print("\n" + "="*50)
                print(f"🎤 РАСПОЗНАНО: \"{recognized_text}\"")
                print("="*50)
                
                text_buffer.append(recognized_text)
                # Хранить только последние 5 текстов
                if len(text_buffer) > 5:
                    text_buffer.pop(0)
                return recognized_text
        else:
            # Получаем промежуточный результат для отладки
            partial = recognizer.PartialResult()
            partial_json = json.loads(partial)
            partial_text = partial_json.get("partial", "")
            if partial_text:
                logging.debug(f"Промежуточный результат: {partial_text}")
                # Можно также выводить промежуточные результаты, но без яркого форматирования
                print(f"\rПромежуточно: {partial_text}", end="")
        
        return None
    except Exception as e:
        logging.error(f"Ошибка распознавания: {e}")
        return None
    finally:
        is_processing = False

# Подключение к микрофонному WebSocket
async def connect_to_mic():
    global reconnect_delay, reconnect_attempts
    
    while True:
        try:
            # Вычисляем задержку для переподключения с экспоненциальным отступом
            if reconnect_attempts > 0:
                current_delay = min(reconnect_delay * (2 ** (reconnect_attempts - 1)), RECONNECT_MAX_DELAY)
                logging.info(f"Повторная попытка подключения через {current_delay:.1f} секунд...")
                await asyncio.sleep(current_delay)
            
            # Проверяем, не превышено ли максимальное количество попыток
            if RECONNECT_MAX_RETRIES > 0 and reconnect_attempts >= RECONNECT_MAX_RETRIES:
                logging.error(f"Превышено максимальное количество попыток подключения ({RECONNECT_MAX_RETRIES})")
                return
            
            logging.info(f"Подключение к микрофонному сервису: {MIC_WS_URI}")
            reconnect_attempts += 1
            
            # Устанавливаем соединение с keepalive ping настройками
            async with websockets.connect(
                MIC_WS_URI, 
                max_size=2**22, 
                ping_interval=30, 
                ping_timeout=10,
                close_timeout=5
            ) as websocket:
                logging.info("Подключено к микрофонному сервису")
                reconnect_attempts = 0  # Сбрасываем счетчик при успешном подключении
                reconnect_delay = RECONNECT_BASE_DELAY  # Сбрасываем задержку
                
                # Обработка сообщений
                async for message in websocket:
                    if isinstance(message, bytes):
                        # Обработка аудио данных
                        logging.debug(f"Получены аудио данные размером {len(message)} байт")
                        await process_audio(message)
                    else:
                        logging.info(f"Получено текстовое сообщение: {message}")
        
        except asyncio.CancelledError:
            logging.info("Задача подключения отменена")
            raise
        except websockets.exceptions.ConnectionClosedError as e:
            logging.warning(f"Соединение закрыто: {e}")
        except websockets.exceptions.ConnectionClosedOK:
            logging.info("Соединение закрыто нормально")
        except websockets.exceptions.InvalidStatusCode as e:
            logging.error(f"Неверный статус код: {e}")
        except (OSError, ConnectionRefusedError) as e:
            logging.error(f"Ошибка сети: {e}")
        except Exception as e:
            logging.error(f"Неожиданная ошибка при подключении: {e}")
        
        # Увеличиваем счетчик попыток для следующей итерации (если не было успешного подключения)
        if reconnect_attempts > 0:
            logging.warning(f"Ошибка подключения к микрофону, попытка {reconnect_attempts}")

# Внешний STT WebSocket сервер
async def stt_handler(websocket):
    client_ip = websocket.remote_address[0]
    logging.info(f"STT клиент подключен: {client_ip}")
    
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
        logging.info(f"STT клиент отключен: {client_ip}")
    except Exception as e:
        logging.error(f"Ошибка WebSocket: {e}")

async def main():
    global VOSK_MODEL_PATH, MIC_WS_URI, RECONNECT_BASE_DELAY, RECONNECT_MAX_DELAY, RECONNECT_MAX_RETRIES
    
    # Разбор аргументов командной строки
    parser = argparse.ArgumentParser(description="Сервис распознавания речи для ARMv8")
    parser.add_argument("--model", type=str, default=VOSK_MODEL_PATH, help="Путь к модели Vosk")
    parser.add_argument("--mic-uri", type=str, default=MIC_WS_URI, help="URI WebSocket микрофонного сервиса")
    parser.add_argument("--port", type=int, default=STT_WS_PORT, help="Порт STT WebSocket сервера")
    parser.add_argument("--debug", action="store_true", help="Включить отладочные сообщения")
    parser.add_argument("--reconnect-delay", type=float, default=RECONNECT_BASE_DELAY, 
                        help="Базовая задержка между попытками переподключения (секунды)")
    parser.add_argument("--max-reconnect-delay", type=float, default=RECONNECT_MAX_DELAY,
                        help="Максимальная задержка между попытками переподключения (секунды)")
    parser.add_argument("--max-retries", type=int, default=RECONNECT_MAX_RETRIES,
                        help="Максимальное количество попыток переподключения (0=бесконечно)")
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Обновляем глобальные настройки
    VOSK_MODEL_PATH = args.model
    MIC_WS_URI = args.mic_uri
    RECONNECT_BASE_DELAY = args.reconnect_delay
    RECONNECT_MAX_DELAY = args.max_reconnect_delay
    RECONNECT_MAX_RETRIES = args.max_retries
    
    logging.info(f"Используется модель Vosk: {VOSK_MODEL_PATH}")
    logging.info(f"Микрофонный сервис: {MIC_WS_URI}")
    logging.info(f"Настройки переподключения: базовая задержка={RECONNECT_BASE_DELAY}с, "
                 f"макс. задержка={RECONNECT_MAX_DELAY}с, макс. попыток={RECONNECT_MAX_RETRIES or 'бесконечно'}")
    
    # Загружаем модель
    if not load_model():
        logging.error("Не удалось загрузить модель Vosk. Выход.")
        return
    
    # Запуск STT WebSocket сервера
    stt_server = await websockets.serve(
        stt_handler, 
        HOST, 
        args.port, 
        max_size=2**22,
        ping_interval=30,
        ping_timeout=10
    )
    
    logging.info(f"STT WebSocket сервер запущен на ws://{HOST}:{args.port}")
    
    # Подключение к микрофонному сервису
    mic_task = asyncio.create_task(connect_to_mic())
    
    # Продолжать выполнение
    try:
        await asyncio.gather(mic_task)
    except asyncio.CancelledError:
        logging.info("Задачи отменены")
    finally:
        stt_server.close()
        await stt_server.wait_closed()

if __name__ == "__main__":
    print("[ИНФО] Запуск сервиса распознавания речи...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[ИНФО] Завершение работы...")
    except Exception as e:
        print(f"[ОШИБКА] Критическая ошибка: {e}")
        sys.exit(1) 