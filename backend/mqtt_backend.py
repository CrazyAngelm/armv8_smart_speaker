# mosquitto_pub -h localhost -t "hermes/intent" -m '{"intent": {"name": "SetTimer"}, "slots": {"minutes": 1, "seconds": 3}, "siteId": 5}'
import json
import paho.mqtt.client as mqtt
import time
import config
from models import Event, Contact
from base import init_db, session
import threading
from datetime import datetime, timedelta
import base_event
from base_event import SetNotificationEvent, SetTimerEvent
from sqlalchemy import select, func
from weather_client import WeatherAPIClient
import os
import subprocess
import sys
from pathlib import Path
import socket
import time

# Константы для подключения к TTS WebSocket серверу
TTS_WS_HOST = os.getenv("TTS_WS_HOST", "localhost")
TTS_WS_PORT = int(os.getenv("TTS_WS_PORT", 8777))

token = os.getenv("WEATHER_API_TOKEN")
weather_client = WeatherAPIClient(token=token)


stop_event_checker = threading.Event()

events = []

# Флаг, указывающий, инициализирован ли pygame
pygame_initialized = False

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected OK")
    client.subscribe(config.RECOGNIZED_INTENT_PATH)
    client.subscribe(config.UNRECOGNIZED_INTENT_PATH)


def on_disconnect(client, userdata, flags, rc):

    client.reconnect()


def get_unfinished_events():

    stmt = select(Event).where(Event.status == 0)
    unfinished_events = session.execute(stmt).scalars().all()

    for event in unfinished_events:

        if datetime.now() < event.timestamp:

            new_event = getattr(base_event, event.intent)(
                **event.to_dict(), 
                session=session,
            )

            events.append(new_event)

        else:
            event.status = 1
            session.commit()


def set_time_handler(nlu_payload, client):

    slots = nlu_payload.get("slots", [])

    hours = 0
    minutes = 0
    seconds = 0
    
    for slot in slots:
        value = int(slot["value"]["value"])
        match slot["slotName"]:
            case "hour":
                hours = value
            case "minute":
                minutes = value
            case "second":
                seconds = value

    event_timestamp = datetime.now() + timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
    timer = SetTimerEvent(timestamp=event_timestamp, session=session)
    events.append(timer)
    
    response_text = "Поставил таймер"
    client.publish("hermes/tts/say", json.dumps({"text": response_text}))
    
    # Отправляем ответ в топик запроса, если указан request_id
    if "request_id" in nlu_payload:
        request_id = nlu_payload["request_id"]
        client.publish(f"{config.RECOGNIZED_INTENT_PATH}/response/{request_id}", 
                       json.dumps({"text": response_text}))
    
    return response_text


def get_time_handler(client, nlu_payload=None):

    datetime_now = datetime.now()

    hours = datetime_now.hour
    minutes = datetime_now.minute

    response_text = f"Текущее время {hours} часов, {minutes} минут"
    print(response_text)
    client.publish("hermes/tts/say", json.dumps({"text": response_text}))
    
    # Отправляем ответ в топик запроса, если указан request_id
    if nlu_payload and "request_id" in nlu_payload:
        request_id = nlu_payload["request_id"]
        client.publish(f"{config.RECOGNIZED_INTENT_PATH}/response/{request_id}", 
                       json.dumps({"text": response_text}))
    
    return response_text


def set_notification_handler(nlu_payload, client):
    
    text = nlu_payload.get('rawInput', '')
    notification_text = ""
    
    if 'о том' in text:
        notification_text = text.split('о том')[1]
    else:
        notification_text = "неуказанное напоминание"

    slots = nlu_payload.get("slots", [])

    hours = 0
    minutes = 0
    seconds = 0
    
    for slot in slots:
        value = int(slot["value"]["value"]) if slot['slotName'] in ['hour', 'minute', 'second'] else None
        match slot["slotName"]:
            case "hour":
                hours = value
            case "minute":
                minutes = value
            case "second":
                seconds = value

    event_timestamp = datetime.now() + timedelta(hours=hours, minutes=minutes, seconds=seconds)

    notifier = SetNotificationEvent(
        timestamp=event_timestamp, 
        notification_text=notification_text,
        session=session,
    )

    events.append(notifier)

    response_text = "Поставил напоминание"
    client.publish("hermes/tts/say", json.dumps({"text": response_text}))
    
    # Отправляем ответ в топик запроса, если указан request_id
    if "request_id" in nlu_payload:
        request_id = nlu_payload["request_id"]
        client.publish(f"{config.RECOGNIZED_INTENT_PATH}/response/{request_id}", 
                       json.dumps({"text": response_text}))
    
    return response_text


def get_weather_hahdler(client, nlu_payload=None):
    response = weather_client.get_weather()

    if not response:
        response_text = "Сервис погоды временно недоступен"
        client.publish("hermes/tts/say", json.dumps({"text": response_text}))
        
        # Отправляем ответ в топик запроса, если указан request_id
        if nlu_payload and "request_id" in nlu_payload:
            request_id = nlu_payload["request_id"]
            client.publish(f"{config.RECOGNIZED_INTENT_PATH}/response/{request_id}", 
                           json.dumps({"text": response_text}))
        
        return response_text

    print(response)
    response_text = (f"Текущая погода для региона " 
            f"{response['region']}: "
            f"температура {response['temperature']} градусов по цельсию, "
            f"скорость ветра {response['wind']} километров в час")

    client.publish("hermes/tts/say", json.dumps({"text": response_text}))
    
    # Отправляем ответ в топик запроса, если указан request_id
    if nlu_payload and "request_id" in nlu_payload:
        request_id = nlu_payload["request_id"]
        client.publish(f"{config.RECOGNIZED_INTENT_PATH}/response/{request_id}", 
                       json.dumps({"text": response_text}))
    
    return response_text


def initiate_call_handler(nlu_payload, client):

    contact_name = ""
    
    # Проверяем разные форматы данных
    if "rawInput" in nlu_payload and "input" in nlu_payload:
        intent_key_words = nlu_payload['input']
        raw_message = nlu_payload['rawInput']
        try:
            contact_name = raw_message.split(intent_key_words)[1].strip()
        except:
            # Если разделение не удалось, используем всё сообщение
            contact_name = raw_message
    elif "contact_name" in nlu_payload:
        # Прямая передача имени контакта
        contact_name = nlu_payload["contact_name"]
    else:
        response_text = "Не указано имя контакта"
        client.publish("hermes/tts/say", json.dumps({"text": response_text}))
        
        # Отправляем ответ в топик запроса, если указан request_id
        if "request_id" in nlu_payload:
            request_id = nlu_payload["request_id"]
            client.publish(f"{config.RECOGNIZED_INTENT_PATH}/response/{request_id}", 
                           json.dumps({"text": response_text}))
        
        return response_text

    # Нечувствительный к регистру поиск с использованием LIKE
    stmt = select(Contact).where(func.lower(Contact.name).like(f"%{contact_name.lower()}%"))
    results = session.execute(stmt).scalars().all()

    if not results or len(results) == 0:
        response_text = f"Контакт {contact_name} не найден"
        print(response_text)
        client.publish("hermes/tts/say", json.dumps({"text": "Контакт не найден"})) 
    else:
        # Если найдено несколько контактов, используем первый
        result = results[0]
        formatted_number = ' '.join(str(result.phone_number))

        response_text = f"Контакт {result.name} найден, звоню {formatted_number}"
        print(response_text)

        client.publish("hermes/tts/say", json.dumps({
            "text": f"Контакт найден, звоню {formatted_number}"
        }))
    
    # Отправляем ответ в топик запроса, если указан request_id
    if "request_id" in nlu_payload:
        request_id = nlu_payload["request_id"]
        client.publish(f"{config.RECOGNIZED_INTENT_PATH}/response/{request_id}", 
                       json.dumps({"text": response_text}))
    
    return response_text



def on_message(client, userdata, msg):

    try:
        nlu_payload = json.loads(msg.payload)
        # print(json.dumps(nlu_payload, indent=4, ensure_ascii=False))

        if msg.topic == config.UNRECOGNIZED_INTENT_PATH:
            
            sentence = "Не поняла"
            print("Recognition failure")
            client.publish("hermes/tts/say", json.dumps({"text": sentence})) 

        else:
            # Проверяем наличие поля intent
            if "intent" in nlu_payload:
                intent_data = nlu_payload["intent"]
                
                # Определяем имя интента
                intent_name = None
                if isinstance(intent_data, dict) and "intentName" in intent_data:
                    intent_name = intent_data["intentName"]
                elif isinstance(intent_data, dict) and "name" in intent_data:
                    intent_name = intent_data["name"]
                
                if intent_name:
                    if intent_name == 'SetTimer':
                        set_time_handler(nlu_payload, client)
                    elif intent_name == 'GetTime':
                        get_time_handler(client, nlu_payload)
                    elif intent_name == 'SetNotification':
                        set_notification_handler(nlu_payload, client)
                    elif intent_name == 'GetWeather':
                        get_weather_hahdler(client, nlu_payload)
                    elif intent_name == 'InitiateCall':
                        initiate_call_handler(nlu_payload, client)
                else:
                    print(f"[ERROR] Неизвестный интент: {nlu_payload}")
                    
                    # Отправляем ответ об ошибке, если указан request_id
                    if "request_id" in nlu_payload:
                        request_id = nlu_payload["request_id"]
                        client.publish(f"{config.RECOGNIZED_INTENT_PATH}/response/{request_id}", 
                                      json.dumps({"text": "Неизвестный интент"}))
    except Exception as e:
        print(f"[ERROR] Ошибка при обработке сообщения: {e}")
        
        # Пытаемся извлечь request_id для отправки ответа об ошибке
        try:
            if isinstance(msg.payload, bytes):
                payload_str = msg.payload.decode('utf-8')
                payload_data = json.loads(payload_str)
                if "request_id" in payload_data:
                    request_id = payload_data["request_id"]
                    client.publish(f"{config.RECOGNIZED_INTENT_PATH}/response/{request_id}", 
                                  json.dumps({"text": f"Ошибка: {str(e)}"}))
        except:
            pass


def play_sound_file(sound_file_path):
    """Воспроизводит звуковой файл с использованием разных методов"""
    global pygame_initialized
    
    try:
        print(f"[INFO] Воспроизведение звукового файла: {sound_file_path}")
        
        # Проверяем наличие файла
        if not Path(sound_file_path).exists():
            print(f"[ERROR] Файл {sound_file_path} не найден!")
            return False
        
        # Метод 1: Попытка использовать pygame
        try:
            import pygame
            
            if not pygame_initialized:
                pygame.mixer.init()
                pygame_initialized = True
                
            pygame.mixer.music.load(sound_file_path)
            pygame.mixer.music.play()
            
            # Ждем окончания воспроизведения
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            return True
        except ImportError:
            print("[WARNING] pygame не установлен. Попытка использовать альтернативные методы.")
        except Exception as e:
            print(f"[ERROR] Ошибка воспроизведения через pygame: {e}")
        
        # Метод 2: Использовать системные команды
        if os.name == 'nt':  # Windows
            try:
                # Вариант 1: PowerShell (работает с MP3)
                cmd = ['powershell', '-c', f'(New-Object Media.SoundPlayer "{sound_file_path}").PlaySync()']
                subprocess.run(cmd, check=True)
                return True
            except Exception as e:
                print(f"[ERROR] Ошибка PowerShell воспроизведения: {e}")
                
                # Вариант 2: Windows Media Player
                try:
                    os.system(f'start wmplayer "{sound_file_path}"')
                    time.sleep(3)  # Даем время на воспроизведение
                    os.system('taskkill /f /im wmplayer.exe')
                    return True
                except Exception as e:
                    print(f"[ERROR] Ошибка WMP воспроизведения: {e}")
        else:  # Linux/Unix
            # Вариант 1: mpg123 для MP3
            if sound_file_path.endswith('.mp3'):
                if os.system('which mpg123 > /dev/null') == 0:
                    os.system(f'mpg123 "{sound_file_path}"')
                    return True
                print("[WARNING] mpg123 не установлен. Установите с помощью: sudo apt-get install mpg123")
            
            # Вариант 2: aplay для WAV
            elif sound_file_path.endswith('.wav'):
                if os.system('which aplay > /dev/null') == 0:
                    os.system(f'aplay "{sound_file_path}"')
                    return True
                print("[WARNING] aplay не установлен.")
        
        print("[ERROR] Не удалось воспроизвести звуковой файл ни одним из методов.")
        return False
    except Exception as e:
        print(f"[ERROR] Ошибка при воспроизведении звукового файла: {e}")
        return False

# Функция для прямого синтеза речи через TTS сервер
def synthesize_speech(text):
    """Синтезирует речь напрямую, отправляя запрос TTS серверу"""
    try:
        import websockets
        import asyncio
        
        async def send_to_tts():
            uri = f"ws://{TTS_WS_HOST}:{TTS_WS_PORT}"
            print(f"[TTS] Подключение к TTS серверу: {uri}")
            
            try:
                async with websockets.connect(uri) as websocket:
                    print(f"[TTS] Отправка текста: {text}")
                    await websocket.send(text)
                    
                    # Получаем аудио данные
                    audio_data = await websocket.recv()
                    
                    if isinstance(audio_data, bytes):
                        print(f"[TTS] Получены аудио данные: {len(audio_data)} байт")
                        # Используем уникальное временное имя файла вместо фиксированного
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
                            temp_file = temp.name
                            temp.write(audio_data)
                        
                        try:
                            # Воспроизводим
                            play_sound_file(temp_file)
                        finally:
                            # Удаляем временный файл в блоке finally чтобы обеспечить очистку даже при ошибках
                            try:
                                if os.path.exists(temp_file):
                                    # Даем небольшую паузу перед удалением, чтобы player успел освободить файл
                                    time.sleep(0.2)
                                    os.remove(temp_file)
                            except Exception as e:
                                print(f"[TTS] Ошибка при удалении временного файла: {e}")
                    else:
                        print(f"[TTS] Ошибка: Получен неправильный формат ответа")
            except Exception as e:
                print(f"[TTS] Ошибка при подключении к TTS серверу: {e}")
        
        # Создаем и запускаем event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_to_tts())
        
    except ImportError:
        print("[TTS] Ошибка: библиотека websockets не установлена")
    except Exception as e:
        print(f"[TTS] Ошибка синтеза речи: {e}")

def event_checker(client):
    while not stop_event_checker.is_set():
        now = datetime.now()
        for event in events:
            if now >= event.timestamp:
                # Получаем результат выполнения события
                response = event.finish_event(session=session)
                
                # Определяем тип события
                event_type = event.__class__.__name__
                
                # Если это таймер, воспроизводим звуковой сигнал
                if event_type == "SetTimerEvent":
                    # Полный путь к timer.mp3 (в корне проекта)
                    timer_sound_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "timer.mp3")
                    
                    # Запускаем воспроизведение в отдельном потоке, чтобы не блокировать основной поток
                    audio_thread = threading.Thread(target=play_sound_file, args=(timer_sound_path,), daemon=True)
                    audio_thread.start()
                    
                    # Даем время на воспроизведение звука
                    time.sleep(1.5)
                    
                    # Напрямую синтезируем речь для воспроизведения
                    tts_thread = threading.Thread(target=synthesize_speech, args=(response,), daemon=True)
                    tts_thread.start()
                    
                    # Даем время на обработку TTS
                    time.sleep(0.5)
                # Если это напоминание, озвучиваем текст напоминания через TTS
                elif event_type == "SetNotificationEvent":
                    tts_thread = threading.Thread(target=synthesize_speech, args=(response,), daemon=True)
                    tts_thread.start()
                    time.sleep(0.5)
                
                # Удаляем событие из списка
                events.remove(event)
                
                # Отправляем сообщение для озвучивания через MQTT (как раньше)
                client.publish("hermes/tts/say", json.dumps({"text": response}))
                
                # Также отправляем в специальный топик для таймеров и напоминаний
                client.publish("hermes/notification", json.dumps({
                    "type": event_type,
                    "text": response,
                    "timestamp": now.isoformat()
                }))
                
                print(f"Event triggered: {response}")
        time.sleep(0.2)


if __name__ == '__main__':

    init_db()
    get_unfinished_events()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    client.connect("localhost", 1883)

    event_checker_thread = threading.Thread(target=event_checker, args=(client,))
    event_checker_thread.daemon = True
    event_checker_thread.start()

    client.loop_forever()
