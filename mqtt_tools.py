import json
import time
import paho.mqtt.client as mqtt
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# MQTT конфигурация
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
RECOGNIZED_INTENT_PATH = os.getenv("RECOGNIZED_INTENT_PATH", "hermes/intent")
TTS_SAY_PATH = os.getenv("TTS_SAY_PATH", "hermes/tts/say")

# Инициализация MQTT клиента
client = None
response_queue = {}

def get_mqtt_client() -> mqtt.Client:
    global client
    if client is None:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = on_connect
        client.on_message = on_message
        try:
            client.connect(MQTT_HOST, MQTT_PORT)
            client.loop_start()
            print(f"[MQTT] Подключен к MQTT брокеру {MQTT_HOST}:{MQTT_PORT}")
        except Exception as e:
            print(f"[MQTT] Ошибка подключения к MQTT: {e}")
    return client

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("[MQTT] Connected OK")
        # Подписка на нужные топики
        client.subscribe("hermes/tts/say")
        client.subscribe(f"{RECOGNIZED_INTENT_PATH}/response/#")
    else:
        print(f"[MQTT] Ошибка подключения: {reason_code}")

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload
    
    try:
        if isinstance(payload, bytes):
            payload = payload.decode('utf-8')
        
        if topic.startswith(f"{RECOGNIZED_INTENT_PATH}/response/"):
            # Извлекаем request_id из топика
            request_id = topic.split('/')[-1]
            if request_id in response_queue:
                response_queue[request_id] = payload
                print(f"[MQTT] Получен ответ для запроса {request_id}: {payload[:100]}...")
    except Exception as e:
        print(f"[MQTT] Ошибка обработки сообщения: {e}")

def wait_for_response(request_id: str, timeout: int = 10) -> Optional[str]:
    """Ожидание ответа от MQTT на запрос с указанным ID"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if request_id in response_queue and response_queue[request_id] is not None:
            response = response_queue[request_id]
            del response_queue[request_id]
            return response
        time.sleep(0.1)
    print(f"[MQTT] Таймаут ожидания ответа для {request_id}")
    if request_id in response_queue:
        del response_queue[request_id]
    return None

# --- Tools для интеграции с LangGraph ---

def tool_get_time():
    """Возвращает текущее время"""
    try:
        client = get_mqtt_client()
        request_id = f"time_{int(time.time())}"
        response_queue[request_id] = None
        
        # Отправляем запрос
        client.publish(
            RECOGNIZED_INTENT_PATH, 
            json.dumps({
                "intent": {"intentName": "GetTime"},
                "request_id": request_id
            })
        )
        
        # Ожидаем ответ
        response = wait_for_response(request_id)
        if response:
            try:
                data = json.loads(response)
                return data.get("text", "Не удалось получить время")
            except:
                return response
        
        # Если нет ответа, самостоятельно формируем
        now = datetime.now()
        return f"Текущее время {now.hour} часов, {now.minute} минут"
    except Exception as e:
        print(f"[ERROR] tool_get_time: {e}")
        return f"Ошибка при получении времени: {str(e)}"

def tool_set_timer(minutes: int = 0, seconds: int = 0, hours: int = 0):
    """Устанавливает таймер на указанное время"""
    try:
        client = get_mqtt_client()
        request_id = f"timer_{int(time.time())}"
        response_queue[request_id] = None
        
        slots = []
        if hours > 0:
            slots.append({"slotName": "hour", "value": {"value": hours}})
        if minutes > 0:
            slots.append({"slotName": "minute", "value": {"value": minutes}})
        if seconds > 0:
            slots.append({"slotName": "second", "value": {"value": seconds}})
        
        # Отправляем запрос
        client.publish(
            RECOGNIZED_INTENT_PATH, 
            json.dumps({
                "intent": {"intentName": "SetTimer"},
                "slots": slots,
                "request_id": request_id
            })
        )
        
        # Ожидаем ответ
        response = wait_for_response(request_id)
        if response:
            try:
                data = json.loads(response)
                return data.get("text", "Таймер установлен")
            except:
                return response
        
        return f"Таймер установлен на {hours} ч, {minutes} мин, {seconds} сек"
    except Exception as e:
        print(f"[ERROR] tool_set_timer: {e}")
        return f"Ошибка при установке таймера: {str(e)}"

def tool_set_notification(text: str, minutes: int = 0, seconds: int = 0, hours: int = 0):
    """Устанавливает напоминание на указанное время"""
    try:
        client = get_mqtt_client()
        request_id = f"notification_{int(time.time())}"
        response_queue[request_id] = None
        
        slots = []
        if hours > 0:
            slots.append({"slotName": "hour", "value": {"value": hours}})
        if minutes > 0:
            slots.append({"slotName": "minute", "value": {"value": minutes}})
        if seconds > 0:
            slots.append({"slotName": "second", "value": {"value": seconds}})
        
        # Отправляем запрос
        client.publish(
            RECOGNIZED_INTENT_PATH, 
            json.dumps({
                "intent": {"intentName": "SetNotification"},
                "slots": slots,
                "rawInput": f"Напомни через {hours} часов {minutes} минут {seconds} секунд о том {text}",
                "request_id": request_id
            })
        )
        
        # Ожидаем ответ
        response = wait_for_response(request_id)
        if response:
            try:
                data = json.loads(response)
                return data.get("text", "Напоминание установлено")
            except:
                return response
        
        return f"Напоминание установлено на {hours} ч, {minutes} мин, {seconds} сек: {text}"
    except Exception as e:
        print(f"[ERROR] tool_set_notification: {e}")
        return f"Ошибка при установке напоминания: {str(e)}"

def tool_get_weather():
    """Получает информацию о погоде"""
    try:
        client = get_mqtt_client()
        request_id = f"weather_{int(time.time())}"
        response_queue[request_id] = None
        
        # Отправляем запрос
        client.publish(
            RECOGNIZED_INTENT_PATH, 
            json.dumps({
                "intent": {"intentName": "GetWeather"},
                "request_id": request_id
            })
        )
        
        # Ожидаем ответ
        response = wait_for_response(request_id)
        if response:
            try:
                data = json.loads(response)
                return data.get("text", "Не удалось получить информацию о погоде")
            except:
                return response
        
        return "Не удалось получить информацию о погоде"
    except Exception as e:
        print(f"[ERROR] tool_get_weather: {e}")
        return f"Ошибка при получении погоды: {str(e)}"

def tool_call_contact(contact_name: str):
    """Выполняет звонок указанному контакту"""
    try:
        client = get_mqtt_client()
        request_id = f"call_{int(time.time())}"
        response_queue[request_id] = None
        
        # Отправляем запрос
        client.publish(
            RECOGNIZED_INTENT_PATH, 
            json.dumps({
                "intent": {"intentName": "InitiateCall"},
                "input": "позвони",
                "rawInput": f"позвони {contact_name}",
                "request_id": request_id
            })
        )
        
        # Ожидаем ответ
        response = wait_for_response(request_id)
        if response:
            try:
                data = json.loads(response)
                return data.get("text", f"Звоню контакту {contact_name}")
            except:
                return response
        
        return f"Звоню контакту {contact_name}"
    except Exception as e:
        print(f"[ERROR] tool_call_contact: {e}")
        return f"Ошибка при звонке контакту: {str(e)}"

# Определение инструментов для LangGraph
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Получает текущее время",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Устанавливает таймер на указанное время",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "Количество минут"
                    },
                    "seconds": {
                        "type": "integer",
                        "description": "Количество секунд"
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Количество часов"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_notification",
            "description": "Устанавливает напоминание на указанное время с указанным текстом",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Текст напоминания"
                    },
                    "minutes": {
                        "type": "integer",
                        "description": "Количество минут"
                    },
                    "seconds": {
                        "type": "integer",
                        "description": "Количество секунд"
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Количество часов"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Получает информацию о текущей погоде",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_contact",
            "description": "Выполняет звонок указанному контакту",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {
                        "type": "string",
                        "description": "Имя контакта для звонка"
                    }
                },
                "required": ["contact_name"]
            }
        }
    }
]

# Словарь для вызова функций по имени
tool_mapping = {
    "get_time": tool_get_time,
    "set_timer": tool_set_timer,
    "set_notification": tool_set_notification,
    "get_weather": tool_get_weather,
    "call_contact": tool_call_contact
}

# Функция для выполнения инструмента
def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Выполняет инструмент по имени с указанными аргументами"""
    if tool_name in tool_mapping:
        try:
            return tool_mapping[tool_name](**tool_args)
        except Exception as e:
            print(f"[ERROR] execute_tool({tool_name}): {e}")
            return f"Ошибка при выполнении инструмента {tool_name}: {str(e)}"
    else:
        return f"Инструмент {tool_name} не найден"

# Инициализация при импорте модуля
def init_mqtt():
    get_mqtt_client()

# --- Тестирование модуля ---
if __name__ == "__main__":
    init_mqtt()
    print("Тестирование MQTT tools...")
    
    print("\nТест получения времени:")
    print(tool_get_time())
    
    print("\nТест установки таймера:")
    print(tool_set_timer(minutes=1))
    
    print("\nТест получения погоды:")
    print(tool_get_weather())
    
    print("\nTest complete.")
    time.sleep(1)
    if client:
        client.loop_stop()
        client.disconnect() 