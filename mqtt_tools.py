# optimized_mqtt_tools.py
import json
import time
import asyncio
import paho.mqtt.client as mqtt
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# MQTT конфигурация
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
RECOGNIZED_INTENT_PATH = os.getenv("RECOGNIZED_INTENT_PATH", "hermes/intent")
TTS_SAY_PATH = os.getenv("TTS_SAY_PATH", "hermes/tts/say")

# Оптимизированные таймауты
MQTT_TIMEOUT = int(os.getenv("MQTT_TIMEOUT", "3"))  # Уменьшен с 10 до 3 секунд
MQTT_ENABLED = os.getenv("MQTT_ENABLED", "true").lower() == "true"

# Глобальные переменные
client = None
response_queue = {}
mqtt_connected = False

class MQTTManager:
    """Менеджер MQTT соединения с отказоустойчивостью"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.connection_attempts = 0
        self.max_attempts = 3
        
    def get_client(self) -> Optional[mqtt.Client]:
        if not MQTT_ENABLED:
            return None
            
        if self.client is None or not self.connected:
            self._try_connect()
        return self.client if self.connected else None
    
    def _try_connect(self):
        if self.connection_attempts >= self.max_attempts:
            print(f"[MQTT] Превышено максимальное количество попыток подключения ({self.max_attempts})")
            return
            
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            
            self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            self.client.loop_start()
            self.connection_attempts += 1
            
        except Exception as e:
            print(f"[MQTT] Ошибка подключения: {e}")
            self.connected = False
            self.connection_attempts += 1
    
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print("[MQTT] Подключение успешно")
            self.connected = True
            self.connection_attempts = 0  # Сбрасываем счетчик при успешном подключении
            client.subscribe(f"{RECOGNIZED_INTENT_PATH}/response/#")
        else:
            print(f"[MQTT] Ошибка подключения: {reason_code}")
            self.connected = False
    
    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload
        
        try:
            if isinstance(payload, bytes):
                payload = payload.decode('utf-8')
            
            if topic.startswith(f"{RECOGNIZED_INTENT_PATH}/response/"):
                request_id = topic.split('/')[-1]
                if request_id in response_queue:
                    response_queue[request_id] = payload
                    
        except Exception as e:
            print(f"[MQTT] Ошибка обработки сообщения: {e}")
    
    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        print(f"[MQTT] Отключение: {reason_code}")
        self.connected = False

# Глобальный экземпляр менеджера
mqtt_manager = MQTTManager()

async def wait_for_response_async(request_id: str, timeout: int = MQTT_TIMEOUT) -> Optional[str]:
    """Асинхронное ожидание ответа от MQTT"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if request_id in response_queue and response_queue[request_id] is not None:
            response = response_queue[request_id]
            del response_queue[request_id]
            return response
        await asyncio.sleep(0.05)  # Небольшая задержка для асинхронности
    
    # Таймаут
    if request_id in response_queue:
        del response_queue[request_id]
    return None

def publish_mqtt_request(intent_name: str, slots: List[Dict] = None, raw_input: str = "") -> Optional[str]:
    """Отправляет MQTT запрос и возвращает request_id"""
    client = mqtt_manager.get_client()
    if not client or not mqtt_manager.connected:
        return None
    
    request_id = f"{intent_name.lower()}_{int(time.time() * 1000)}"
    response_queue[request_id] = None
    
    payload = {
        "intent": {"intentName": intent_name},
        "request_id": request_id
    }
    
    if slots:
        payload["slots"] = slots
    if raw_input:
        payload["rawInput"] = raw_input
    
    try:
        client.publish(RECOGNIZED_INTENT_PATH, json.dumps(payload))
        return request_id
    except Exception as e:
        print(f"[MQTT] Ошибка отправки: {e}")
        if request_id in response_queue:
            del response_queue[request_id]
        return None

# --- Оптимизированные инструменты ---

async def tool_get_time_async():
    """Асинхронная версия получения времени"""
    if not MQTT_ENABLED:
        # Локальная реализация без MQTT
        now = datetime.now()
        return f"Текущее время {now.hour} часов, {now.minute} минут"
    
    request_id = publish_mqtt_request("GetTime")
    if not request_id:
        # Fallback к локальной реализации
        now = datetime.now()
        return f"Текущее время {now.hour} часов, {now.minute} минут"
    
    response = await wait_for_response_async(request_id)
    if response:
        try:
            data = json.loads(response)
            return data.get("text", "Не удалось получить время")
        except:
            return response
    
    # Fallback
    now = datetime.now()
    return f"Текущее время {now.hour} часов, {now.minute} минут"

def tool_get_time():
    """Синхронная версия для совместимости"""
    try:
        # Запускаем асинхронную версию в новом event loop
        return asyncio.run(tool_get_time_async())
    except Exception as e:
        print(f"[ERROR] tool_get_time: {e}")
        # Fallback
        now = datetime.now()
        return f"Текущее время {now.hour} часов, {now.minute} минут"

async def tool_set_timer_async(minutes: int = 0, seconds: int = 0, hours: int = 0):
    """Асинхронная версия установки таймера"""
    if not MQTT_ENABLED:
        # Локальная симуляция
        time_parts = []
        if hours > 0:
            time_parts.append(f"{hours} час" + ("а" if hours in [2,3,4] else "ов" if hours > 4 else ""))
        if minutes > 0:
            time_parts.append(f"{minutes} минут" + ("ы" if minutes in [2,3,4] else "" if minutes == 1 else ""))
        if seconds > 0:
            time_parts.append(f"{seconds} секунд" + ("ы" if seconds in [2,3,4] else "" if seconds == 1 else ""))
        
        return f"Таймер установлен на {' '.join(time_parts) if time_parts else '1 минуту'}"
    
    slots = []
    if hours > 0:
        slots.append({"slotName": "hour", "value": {"value": hours}})
    if minutes > 0:
        slots.append({"slotName": "minute", "value": {"value": minutes}})
    if seconds > 0:
        slots.append({"slotName": "second", "value": {"value": seconds}})
    
    request_id = publish_mqtt_request("SetTimer", slots)
    if not request_id:
        return "Не удалось установить таймер"
    
    response = await wait_for_response_async(request_id)
    if response:
        try:
            data = json.loads(response)
            return data.get("text", "Таймер установлен")
        except:
            return response
    
    return "Таймер установлен"

def tool_set_timer(minutes: int = 0, seconds: int = 0, hours: int = 0):
    """Синхронная версия для совместимости"""
    try:
        return asyncio.run(tool_set_timer_async(minutes, seconds, hours))
    except Exception as e:
        print(f"[ERROR] tool_set_timer: {e}")
        return f"Ошибка при установке таймера: {str(e)}"

async def tool_set_notification_async(text: str, minutes: int = 0, seconds: int = 0, hours: int = 0):
    """Асинхронная версия установки напоминания"""
    if not MQTT_ENABLED:
        # Локальная симуляция
        time_parts = []
        if hours > 0:
            time_parts.append(f"{hours} час" + ("а" if hours in [2,3,4] else "ов" if hours > 4 else ""))
        if minutes > 0:
            time_parts.append(f"{minutes} минут" + ("ы" if minutes in [2,3,4] else "" if minutes == 1 else ""))
        if seconds > 0:
            time_parts.append(f"{seconds} секунд" + ("ы" if seconds in [2,3,4] else "" if seconds == 1 else ""))
        
        if time_parts:
            return f"Хорошо, напомню через {' '.join(time_parts)} о том, что {text}"
        else:
            return f"Хорошо, напомню о том, что {text}"
    
    slots = []
    if hours > 0:
        slots.append({"slotName": "hour", "value": {"value": hours}})
    if minutes > 0:
        slots.append({"slotName": "minute", "value": {"value": minutes}})
    if seconds > 0:
        slots.append({"slotName": "second", "value": {"value": seconds}})
    
    raw_input = f"Напомни через {hours} часов {minutes} минут {seconds} секунд о том {text}"
    request_id = publish_mqtt_request("SetNotification", slots, raw_input)
    
    if not request_id:
        return f"Хорошо, напомню о том, что {text}"
    
    response = await wait_for_response_async(request_id)
    if response:
        try:
            data = json.loads(response)
            return data.get("text", "Напоминание установлено")
        except:
            return response
    
    return f"Хорошо, напомню о том, что {text}"

def tool_set_notification(text: str, minutes: int = 0, seconds: int = 0, hours: int = 0):
    """Синхронная версия для совместимости"""
    try:
        return asyncio.run(tool_set_notification_async(text, minutes, seconds, hours))
    except Exception as e:
        print(f"[ERROR] tool_set_notification: {e}")
        return f"Ошибка при установке напоминания: {str(e)}"

async def tool_get_weather_async():
    """Асинхронная версия получения погоды"""
    if not MQTT_ENABLED:
        return "Информация о погоде недоступна без MQTT подключения"
    
    request_id = publish_mqtt_request("GetWeather")
    if not request_id:
        return "Не удалось получить информацию о погоде"
    
    response = await wait_for_response_async(request_id)
    if response:
        try:
            data = json.loads(response)
            return data.get("text", "Не удалось получить информацию о погоде")
        except:
            return response
    
    return "Не удалось получить информацию о погоде"

def tool_get_weather():
    """Синхронная версия для совместимости"""
    try:
        return asyncio.run(tool_get_weather_async())
    except Exception as e:
        print(f"[ERROR] tool_get_weather: {e}")
        return f"Ошибка при получении погоды: {str(e)}"

async def tool_call_contact_async(contact_name: str):
    """Асинхронная версия звонка"""
    if not MQTT_ENABLED:
        return f"Симуляция звонка контакту {contact_name}"
    
    raw_input = f"позвони {contact_name}"
    request_id = publish_mqtt_request("InitiateCall", raw_input=raw_input)
    
    if not request_id:
        return f"Не удалось совершить звонок {contact_name}"
    
    response = await wait_for_response_async(request_id)
    if response:
        try:
            data = json.loads(response)
            return data.get("text", f"Звоню контакту {contact_name}")
        except:
            return response
    
    return f"Звоню контакту {contact_name}"

def tool_call_contact(contact_name: str):
    """Синхронная версия для совместимости"""
    try:
        return asyncio.run(tool_call_contact_async(contact_name))
    except Exception as e:
        print(f"[ERROR] tool_call_contact: {e}")
        return f"Ошибка при звонке контакту: {str(e)}"

# Определение инструментов для LangGraph (без изменений)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Получает текущее время",
            "parameters": {"type": "object", "properties": {}, "required": []}
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
                    "minutes": {"type": "integer", "description": "Количество минут"},
                    "seconds": {"type": "integer", "description": "Количество секунд"},
                    "hours": {"type": "integer", "description": "Количество часов"}
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
                    "text": {"type": "string", "description": "Текст напоминания"},
                    "minutes": {"type": "integer", "description": "Количество минут"},
                    "seconds": {"type": "integer", "description": "Количество секунд"},
                    "hours": {"type": "integer", "description": "Количество часов"}
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
            "parameters": {"type": "object", "properties": {}, "required": []}
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
                    "contact_name": {"type": "string", "description": "Имя контакта для звонка"}
                },
                "required": ["contact_name"]
            }
        }
    }
]

# Оптимизированный словарь для вызова функций
tool_mapping = {
    "get_time": tool_get_time,
    "set_timer": tool_set_timer,
    "set_notification": tool_set_notification,
    "get_weather": tool_get_weather,
    "call_contact": tool_call_contact
}

def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Выполняет инструмент по имени с указанными аргументами"""
    if tool_name in tool_mapping:
        try:
            return tool_mapping[tool_name](**tool_args)
        except Exception as e:
            print(f"[ERROR] execute_tool({tool_name}): {e}")
            # Возвращаем более дружелюбное сообщение об ошибке
            error_messages = {
                "get_time": "Не удалось получить текущее время",
                "set_timer": "Не удалось установить таймер",
                "set_notification": "Не удалось создать напоминание",
                "get_weather": "Не удалось получить информацию о погоде",
                "call_contact": "Не удалось совершить звонок"
            }
            return error_messages.get(tool_name, f"Ошибка при выполнении команды")
    else:
        return "Команда не найдена"

def init_mqtt():
    """Инициализация MQTT (опционально)"""
    if MQTT_ENABLED:
        mqtt_manager.get_client()
        print(f"[MQTT] Инициализация завершена. Статус: {'подключен' if mqtt_manager.connected else 'отключен'}")
    else:
        print("[MQTT] MQTT отключен в настройках")

def get_mqtt_status() -> Dict[str, Any]:
    """Возвращает статус MQTT подключения"""
    return {
        "enabled": MQTT_ENABLED,
        "connected": mqtt_manager.connected if mqtt_manager else False,
        "host": MQTT_HOST,
        "port": MQTT_PORT,
        "timeout": MQTT_TIMEOUT
    }

# Тестирование
if __name__ == "__main__":
    import asyncio
    
    async def test_tools():
        print("🧪 Тестирование оптимизированных MQTT инструментов\n")
        
        # Инициализация
        init_mqtt()
        await asyncio.sleep(1)  # Даем время на подключение
        
        print("Статус MQTT:", get_mqtt_status())
        print()
        
        # Тесты
        tests = [
            ("Получение времени", tool_get_time),
            ("Установка таймера", lambda: tool_set_timer(minutes=1)),
            ("Получение погоды", tool_get_weather),
        ]
        
        for test_name, test_func in tests:
            print(f"Тест: {test_name}")
            try:
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time
                print(f"  Результат: {result}")
                print(f"  Время: {duration:.2f}s")
            except Exception as e:
                print(f"  Ошибка: {e}")
            print()
        
        # Очистка
        if mqtt_manager.client:
            mqtt_manager.client.loop_stop()
            mqtt_manager.client.disconnect()
    
    asyncio.run(test_tools())