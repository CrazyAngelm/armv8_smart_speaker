import asyncio, os, websockets
from dataclasses import dataclass
from typing import Any, Literal, Optional, Dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from llm_module import LLMManager
import time
import json
from langgraph.prebuilt import ToolNode, tools_condition
# Импортируем mqtt_tools
from mqtt_tools import tools, execute_tool, init_mqtt
import re

load_dotenv()

# Инициализация MQTT при запуске
init_mqtt()

# ---------- Типы сообщений ----------
@dataclass
class AudioMsg:
    raw: bytes
    sr: int = 16000

@dataclass
class TextMsg:
    text: str

@dataclass
class AgentState:
    audio: Optional[AudioMsg] = None
    text:  Optional[TextMsg]  = None
    tool_calls: Optional[list] = None  # Добавляем поддержку вызовов инструментов
    tool_results: Optional[Dict[str, Any]] = None  # Результаты выполнения инструментов

# ---------- WebSocket клиенты ----------
STT_WS_HOST = os.getenv("STT_WS_HOST", "localhost") 
STT_WS_PORT = int(os.getenv("STT_WS_PORT", 8778))
TTS_WS_HOST = os.getenv("TTS_WS_HOST", "localhost")
TTS_WS_PORT = int(os.getenv("TTS_WS_PORT", 8777))

# Глобальный замок для предотвращения перекрытия аудио
processing_lock = asyncio.Lock()

# ---------- STT клиент ----------
async def stt_vosk(audio: AudioMsg) -> str:
    t0 = time.perf_counter()
    print(f"[LOG] [STT] Отправка аудио ({len(audio.raw)} байт, sr={audio.sr}) в Vosk STT WS")
    try:
        async with websockets.connect(f"ws://{STT_WS_HOST}:{STT_WS_PORT}", max_size=8*2**20) as ws:
            await ws.send(audio.raw)
            resp = await ws.recv()
            if isinstance(resp, str) and not resp.startswith("ERROR"):
                t1 = time.perf_counter()
                print(f"[PROFILE] STT (Vosk): {t1-t0:.2f} сек")
                return resp
            raise RuntimeError(f"Vosk STT WS error: {resp}")
    except Exception as e:
        print(f"[ERROR] Vosk STT connection error: {e}")
        raise

# ---------- Вспомогательная функция для TTS ----------
def extract_tts_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # Удаляем содержимое <think>...</think> вместе с тегами
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    if text.startswith('[') or text.startswith('{'):
        try:
            data = json.loads(text)
            extracted_text = ""
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'text' in item:
                        extracted_text += item['text'] + " "
                    elif isinstance(item, dict) and item.get('type') == 'tool_use':
                        continue
                    elif isinstance(item, str):
                        extracted_text += item + " "
            elif isinstance(data, dict):
                if 'text' in data:
                    extracted_text = data['text']
                elif 'content' in data:
                    content = data['content']
                    if isinstance(content, str):
                        extracted_text = content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                extracted_text += item['text'] + " "
                            elif isinstance(item, str):
                                extracted_text += item + " "
            if extracted_text:
                print(f"[LOG] [TTS] Успешно извлечен текст из структуры: {extracted_text}...")
                return extracted_text.strip()
        except Exception as e:
            print(f"[LOG] [TTS] Ошибка при извлечении текста из структуры: {e}")
    return text.strip()

# ---------- TTS клиент ----------
async def tts_client(text: str) -> bytes:
    t0 = time.perf_counter()
    try:
        text = extract_tts_text(text)
    except Exception as e:
        print(f"[ERROR] [TTS] Ошибка при обработке текста: {e}")
    print(f"[LOG] [TTS] Отправка текста в TTS: {text}...")
    try:
        async with websockets.connect(f"ws://{TTS_WS_HOST}:{TTS_WS_PORT}", max_size=8*2**20) as ws:
            await ws.send(text)
            resp = await ws.recv()
            if isinstance(resp, bytes):
                t1 = time.perf_counter()
                print(f"[PROFILE] TTS (Piper): {t1-t0:.2f} сек")
                return resp
            raise RuntimeError(f"TTS WS error: {resp}")
    except Exception as e:
        print(f"[ERROR] TTS connection error: {e}")
        raise

# Получаем системный промпт
def get_system_prompt():
    return (
        "Ты — умный голосовой помощник для компании. "
        "Ты локальная модель, и пишешь ответ в мужском роде и на русском языке. "
        "Отвечай четко, коротко, по делу и профессионально. "
        "Если не знаешь ответ, честно скажи об этом.\n\n"
        "ВАЖНО: Если пользователь просит выполнить одно из следующих действий, "
        "ОБЯЗАТЕЛЬНО используй соответствующую функцию вместо собственного ответа:\n"
        "- Узнать время → вызови get_time\n"
        "- Поставить таймер → вызови set_timer\n"
        "- Поставить напоминание/уведомление → вызови set_notification\n"
        "- Узнать погоду → вызови get_weather\n"
        "- Позвонить кому-то → вызови call_contact\n\n"
        "Функции вернут готовый ответ для пользователя, тебе НЕ нужно "
        "добавлять к нему свои комментарии или пояснения.\n\n"
        "Для всех остальных вопросов отвечай самостоятельно."
        "Для самостоятельных ответов используй только знаки препинания: .,!? другие особые символы не используй."
    )

# Инициализация менеджера LLM
llm_manager = LLMManager()
# Получаем информацию о текущем провайдере
llm_info = llm_manager.get_provider_info()
SHOW_TEXT = os.getenv("SHOW_TEXT", "true").lower() == "true"
if SHOW_TEXT:
    print(f"[INFO] Using LLM: {llm_info['provider']} ({llm_info['model']})")

async def stt_node(state: AgentState) -> AgentState:
    if state.audio:
        print(f"[LOG] [STT_NODE] Получено аудио: {len(state.audio.raw)} байт")
        try:
            recognized_text = await stt_vosk(state.audio)
            if not recognized_text or recognized_text.strip() == "Не удалось распознать речь":
                print(f"[LOG] [STT_NODE] Аудио не содержит речи или содержит только фоновый шум")
                state.text = None  # Не заполняем текст, чтобы остановить обработку
                return state
            print(f"[LOG] [STT_NODE] Распознанный текст: '{recognized_text}'")
            state.text = TextMsg(recognized_text)
            print(f"[INFO] Распознан текст: {recognized_text}")
        except Exception as e:
            print(f"[ERROR] STT node error: {e}")
            state.text = TextMsg("Ошибка распознавания речи")
    else:
        print("[LOG] [STT_NODE] Аудио отсутствует в состоянии")
    return state

async def llm_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    if not state.text:
        return state
    txt = state.text.text
    print(f"[LOG] [LLM] Отправка текста в LLM: {txt}")
    
    try:
        # Получаем системный промпт
        current_system_prompt = get_system_prompt()
        t_llm0 = time.perf_counter()
        
        # Привязываем инструменты к LLM
        llm_with_tools = llm_manager.llm.bind_tools(tools)
        
        # Используем LLM с инструментами для генерации ответа
        result = await llm_with_tools.ainvoke([
            {"role": "system", "content": current_system_prompt},
            {"role": "user", "content": txt}
        ])
        
        t_llm1 = time.perf_counter()
        print(f"[PROFILE] LLM tool invocation: {t_llm1-t_llm0:.2f} сек")
        
        # Проверяем наличие tool_calls в ответе
        if hasattr(result, 'tool_calls') and result.tool_calls:
            print(f"[LOG] [LLM] LLM вызвал инструменты: {result.tool_calls}")
            state.tool_calls = result.tool_calls
            return state
            
        # Если нет вызовов инструментов, используем обычный ответ
        reply = result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        print(f"[ERROR] LLM error: {e}")
        reply = "Извините, произошла ошибка. Можете повторить ваш вопрос?"
    t1 = time.perf_counter()
    print(f"[PROFILE] LLM node (всего): {t1-t0:.2f} сек")
    state.text = TextMsg(reply)
    return state

# Узел для выполнения инструментов
async def tools_node(state: AgentState) -> AgentState:
    """Выполняет инструменты, вызванные LLM"""
    if not state.tool_calls:
        return state
    
    print(f"[LOG] [TOOLS] Начинаю выполнение {len(state.tool_calls)} инструментов")
    results = {}
    
    for tool_call in state.tool_calls:
        tool_name = tool_call.get("name", None)
        if not tool_name:
            print(f"[ERROR] Инструмент не содержит имя: {tool_call}")
            continue
            
        # Извлекаем аргументы
        try:
            tool_args = tool_call.get("args", {})
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)
        except Exception as e:
            print(f"[ERROR] Ошибка при разборе аргументов инструмента: {e}")
            tool_args = {}
        
        print(f"[LOG] [TOOLS] Выполнение инструмента: {tool_name} с аргументами {tool_args}")
        
        # Выполняем инструмент
        result = execute_tool(tool_name, tool_args)
        results[tool_call.get("id", f"tool_{len(results)}")] = result
        print(f"[LOG] [TOOLS] Результат выполнения {tool_name}: {result}")
    
    state.tool_results = results
    return state

# Узел для обработки результатов инструментов
async def tool_results_processor(state: AgentState) -> AgentState:
    """Обрабатывает результаты выполнения инструментов и формирует ответ"""
    if not state.tool_results:
        return state
    
    try:
        # Оптимизация: используем результат инструмента напрямую
        # Инструменты уже возвращают готовые для озвучивания тексты
        
        # Собираем все результаты инструментов
        responses = []
        for tool_id, result in state.tool_results.items():
            result_text = str(result).strip()
            if result_text:
                responses.append(result_text)
        
        # Формируем финальный ответ
        if responses:
            # Если несколько результатов, объединяем их
            if len(responses) == 1:
                response_text = responses[0]
            else:
                response_text = ". ".join(responses)
            
            print(f"[LOG] [TOOL_RESULTS] Прямой ответ от инструмента(ов): {response_text}")
        else:
            response_text = "Команда выполнена"
            print(f"[LOG] [TOOL_RESULTS] Использую стандартный ответ: {response_text}")
        
        state.text = TextMsg(response_text)
        
        # Очищаем информацию об инструментах, так как обработка завершена
        state.tool_calls = None
        state.tool_results = None
        
    except Exception as e:
        print(f"[ERROR] Ошибка при обработке результатов инструментов: {e}")
        # Используем результат инструмента, если что-то пошло не так
        if state.tool_results and len(state.tool_results) > 0:
            first_result = next(iter(state.tool_results.values()))
            state.text = TextMsg(str(first_result))
        else:
            state.text = TextMsg(f"Произошла ошибка при обработке результатов: {str(e)}")
    
    return state

async def tts_node(state: AgentState) -> AgentState:
    if state.text:
        try:
            audio_bytes = await tts_client(state.text.text)
            state.audio = AudioMsg(audio_bytes, sr=48000)
        except Exception as e:
            print(f"[ERROR] TTS error: {e}")
    return state

# ---------- Функция-маршрутизатор для инструментов ----------
def tools_router(state: AgentState) -> Literal["tools", "tool_results_processor", END]:
    """Маршрутизирует запросы на основе наличия вызовов инструментов"""
    if state.tool_calls:
        return "tools"
    elif state.tool_results:
        return "tool_results_processor"
    return END

# ---------- Функция‑маршрутизатор ----------
def wake_router(state: AgentState) -> Literal["llm", END]:
    """Всегда переходим к узлу 'llm' для обработки входящего текста."""
    if state.text:
        return "llm"
    return END

# ---------- Построение графа ----------
workflow = StateGraph(AgentState)
workflow.add_node("stt", stt_node)
workflow.add_node("llm", llm_node)
workflow.add_node("tools", tools_node)
workflow.add_node("tool_results_processor", tool_results_processor)
workflow.add_node("tts", tts_node)

workflow.add_edge(START, "stt")      # начало → STT
workflow.add_conditional_edges(      # STT → (LLM или END)
    "stt",
    wake_router                      # callable‑router
)

# Добавляем условный переход от LLM к обработчику инструментов или TTS
workflow.add_conditional_edges(
    "llm",
    tools_router
)

# Добавляем переход от инструментов к обработке результатов
workflow.add_edge("tools", "tool_results_processor")

# Добавляем переход от результатов инструментов к TTS
workflow.add_edge("tool_results_processor", "tts")

app = workflow.compile()

# ---------- WebSocket сервер ----------
HOST, PORT = os.getenv("MAGUS_WS_HOST","0.0.0.0"), int(os.getenv("MAGUS_WS_PORT",8765))

# Функция для разбиения больших аудио на части
def split_audio_data(audio_data: bytes, max_chunk_size: int = 1024 * 1024) -> list:
    """Разбивает большие аудио-данные на части не более max_chunk_size байт."""
    chunks = []
    for i in range(0, len(audio_data), max_chunk_size):
        chunks.append(audio_data[i:i + max_chunk_size])
    return chunks

async def handle(ws):
    audio_chunks = []
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                print(f"[LOG] [WS] Получен фрагмент аудио: {len(msg)} байт")
                audio_chunks.append(msg)
            elif isinstance(msg, str):
                print(f"[LOG] [WS] Получена команда: {msg}")
                if msg.strip().upper() == "END":
                    # Проверяем, можно ли обработать запрос (используем неблокирующую проверку)
                    if processing_lock.locked():
                        print(f"[LOG] [WS] Обработка предыдущего запроса еще не завершена, игнорируем новый запрос")
                        await ws.send("BUSY")
                        audio_chunks = []  # Очищаем буфер
                        continue
                    
                    # Собираем аудио и обрабатываем
                    audio_data = b"".join(audio_chunks)
                    print(f"[LOG] [WS] Собрано аудио: {len(audio_data)} байт")
                    if not audio_data:
                        await ws.send("ERROR: No audio data received")
                        audio_chunks = []
                        continue
                    
                    # Используем замок для предотвращения параллельной обработки
                    async with processing_lock:
                        state = AgentState(audio=AudioMsg(audio_data))
                        print(f"[LOG] [AGENT] Создано начальное состояние с аудио {len(audio_data)} байт")
                        try:
                            # Вызов LangGraph приложения
                            print(f"[LOG] [AGENT] Запуск обработки через LangGraph...")
                            result = await app.ainvoke(state)
                            print(f"[LOG] [AGENT] Обработка завершена. Тип результата: {type(result)}")
                            
                            # Извлекаем аудио ответ или текст для синтеза ответа
                            audio_result = None
                            text_to_speak = None
                            
                            # Проверяем, есть ли аудио в результате
                            if hasattr(result, 'values'):
                                print(f"[LOG] [AGENT] Анализ результата...")
                                values_dict = dict(result)
                                print(f"[LOG] [AGENT] Ключи результата: {list(values_dict.keys())}")
                                
                                # Ищем аудио в структуре результата
                                for key, value in values_dict.items():
                                    print(f"[LOG] [AGENT] Проверка ключа {key}, тип: {type(value)}")
                                    if hasattr(value, 'audio') and value.audio is not None:
                                        print(f"[LOG] [AGENT] Найдено аудио в ключе {key}")
                                        audio_result = value.audio
                                        break
                                
                                # Если не нашли аудио, ищем текст для озвучивания
                                if not audio_result:
                                    print(f"[LOG] [AGENT] Аудио не найдено, ищем текст для синтеза...")
                                    for key, value in values_dict.items():
                                        # Проверяем, является ли значение TextMsg
                                        if hasattr(value, 'text') and value.text and hasattr(value.text, 'text'):
                                            print(f"[LOG] [AGENT] Найден TextMsg в ключе {key}")
                                            text_to_speak = value.text.text
                                            break
                                        # Проверяем, является ли значение строкой
                                        elif isinstance(value, str):
                                            print(f"[LOG] [AGENT] Найдена строка в ключе {key}")
                                            text_to_speak = value
                                            break
                                        # Проверяем, является ли само значение TextMsg
                                        elif isinstance(value, TextMsg):
                                            print(f"[LOG] [AGENT] Найден прямой TextMsg в ключе {key}")
                                            text_to_speak = value.text
                                            break
                                    
                                    # Проверяем, получили ли мы текст
                                    if text_to_speak:
                                        print(f"[LOG] [AGENT] Синтезируем речь из найденного текста: {text_to_speak}")
                                        try:
                                            audio_bytes = await tts_client(text_to_speak)
                                            audio_result = AudioMsg(audio_bytes, sr=48000)
                                            print(f"[LOG] [AGENT] Успешно синтезирована речь, размер аудио: {len(audio_bytes)} байт")
                                        except Exception as e:
                                            print(f"[ERROR] [AGENT] Ошибка при синтезе речи: {e}")
                                            await ws.send(f"ERROR: Ошибка при синтезе речи: {e}")
                                            audio_chunks = []
                                            continue
                            
                            # Если аудио найдено, отправляем его (с разбивкой на части, если нужно)
                            if audio_result:
                                audio_size = len(audio_result.raw)
                                print(f"[LOG] [WS] Отправка аудио-ответа: {audio_size} байт")
                                
                                # Проверяем, нужно ли разбивать аудио на части
                                if audio_size > 1024 * 1024:  # Если больше 1 МБ
                                    print(f"[LOG] [WS] Аудио слишком большое, разбиваем на части")
                                    
                                    # Отправляем клиенту команду, что начинаем передачу частей
                                    await ws.send("AUDIO_CHUNKS_BEGIN")
                                    
                                    # Разбиваем и отправляем части
                                    chunks = split_audio_data(audio_result.raw)
                                    for i, chunk in enumerate(chunks):
                                        print(f"[LOG] [WS] Отправка части {i+1}/{len(chunks)}, размер: {len(chunk)} байт")
                                        await ws.send(chunk)
                                    
                                    # Отправляем сигнал завершения передачи
                                    await ws.send("AUDIO_CHUNKS_END")
                                else:
                                    # Отправляем целиком, если маленькое
                                    await ws.send(audio_result.raw)
                            else:
                                print("[ERROR] [WS] Аудио-ответ не найден и не удалось синтезировать речь!")
                                # Делаем фиктивный звуковой сигнал
                                await ws.send(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
                            
                        except Exception as e:
                            print(f"[ERROR] Error processing request: {e}")
                            await ws.send(f"ERROR: {e}")
                    
                    # Очищаем буфер аудио
                    audio_chunks = []
                else:
                    await ws.send("ACK")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

async def main_ws():
    print(f"[WS] Serving on ws://{HOST}:{PORT}")
    async with websockets.serve(
        handle, HOST, PORT, max_size=8*2**20,
        ping_interval=300,  # увеличиваем до 5 минут
        ping_timeout=None   # отключаем таймаут полностью
    ):
        await asyncio.Future()

async def cli_loop():
    print("\n[CLI] Умный голосовой помощник — текстовый режим. Введите 'exit' для выхода.\n")
    while True:
        try:
            user_input = input("Вы: ").strip()
            if user_input.lower() in ("exit", "quit", "выход"):
                print("[CLI] Завершение работы.")
                break
            if not user_input:
                continue
            # Создаем состояние агента с текстом
            state = AgentState(text=TextMsg(user_input))
            
            # Запускаем обработку через граф
            print(f"[LOG] [CLI] Запуск обработки через LangGraph...")
            result = await app.ainvoke(state)
            
            # Извлекаем ответ
            response_text = None
            if hasattr(result, 'values'):
                for key, value in dict(result).items():
                    # Проверяем, является ли значение TextMsg объектом
                    if hasattr(value, 'text') and value.text:
                        if isinstance(value.text, TextMsg):
                            response_text = value.text.text
                        elif isinstance(value.text, str):
                            response_text = value.text
                        break
                    # Если значение строка, используем напрямую
                    elif isinstance(value, str):
                        response_text = value
                        break
                    # Проверяем, является ли само значение TextMsg
                    elif isinstance(value, TextMsg):
                        response_text = value.text
                        break
            
            if response_text:
                print(f"Ассистент: {response_text}")
            else:
                print("Ассистент: [Нет ответа]")
                
        except (KeyboardInterrupt, EOFError):
            print("\n[CLI] Завершение работы.")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main_ws())
    except KeyboardInterrupt:
        print("Interrupted.")
