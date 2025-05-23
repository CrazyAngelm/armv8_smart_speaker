import asyncio, os, websockets
from dataclasses import dataclass
from typing import Any, Literal, Optional, Dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from llm_module import LLMManager
import time
import json
from mqtt_tools import tools, execute_tool, init_mqtt
import re
import hashlib
import socket

load_dotenv()
init_mqtt()

# Мониторинг производительности
class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
        self.enabled = os.getenv("PERF_MONITOR", "true").lower() == "true"
    
    def start(self, phase: str):
        if self.enabled:
            self.timings[f"{phase}_start"] = time.perf_counter()
    
    def end(self, phase: str):
        if self.enabled and f"{phase}_start" in self.timings:
            duration = time.perf_counter() - self.timings[f"{phase}_start"]
            print(f"[PERF] {phase}: {duration:.2f}s")
            return duration

perf = PerformanceMonitor()

# Типы сообщений
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
    text: Optional[TextMsg] = None
    tool_calls: Optional[list] = None
    tool_results: Optional[Dict[str, Any]] = None

# WebSocket настройки
STT_WS_HOST = os.getenv("STT_WS_HOST", "localhost") 
STT_WS_PORT = int(os.getenv("STT_WS_PORT", 8778))
TTS_WS_HOST = os.getenv("TTS_WS_HOST", "localhost")
TTS_WS_PORT = int(os.getenv("TTS_WS_PORT", 8777))

processing_lock = asyncio.Lock()

# STT клиент
async def stt_vosk(audio: AudioMsg) -> str:
    print(f"[LOG] [STT] Отправка аудио ({len(audio.raw)} байт)")
    try:
        async with websockets.connect(f"ws://{STT_WS_HOST}:{STT_WS_PORT}", max_size=8*2**20) as ws:
            await ws.send(audio.raw)
            resp = await ws.recv()
            if isinstance(resp, str) and not resp.startswith("ERROR"):
                return resp
            raise RuntimeError(f"STT error: {resp}")
    except Exception as e:
        print(f"[ERROR] STT error: {e}")
        raise

# Извлечение текста для TTS
def extract_tts_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # Удаляем <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Парсим JSON если есть
    if text.startswith('[') or text.startswith('{'):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return " ".join(item.get('text', str(item)) for item in data if isinstance(item, dict) and item.get('type') != 'tool_use')
            elif isinstance(data, dict):
                return data.get('text', data.get('content', str(data)))
        except:
            pass
    return text.strip()

# TTS клиент
async def tts_client(text: str) -> bytes:
    text = extract_tts_text(text)
    print(f"[LOG] [TTS] Синтез: {text[:100]}...")
    try:
        async with websockets.connect(f"ws://{TTS_WS_HOST}:{TTS_WS_PORT}", max_size=8*2**20) as ws:
            await ws.send(text)
            resp = await ws.recv()
            if isinstance(resp, bytes):
                return resp
            raise RuntimeError(f"TTS error: {resp}")
    except Exception as e:
        print(f"[ERROR] TTS error: {e}")
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

llm_manager = LLMManager()
llm_cache = {}

def get_cache_key(prompt: str, system_prompt: str) -> str:
    return hashlib.md5(f"{system_prompt}|{prompt}".encode()).hexdigest()

def get_cached_response(prompt: str, system_prompt: str) -> Optional[str]:
    cache_key = get_cache_key(prompt, system_prompt)
    return llm_cache.get(cache_key)

def cache_response(prompt: str, system_prompt: str, response: str):
    cache_key = get_cache_key(prompt, system_prompt)
    llm_cache[cache_key] = response
    if len(llm_cache) > 100:
        oldest_key = next(iter(llm_cache))
        del llm_cache[oldest_key]

def get_system_prompt():
    return ("Ты — умный голосовой помощник. Отвечай кратко и по делу. "
            "ВСЕГДА используй доступные функции для получения актуальной информации: "
            "- get_time() для времени "
            "- set_timer() для таймеров "
            "- set_notification() для напоминаний "
            "- get_weather() для погоды "
            "- call_contact() для звонков. "
            "НЕ спрашивай разрешения - сразу вызывай нужную функцию!")

# Предзагрузка моделей
async def preload_models():
    """Предзагружает модели для быстрого первого отклика"""
    print("[INFO] Предзагрузка моделей...")
    
    try:
        # Тест STT
        test_audio = AudioMsg(b'\x00' * 1600, sr=16000)
        await stt_vosk(test_audio)
        print("[INFO] STT готов")
    except:
        print("[WARNING] STT недоступен")
    
    try:
        # Тест TTS
        await tts_client("Тест")
        print("[INFO] TTS готов")
    except:
        print("[WARNING] TTS недоступен")
    
    print("[INFO] Предзагрузка завершена")

# Узлы обработки
async def stt_node(state: AgentState) -> AgentState:
    perf.start("stt")
    if state.audio:
        try:
            recognized_text = await stt_vosk(state.audio)
            if recognized_text and recognized_text.strip() != "Не удалось распознать речь":
                state.text = TextMsg(recognized_text)
                print(f"[INFO] Распознан текст: {recognized_text}")
            else:
                state.text = None
        except Exception as e:
            print(f"[ERROR] STT error: {e}")
            state.text = TextMsg("Ошибка распознавания речи")
    perf.end("stt")
    return state

async def llm_node(state: AgentState) -> AgentState:
    perf.start("llm")
    if not state.text:
        perf.end("llm")
        return state
    
    txt = state.text.text
    system_prompt = get_system_prompt()
    
    # Проверяем кэш
    cached = get_cached_response(txt, system_prompt)
    if cached:
        state.text = TextMsg(cached)
        perf.end("llm")
        return state
    
    try:
        print(f"[DEBUG] LLM узел: проверка поддержки bind_tools")
        if hasattr(llm_manager.llm, 'bind_tools'):
            print(f"[DEBUG] bind_tools поддерживается, привязываем {len(tools)} инструментов")
            llm_with_tools = llm_manager.llm.bind_tools(tools)
            result = await llm_with_tools.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": txt}
            ])
            
            print(f"[DEBUG] Результат LLM: type={type(result)}")
            print(f"[DEBUG] hasattr tool_calls: {hasattr(result, 'tool_calls')}")
            
            if hasattr(result, 'tool_calls') and result.tool_calls:
                print(f"[DEBUG] Найдено {len(result.tool_calls)} tool_calls: {result.tool_calls}")
                state.tool_calls = result.tool_calls
                perf.end("llm")
                return state
            else:
                print(f"[DEBUG] tool_calls не найдены или пусты")
                # Попробуем парсить текст для поиска упоминаний функций
                content = result.content if hasattr(result, 'content') else str(result)
                print(f"[DEBUG] Контент ответа: {content}")
                
                # Ищем упоминания функций в тексте
                parsed_tools = parse_tool_mentions(content)
                if parsed_tools:
                    print(f"[DEBUG] Найдены упоминания инструментов в тексте: {parsed_tools}")
                    state.tool_calls = parsed_tools
                    perf.end("llm")
                    return state
        else:
            print(f"[DEBUG] bind_tools НЕ поддерживается")
            result = await llm_manager.llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": txt}
            ])
            content = result.content if hasattr(result, 'content') else str(result)
            
            # Ищем упоминания функций в тексте
            parsed_tools = parse_tool_mentions(content)
            if parsed_tools:
                print(f"[DEBUG] Найдены упоминания инструментов в тексте: {parsed_tools}")
                state.tool_calls = parsed_tools
                perf.end("llm")
                return state
        
        reply = result.content if hasattr(result, 'content') else str(result)
        cache_response(txt, system_prompt, reply)
        state.text = TextMsg(reply)
        
    except Exception as e:
        print(f"[ERROR] LLM error: {e}")
        import traceback
        traceback.print_exc()
        state.text = TextMsg("Извините, произошла ошибка.")
    
    perf.end("llm")
    return state

def parse_tool_mentions(content: str) -> Optional[list]:
    """Парсит текст для поиска упоминаний функций и создает tool_calls"""
    import re
    
    # Паттерны для поиска упоминаний функций
    patterns = {
        "get_time": r"(?:время|сколько времени|который час)",
        "get_weather": r"(?:погода|weather|температура|градус)",
        "set_timer": r"(?:таймер|timer|поставь таймер|установи таймер)",
        "set_notification": r"(?:напомни|напоминание|notification|reminder)",
        "call_contact": r"(?:позвони|звони|call|вызов)"
    }
    
    tool_calls = []
    
    for tool_name, pattern in patterns.items():
        if re.search(pattern, content.lower()):
            print(f"[DEBUG] Найдено упоминание {tool_name} в тексте")
            
            # Создаем базовые аргументы для разных функций
            args = {}
            if tool_name == "set_timer":
                # Ищем числа для таймера
                minutes_match = re.search(r"(\d+)\s*мин", content)
                seconds_match = re.search(r"(\d+)\s*сек", content)
                hours_match = re.search(r"(\d+)\s*час", content)
                
                if minutes_match:
                    args["minutes"] = int(minutes_match.group(1))
                if seconds_match:
                    args["seconds"] = int(seconds_match.group(1))
                if hours_match:
                    args["hours"] = int(hours_match.group(1))
                    
                # Если не нашли конкретное время, ставим 1 минуту по умолчанию
                if not args:
                    args["minutes"] = 1
                    
            elif tool_name == "set_notification":
                # Ищем текст напоминания
                reminder_match = re.search(r"напомни.*?(?:о том|что)\s+(.+)", content, re.IGNORECASE)
                if reminder_match:
                    args["text"] = reminder_match.group(1).strip()
                else:
                    args["text"] = "Напоминание"
                    
                # Ищем время
                minutes_match = re.search(r"(\d+)\s*мин", content)
                if minutes_match:
                    args["minutes"] = int(minutes_match.group(1))
                else:
                    args["minutes"] = 5  # По умолчанию 5 минут
                    
            elif tool_name == "call_contact":
                # Ищем имя контакта
                contact_match = re.search(r"позвони\s+(.+)", content, re.IGNORECASE)
                if contact_match:
                    args["contact_name"] = contact_match.group(1).strip()
                else:
                    args["contact_name"] = "неизвестный контакт"
            
            tool_call = {
                "name": tool_name,
                "args": args,
                "id": f"tool_{tool_name}_{int(time.time())}"
            }
            tool_calls.append(tool_call)
            break  # Берем только первую найденную функцию
    
    return tool_calls if tool_calls else None

# Streaming LLM узел
async def llm_streaming_node(state: AgentState) -> AgentState:
    perf.start("llm_streaming")
    if not state.text:
        perf.end("llm_streaming")
        return state
    
    txt = state.text.text
    system_prompt = get_system_prompt()
    
    # Проверяем кэш
    cached = get_cached_response(txt, system_prompt)
    if cached:
        state.text = TextMsg(cached)
        perf.end("llm_streaming")
        return state
    
    try:
        # Используем потоковую генерацию если доступна
        if hasattr(llm_manager, 'stream_response_with_early_synthesis'):
            accumulated_response = ""
            async for text_chunk, should_start_tts in llm_manager.stream_response_with_early_synthesis(
                txt, system_prompt, min_chars_for_tts=30
            ):
                accumulated_response += text_chunk
                # Можно добавить логику раннего TTS здесь если нужно
            
            # Проверяем упоминания инструментов в накопленном тексте
            parsed_tools = parse_tool_mentions(accumulated_response)
            if parsed_tools:
                print(f"[DEBUG] В streaming найдены инструменты: {parsed_tools}")
                state.tool_calls = parsed_tools
                perf.end("llm_streaming")
                return state
            
            cache_response(txt, system_prompt, accumulated_response)
            state.text = TextMsg(accumulated_response)
        else:
            # Fallback к обычному LLM
            return await llm_node(state)
        
    except Exception as e:
        print(f"[ERROR] LLM streaming error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback к обычному LLM
        return await llm_node(state)
    
    perf.end("llm_streaming")
    return state

async def tools_node(state: AgentState) -> AgentState:
    if not state.tool_calls:
        return state
    
    perf.start("tools")
    print(f"[LOG] [TOOLS] Выполнение {len(state.tool_calls)} инструментов")
    
    async def execute_tool_async(tool_call):
        # Обрабатываем разные форматы tool_call
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", f"tool_{tool_name}_{int(time.time())}")
        else:
            # Если это объект с атрибутами (стандартный формат LangChain)
            tool_name = getattr(tool_call, "name", None)
            tool_args = getattr(tool_call, "args", {})
            tool_id = getattr(tool_call, "id", f"tool_{tool_name}_{int(time.time())}")
        
        if not tool_name:
            print(f"[ERROR] Не найдено имя инструмента в: {tool_call}")
            return None
        
        print(f"[DEBUG] Выполняю инструмент: {tool_name} с аргументами: {tool_args}")
        
        try:
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)
        except:
            tool_args = {}
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, execute_tool, tool_name, tool_args)
        print(f"[DEBUG] Результат инструмента {tool_name}: {result}")
        return (tool_id, result)
    
    tasks = [execute_tool_async(tc) for tc in state.tool_calls]
    results = await asyncio.gather(*tasks)
    state.tool_results = {id_: res for id_, res in results if id_ is not None}
    
    perf.end("tools")
    return state

async def tool_results_processor(state: AgentState) -> AgentState:
    if not state.tool_results:
        return state
    
    perf.start("tool_results")
    
    if len(state.tool_results) == 1:
        result = next(iter(state.tool_results.values()))
    else:
        result = "\n".join(str(r) for r in state.tool_results.values())
    
    state.text = TextMsg(str(result))
    state.tool_calls = None
    state.tool_results = None
    
    perf.end("tool_results")
    return state

async def tts_node(state: AgentState) -> AgentState:
    perf.start("tts")
    if state.text:
        try:
            audio_bytes = await tts_client(state.text.text)
            state.audio = AudioMsg(audio_bytes, sr=48000)
        except Exception as e:
            print(f"[ERROR] TTS error: {e}")
    perf.end("tts")
    return state

# Маршрутизаторы
def tools_router(state: AgentState) -> Literal["tools", "tool_results_processor", "tts"]:
    if state.tool_calls:
        return "tools"
    elif state.tool_results:
        return "tool_results_processor"
    return "tts"

def llm_type_router(state: AgentState) -> Literal["llm_streaming", "llm"]:
    """Выбирает streaming или обычный LLM"""
    if not state.text:
        return "llm"
    
    use_streaming = os.getenv("LLM_STREAMING", "true").lower() == "true"
    return "llm_streaming" if use_streaming else "llm"

# Построение графа
workflow = StateGraph(AgentState)
workflow.add_node("stt", stt_node)
workflow.add_node("llm", llm_node)
workflow.add_node("llm_streaming", llm_streaming_node)
workflow.add_node("tools", tools_node)
workflow.add_node("tool_results_processor", tool_results_processor)
workflow.add_node("tts", tts_node)

workflow.add_edge(START, "stt")
workflow.add_conditional_edges("stt", lambda state: llm_type_router(state) if state.text else "tts", 
                               {"llm_streaming": "llm_streaming", "llm": "llm", "tts": "tts"})
workflow.add_conditional_edges("llm", tools_router, {"tools": "tools", "tool_results_processor": "tool_results_processor", "tts": "tts"})
workflow.add_conditional_edges("llm_streaming", tools_router, {"tools": "tools", "tool_results_processor": "tool_results_processor", "tts": "tts"})
workflow.add_edge("tools", "tool_results_processor")
workflow.add_edge("tool_results_processor", "tts")
workflow.add_edge("tts", END)

app = workflow.compile()

# WebSocket сервер
HOST, PORT = os.getenv("MAGUS_WS_HOST", "0.0.0.0"), int(os.getenv("MAGUS_WS_PORT", 8765))

def split_audio_data(audio_data: bytes, max_chunk_size: int = 1024 * 1024) -> list:
    return [audio_data[i:i + max_chunk_size] for i in range(0, len(audio_data), max_chunk_size)]

async def handle(ws):
    audio_chunks = []
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                audio_chunks.append(msg)
            elif isinstance(msg, str) and msg.strip().upper() == "END":
                if processing_lock.locked():
                    await ws.send("BUSY")
                    audio_chunks = []
                    continue
                
                audio_data = b"".join(audio_chunks)
                if not audio_data:
                    await ws.send("ERROR: No audio data")
                    continue
                
                async with processing_lock:
                    state = AgentState(audio=AudioMsg(audio_data))
                    try:
                        result = await app.ainvoke(state)
                        
                        # Извлекаем аудио результат
                        audio_result = None
                        for value in dict(result).values():
                            if hasattr(value, 'audio') and value.audio:
                                audio_result = value.audio
                                break
                        
                        if not audio_result:
                            # Синтезируем из текста
                            for value in dict(result).values():
                                text_to_speak = None
                                if hasattr(value, 'text') and value.text:
                                    text_to_speak = value.text.text if hasattr(value.text, 'text') else value.text
                                elif isinstance(value, str):
                                    text_to_speak = value
                                elif isinstance(value, TextMsg):
                                    text_to_speak = value.text
                                
                                if text_to_speak:
                                    audio_bytes = await tts_client(text_to_speak)
                                    audio_result = AudioMsg(audio_bytes, sr=48000)
                                    break
                        
                        # Отправляем аудио
                        if audio_result:
                            if len(audio_result.raw) > 1024 * 1024:
                                await ws.send("AUDIO_CHUNKS_BEGIN")
                                for chunk in split_audio_data(audio_result.raw):
                                    await ws.send(chunk)
                                await ws.send("AUDIO_CHUNKS_END")
                            else:
                                await ws.send(audio_result.raw)
                        else:
                            await ws.send(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
                        
                    except Exception as e:
                        print(f"[ERROR] Processing error: {e}")
                        await ws.send(f"ERROR: {e}")
                
                audio_chunks = []
            else:
                await ws.send("ACK")
    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")

async def main_ws():
    await preload_models()
    print(f"[WS] Serving on ws://{HOST}:{PORT}")
    
    # Пытаемся запустить сервер с обработкой ошибки занятого порта
    try:
        async with websockets.serve(handle, HOST, PORT, max_size=8*2**20, ping_interval=300, ping_timeout=None):
            print(f"[WS] WebSocket server started successfully on {HOST}:{PORT}")
            await asyncio.Future()
    except OSError as e:
        if e.errno == 10048:  # WSAEADDRINUSE
            print(f"[ERROR] Port {PORT} is already in use. Trying alternative ports...")
            # Пробуем альтернативные порты
            for alt_port in range(PORT + 1, PORT + 10):
                try:
                    async with websockets.serve(handle, HOST, alt_port, max_size=8*2**20, ping_interval=300, ping_timeout=None):
                        print(f"[WS] WebSocket server started on alternative port {HOST}:{alt_port}")
                        print(f"[WS] Update your client to connect to port {alt_port}")
                        await asyncio.Future()
                        break
                except OSError:
                    continue
            else:
                print(f"[ERROR] Could not bind to any port in range {PORT}-{PORT+9}")
                raise e
        else:
            raise e

# CLI режим
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
            
            state = AgentState(text=TextMsg(user_input))
            result = await app.ainvoke(state)
            
            # Извлекаем текстовый ответ
            response_text = None
            for value in dict(result).values():
                if hasattr(value, 'text') and value.text:
                    response_text = value.text.text if hasattr(value.text, 'text') else value.text
                elif isinstance(value, str):
                    response_text = value
                elif isinstance(value, TextMsg):
                    response_text = value.text
                
                if response_text:
                    break
            
            print(f"Ассистент: {response_text or '[Нет ответа]'}")
                
        except (KeyboardInterrupt, EOFError):
            print("\n[CLI] Завершение работы.")
            break

if __name__ == "__main__":
    import sys
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        print("[INFO] Запуск в CLI режиме")
        try:
            asyncio.run(cli_loop())
        except KeyboardInterrupt:
            print("Interrupted.")
    else:
        print("[INFO] Запуск WebSocket сервера")
        try:
            asyncio.run(main_ws())
        except KeyboardInterrupt:
            print("Interrupted.")
