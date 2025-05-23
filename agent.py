# hybrid_agent.py
import asyncio, os, websockets
from dataclasses import dataclass
from typing import Any, Literal, Optional, Dict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from llm_module import LLMManager
import time
import json
from mqtt_tools import tools, execute_tool, init_mqtt
import re
import hashlib

# Импортируем оптимизированную систему парсинга
from improved_tool_parser import OptimizedToolParser, ToolCall

load_dotenv()
init_mqtt()

# Создаем глобальный экземпляр парсера
tool_parser = OptimizedToolParser()

# Настройки производительности
PERFORMANCE_MODE = os.getenv("PERFORMANCE_MODE", "balanced").lower()  # fast, balanced, accurate
USE_LLM_FALLBACK = os.getenv("USE_LLM_FALLBACK", "true").lower() == "true"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))

# Устанавливаем порог уверенности
tool_parser.set_confidence_threshold(CONFIDENCE_THRESHOLD)

class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
        self.enabled = os.getenv("PERF_MONITOR", "true").lower() == "true"
        self.stats = {"total_requests": 0, "tool_calls": 0, "llm_calls": 0, "direct_parse": 0}
    
    def start(self, phase: str):
        if self.enabled:
            self.timings[f"{phase}_start"] = time.perf_counter()
    
    def end(self, phase: str):
        if self.enabled and f"{phase}_start" in self.timings:
            duration = time.perf_counter() - self.timings[f"{phase}_start"]
            print(f"[PERF] {phase}: {duration:.2f}s")
            return duration
    
    def log_stat(self, stat_name: str):
        if stat_name in self.stats:
            self.stats[stat_name] += 1
    
    def get_stats(self):
        return self.stats.copy()

perf = PerformanceMonitor()

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
    # Новые поля для отслеживания
    parse_method: Optional[str] = None  # direct, llm_assisted, llm_only
    confidence: Optional[float] = None

# WebSocket настройки
STT_WS_HOST = os.getenv("STT_WS_HOST", "localhost") 
STT_WS_PORT = int(os.getenv("STT_WS_PORT", 8778))
TTS_WS_HOST = os.getenv("TTS_WS_HOST", "localhost")
TTS_WS_PORT = int(os.getenv("TTS_WS_PORT", 8777))

processing_lock = asyncio.Lock()

# Кэш для LLM
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
    if len(llm_cache) > 50:  # Уменьшили размер кэша для экономии памяти
        oldest_key = next(iter(llm_cache))
        del llm_cache[oldest_key]

# STT и TTS клиенты (без изменений)
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

def extract_tts_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # Удаляем все блоки <think>...</think> и берем только то, что после последнего </think>
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
    if re.search(think_pattern, text):
        # Берем только текст после последнего </think>
        parts = re.split(r'</think>', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            return parts[-1].strip()
        else:
            # Если почему-то нет закрывающего тега, просто удаляем <think>...</think>
            return re.sub(think_pattern, '', text).strip()
    # Старое поведение для json-ответов
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

# Гибридная функция для LLM-помощи в парсинге
async def llm_assisted_parse(text: str) -> Optional[List[ToolCall]]:
    """Использует LLM для помощи в парсинге неоднозначных команд"""
    system_prompt = tool_parser.get_simple_system_prompt()
    
    # Проверяем кэш
    cached = get_cached_response(text, system_prompt)
    if cached:
        return _parse_llm_response(cached, text)
    
    try:
        print(f"[DEBUG] LLM-помощь для парсинга: '{text}'")
        perf.log_stat("llm_calls")
        
        result = await llm_manager.llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ])
        
        content = result.content if hasattr(result, 'content') else str(result)
        content = content.strip().upper()
        
        print(f"[DEBUG] LLM ответ: {content}")
        cache_response(text, system_prompt, content)
        
        return _parse_llm_response(content, text)
        
    except Exception as e:
        print(f"[ERROR] LLM-помощь failed: {e}")
        return None

def _parse_llm_response(llm_response: str, original_text: str) -> Optional[List[ToolCall]]:
    """Парсит ответ LLM и создает ToolCall"""
    llm_response = llm_response.strip().upper()
    
    tool_mapping = {
        "ВРЕМЯ": "get_time",
        "ПОГОДА": "get_weather", 
        "ТАЙМЕР": "set_timer",
        "НАПОМИНАНИЕ": "set_notification",
        "ЗВОНОК": "call_contact"
    }
    
    if llm_response in tool_mapping:
        tool_name = tool_mapping[llm_response]
        args = tool_parser._extract_args(tool_name, original_text)
        return [ToolCall(name=tool_name, args=args, confidence=0.7)]
    
    return None

# Предзагрузка моделей
async def preload_models():
    print("[INFO] Предзагрузка моделей...")
    
    try:
        test_audio = AudioMsg(b'\x00' * 1600, sr=16000)
        await stt_vosk(test_audio)
        print("[INFO] STT готов")
    except:
        print("[WARNING] STT недоступен")
    
    try:
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

async def intelligent_parsing_node(state: AgentState) -> AgentState:
    """Умный узел парсинга с гибридным подходом"""
    perf.start("parsing")
    perf.log_stat("total_requests")
    
    if not state.text:
        perf.end("parsing")
        return state
    
    txt = state.text.text
    print(f"[DEBUG] Интеллектуальный парсинг: '{txt}'")
    
    # 1. Пробуем прямой парсинг
    direct_result = tool_parser.parse_text_for_tools(txt, use_llm_fallback=False)
    
    if direct_result and direct_result[0].confidence >= CONFIDENCE_THRESHOLD:
        print(f"[DEBUG] Прямой парсинг успешен: {direct_result[0].name} (conf: {direct_result[0].confidence:.2f})")
        state.tool_calls = [_convert_to_tool_call_dict(tc) for tc in direct_result]
        state.parse_method = "direct"
        state.confidence = direct_result[0].confidence
        perf.log_stat("direct_parse")
        perf.end("parsing")
        return state
    
    # 2. Если прямой парсинг неуспешен и разрешен LLM fallback
    if USE_LLM_FALLBACK and PERFORMANCE_MODE != "fast":
        llm_result = await llm_assisted_parse(txt)
        
        if llm_result and llm_result[0].confidence >= CONFIDENCE_THRESHOLD:
            print(f"[DEBUG] LLM-помощь успешна: {llm_result[0].name} (conf: {llm_result[0].confidence:.2f})")
            state.tool_calls = [_convert_to_tool_call_dict(tc) for tc in llm_result]
            state.parse_method = "llm_assisted"
            state.confidence = llm_result[0].confidence
            perf.end("parsing")
            return state
    
    # 3. Если ничего не сработало, используем обычный LLM для генерации ответа
    if PERFORMANCE_MODE == "accurate":
        state.parse_method = "llm_only"
        # Переходим к обычной генерации LLM
    
    perf.end("parsing")
    return state

def _convert_to_tool_call_dict(tc: ToolCall) -> Dict[str, Any]:
    """Конвертирует ToolCall в формат для tools_node"""
    return {
        "name": tc.name,
        "args": tc.args,
        "id": f"tool_{tc.name}_{int(time.time())}"
    }

async def llm_node(state: AgentState) -> AgentState:
    """Упрощенный LLM узел для случаев когда парсинг не сработал"""
    perf.start("llm")
    
    if not state.text:  # Убираем проверку на tool_calls
        perf.end("llm")
        return state
    
    txt = state.text.text
    
    # Простой системный промпт для разговора
    system_prompt = """Ты дружелюбный голосовой помощник. 
Отвечай кратко и естественно на русском языке.
Если не понимаешь команду, честно скажи об этом и предложи помощь."""
    
    # Проверяем кэш
    cached = get_cached_response(txt, system_prompt)
    if cached:
        state.text = TextMsg(cached)
        perf.end("llm")
        return state
    
    try:
        print(f"[DEBUG] LLM генерация ответа для: '{txt}'")
        perf.log_stat("llm_calls")
        
        result = await llm_manager.llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": txt}
        ])
        
        content = result.content if hasattr(result, 'content') else str(result)
        cache_response(txt, system_prompt, content)
        state.text = TextMsg(content)
        
    except Exception as e:
        print(f"[ERROR] LLM error: {e}")
        state.text = TextMsg("Извините, произошла ошибка.")
    
    perf.end("llm")
    return state

# Остальные узлы (без изменений)
async def tools_node(state: AgentState) -> AgentState:
    if not state.tool_calls:
        return state
    
    perf.start("tools")
    perf.log_stat("tool_calls")
    print(f"[LOG] [TOOLS] Выполнение {len(state.tool_calls)} инструментов")
    
    async def execute_tool_async(tool_call):
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", f"tool_{tool_name}_{int(time.time())}")
        else:
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
def parsing_router(state: AgentState) -> Literal["tools", "llm", "tts"]:
    """Маршрутизатор после парсинга"""
    if state.tool_calls:
        return "tools"
    elif state.text:
        # Если есть текст но нет инструментов, отправляем к LLM для генерации ответа
        return "llm"
    return "tts"

def tools_router(state: AgentState) -> Literal["tool_results_processor", "tts"]:
    if state.tool_results:
        return "tool_results_processor"
    return "tts"

# Построение графа
workflow = StateGraph(AgentState)
workflow.add_node("stt", stt_node)
workflow.add_node("intelligent_parsing", intelligent_parsing_node)
workflow.add_node("llm", llm_node)
workflow.add_node("tools", tools_node)
workflow.add_node("tool_results_processor", tool_results_processor)
workflow.add_node("tts", tts_node)

workflow.add_edge(START, "stt")
workflow.add_edge("stt", "intelligent_parsing")
workflow.add_conditional_edges("intelligent_parsing", parsing_router, 
                               {"tools": "tools", "llm": "llm", "tts": "tts"})
workflow.add_edge("llm", "tts")
workflow.add_conditional_edges("tools", tools_router, 
                               {"tool_results_processor": "tool_results_processor", "tts": "tts"})
workflow.add_edge("tool_results_processor", "tts")
workflow.add_edge("tts", END)

app = workflow.compile()

# WebSocket сервер и остальной код остается без изменений...
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
                        
                        # Логируем статистику
                        if hasattr(result.get('intelligent_parsing'), 'parse_method'):
                            method = result['intelligent_parsing'].parse_method
                            confidence = result['intelligent_parsing'].confidence
                            print(f"[STATS] Метод: {method}, Уверенность: {confidence:.2f}")
                        
                        audio_result = None
                        for value in dict(result).values():
                            if hasattr(value, 'audio') and value.audio:
                                audio_result = value.audio
                                break
                        
                        if not audio_result:
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
    print(f"[CONFIG] Performance mode: {PERFORMANCE_MODE}")
    print(f"[CONFIG] LLM fallback: {USE_LLM_FALLBACK}")
    print(f"[CONFIG] Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    try:
        async with websockets.serve(handle, HOST, PORT, max_size=8*2**20, ping_interval=300, ping_timeout=None):
            print(f"[WS] WebSocket server started successfully on {HOST}:{PORT}")
            await asyncio.Future()
    except OSError as e:
        if e.errno == 10048:
            print(f"[ERROR] Port {PORT} is already in use. Trying alternative ports...")
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

async def cli_loop():
    print("\n[CLI] Умный голосовой помощник — текстовый режим. Введите 'exit' для выхода.")
    print(f"[CLI] Режим производительности: {PERFORMANCE_MODE}")
    print(f"[CLI] Введите 'stats' для просмотра статистики\n")
    
    while True:
        try:
            user_input = input("Вы: ").strip()
            if user_input.lower() in ("exit", "quit", "выход"):
                print("[CLI] Завершение работы.")
                break
            if user_input.lower() == "stats":
                stats = perf.get_stats()
                print(f"[STATS] {stats}")
                continue
            if not user_input:
                continue
            
            state = AgentState(text=TextMsg(user_input))
            result = await app.ainvoke(state)
            
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