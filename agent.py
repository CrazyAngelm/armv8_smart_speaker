import asyncio, os, websockets
from dataclasses import dataclass
from typing import Any, Literal, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from llm_module import LLMManager
import time
import json

load_dotenv()

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
    return text

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
        "Ты — умный голосовой помощник для промышленной среды. "
        "Отвечай четко, по делу и профессионально. "
        "Старайся давать полезные ответы на запросы пользователя. "
        "Помогай с информацией о процессах, оборудовании или технических вопросах. "
        "Если не знаешь ответ, честно скажи об этом."
    )

# Инициализация менеджера LLM
llm_manager = LLMManager()
# Получаем информацию о текущем провайдере
llm_info = llm_manager.get_provider_info()
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
        # Используем LLMManager для генерации ответа
        reply = await llm_manager.generate_response(
            prompt=txt,
            system_prompt=current_system_prompt
        )
        t_llm1 = time.perf_counter()
        print(f"[PROFILE] LLMManager.generate_response: {t_llm1-t_llm0:.2f} сек")
    except Exception as e:
        print(f"[ERROR] LLM error: {e}")
        reply = "Извините, произошла ошибка. Можете повторить ваш вопрос?"
    t1 = time.perf_counter()
    print(f"[PROFILE] LLM node (всего): {t1-t0:.2f} сек")
    state.text = TextMsg(reply)
    return state

async def tts_node(state: AgentState) -> AgentState:
    if state.text:
        try:
            audio_bytes = await tts_client(state.text.text)
            state.audio = AudioMsg(audio_bytes, sr=48000)
        except Exception as e:
            print(f"[ERROR] TTS error: {e}")
    return state

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
workflow.add_node("tts", tts_node)

workflow.add_edge(START, "stt")      # начало → STT
workflow.add_conditional_edges(      # STT → (LLM или END)
    "stt",
    wake_router                      # callable‑router
)
workflow.add_edge("llm", "tts")      # LLM → TTS

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
                            
                            # Извлекаем аудио ответ
                            audio_result = None
                            
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
                                        
                                # Если не нашли аудио, но есть текст, синтезируем заново
                                if not audio_result:
                                    print(f"[LOG] [AGENT] Аудио не найдено, ищем текст для синтеза...")
                                    for key, value in values_dict.items():
                                        text_message = None
                                        # Проверяем, является ли значение TextMsg
                                        if hasattr(value, 'text') and value.text and hasattr(value.text, 'text'):
                                            print(f"[LOG] [AGENT] Найден TextMsg в ключе {key}")
                                            text_message = value.text.text
                                        # Проверяем, является ли значение строкой
                                        elif isinstance(value, str):
                                            print(f"[LOG] [AGENT] Найдена строка в ключе {key}")
                                            text_message = value
                                        # Проверяем, является ли само значение TextMsg
                                        elif isinstance(value, TextMsg):
                                            print(f"[LOG] [AGENT] Найден прямой TextMsg в ключе {key}")
                                            text_message = value.text
                                        
                                        if text_message:
                                            print(f"[LOG] [AGENT] Синтезируем речь из найденного текста: {text_message}")
                                            audio_bytes = await tts_client(text_message)
                                            audio_result = AudioMsg(audio_bytes, sr=48000)
                                            break
                                            
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
                                print("[LOG] [WS] Аудио-ответ не найден!")
                                await ws.send("ERROR: No audio result found")
                            
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
        ping_interval=30,  # seconds between pings
        ping_timeout=30    # seconds to wait for pong
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
            # Обрабатываем через llm_node
            new_state = await llm_node(state)
            if new_state.text:
                print(f"Ассистент: {new_state.text.text}")
        except (KeyboardInterrupt, EOFError):
            print("\n[CLI] Завершение работы.")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main_ws())
    except KeyboardInterrupt:
        print("Interrupted.")
