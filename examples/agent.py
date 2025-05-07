import asyncio, os, re, websockets
from dataclasses import dataclass
from typing import Any, Literal, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from dice_roller import dice_roller, DiceRollerTool
from llm_module import LLMManager
from memory import MemoryManager
from config import config
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
WHISPER_WS_HOST = os.getenv("WHISPER_WS_HOST", "localhost") 
WHISPER_WS_PORT = int(os.getenv("WHISPER_WS_PORT", 8779))
TTS_WS_HOST = os.getenv("TTS_WS_HOST", "localhost")
TTS_WS_PORT = int(os.getenv("TTS_WS_PORT", 8777))

# Глобальный замок для предотвращения перекрытия аудио
processing_lock = asyncio.Lock()

# ---------- STT клиент ----------
async def stt_whisper(audio: AudioMsg) -> str:
    t0 = time.perf_counter()
    print(f"[LOG] [STT] Отправка аудио ({len(audio.raw)} байт, sr={audio.sr}) в Whisper STT WS")
    try:
        async with websockets.connect(f"ws://{WHISPER_WS_HOST}:{WHISPER_WS_PORT}", max_size=2**22) as ws:
            await ws.send(audio.raw)
            resp = await ws.recv()
            if isinstance(resp, str) and not resp.startswith("ERROR"):
                t1 = time.perf_counter()
                print(f"[PROFILE] STT (Whisper): {t1-t0:.2f} сек")
                return resp
            raise RuntimeError(f"Whisper STT WS error: {resp}")
    except Exception as e:
        print(f"[ERROR] Whisper STT connection error: {e}")
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
async def tts_yandex(text: str) -> bytes:
    t0 = time.perf_counter()
    try:
        text = extract_tts_text(text)
    except Exception as e:
        print(f"[ERROR] [TTS] Ошибка при обработке текста: {e}")
    print(f"[LOG] [TTS] Отправка текста в TTS: {text}...")
    try:
        async with websockets.connect(f"ws://{TTS_WS_HOST}:{TTS_WS_PORT}", max_size=2**20) as ws:
            await ws.send(text)
            resp = await ws.recv()
            if isinstance(resp, bytes):
                t1 = time.perf_counter()
                print(f"[PROFILE] TTS (Yandex): {t1-t0:.2f} сек")
                return resp
            raise RuntimeError(f"TTS WS error: {resp}")
    except Exception as e:
        print(f"[ERROR] TTS connection error: {e}")
        raise

# ---------- Узлы графа ----------
_DICE_RE = re.compile(r"(\d*)d(\d+)([+-]\d+)?", re.I)
# Новый, более естественный и ролевой промпт с использованием переменной имени персонажа
def get_system_prompt():
    return (
        f"Ты — игрок в настольную ролевую игру (НРИ), отыгрываешь персонажа по имени {config.player_character_name}. "
        "Общайся естественно, дружелюбно, с юмором, как настоящий человек за столом. "
        "Не выдавай себя за ИИ. Не используй шаблонные фразы. "
        "Важно: НИКОГДА не придумывай факты о мире, локациях, персонажах или прошлых событиях. "
        "Строго полагайся только на факты в контексте или то, что непосредственно говорит собеседник. "
        "Если не знаешь чего-то о мире или истории персонажа, спрашивай, а не придумывай. "
        "Ты можешь выражать свое мнение и отношение к ситуации, но не создавай новые факты о мире. "
        "Бросай кубики только если тебя явно просят (например: 'сделай бросок на силу', 'брось 1d20', 'проверь ловкость', 'кинь кубик'). "
        "Если не понимаешь запрос, вежливо попроси уточнить. "
        "Если просят бросить кубик, ответь в формате '1d20=15' или аналогично. "
    )

# Строгое регулярное выражение для смены персонажа по команде с обращением к ИИ
_CHARACTER_CHANGE_STRICT = re.compile(r'(?:ИИ|AI|Sanya|Саня)[,\s]+(?:теперь\s+)?(?:игра(?:ешь|й)\s+за|ты\s+персонаж(?:а|ем)?|тво[ёе]\s+имя|зови\s+себя|тебя\s+зовут)\s+(?:персонажа\s+)?([А-Яа-яA-Za-z\-]{2,})', re.I)

def extract_character_name(text):
    match = _CHARACTER_CHANGE_STRICT.search(text)
    if match:
        return match.group(1).strip()
    return None

# Инициализация менеджера LLM
llm_manager = LLMManager()
# Инициализация менеджера памяти
memory_manager = MemoryManager()
# Получаем информацию о текущем провайдере
llm_info = llm_manager.get_provider_info()
print(f"[INFO] Using LLM: {llm_info['provider']} ({llm_info['model']})")

async def stt_node(state: AgentState) -> AgentState:
    if state.audio:
        print(f"[LOG] [STT_NODE] Получено аудио: {len(state.audio.raw)} байт")
        try:
            recognized_text = await stt_whisper(state.audio)
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

    # Проверяем, запрашивает ли пользователь смену персонажа
    character_change_match = extract_character_name(txt)
    if character_change_match:
        print(f"[LOG] [CHARACTER] Обнаружен запрос на смену персонажа на: '{character_change_match}'")
        
        # Пытаемся обновить имя персонажа
        success, message = config.update_player_character_name(character_change_match)
        if success:
            print(f"[LOG] [CHARACTER] {message}")
            state.text = TextMsg(f"Хорошо, теперь я играю за персонажа по имени {config.player_character_name}!")
            t1 = time.perf_counter()
            print(f"[PROFILE] LLM node (смена персонажа): {t1-t0:.2f} сек")
            return state
        else:
            print(f"[WARNING] [CHARACTER] {message}")
            # Если не удалось обновить, продолжаем обработку как обычно

    # Обрабатываем новую информацию из текста (извлекаем и сохраняем факты, персонажей, изменения)
    t_mem0 = time.perf_counter()
    learning_results = await memory_manager.process_new_information(txt)
    t_mem1 = time.perf_counter()
    print(f"[PROFILE] MemoryManager.process_new_information: {t_mem1-t_mem0:.2f} сек")
    
    if learning_results["stat_changes"]:
        print(f"[LOG] [LEARNING] Обнаружены и обработаны изменения характеристик персонажей")
    
    if learning_results["world_facts"]:
        print(f"[LOG] [LEARNING] Извлечены факты о мире: {len(learning_results['world_facts'])}")
        
    if learning_results["character_traits"]:
        print(f"[LOG] [LEARNING] Извлечены черты персонажей: {len(learning_results['character_traits'])}")

    # Добавляем реплику в кратковременную память
    memory_manager.add_short_term(txt)
    short_term_context = memory_manager.get_short_term_context()

    # Поиск упоминаний персонажей в тексте и загрузка их данных
    t_mem2 = time.perf_counter()
    characters_data = await memory_manager.detect_and_load_characters(txt)
    t_mem3 = time.perf_counter()
    print(f"[PROFILE] MemoryManager.detect_and_load_characters: {t_mem3-t_mem2:.2f} сек")
    character_context = ""
    character_memories = []
    
    if characters_data:
        # Форматируем данные о персонажах для контекста
        t_mem4 = time.perf_counter()
        character_context = await memory_manager.prepare_character_context(characters_data)
        t_mem5 = time.perf_counter()
        print(f"[PROFILE] MemoryManager.prepare_character_context: {t_mem5-t_mem4:.2f} сек")
        print(f"[LOG] [MEMORY] Найдены упоминания персонажей: {', '.join(characters_data.keys())}")
        
        # Ищем дополнительные воспоминания о персонажах из векторной БД
        for character_name in characters_data.keys():
            t_mem6 = time.perf_counter()
            memories = await memory_manager.search_character_related_memories(character_name, top_k=2)
            t_mem7 = time.perf_counter()
            print(f"[PROFILE] MemoryManager.search_character_related_memories('{character_name}'): {t_mem7-t_mem6:.2f} сек")
            if memories:
                print(f"[LOG] [MEMORY] Найдены воспоминания о персонаже {character_name}")
                character_memories.extend(memories)

    # Поиск релевантных фактов в памяти
    t_mem8 = time.perf_counter()
    general_facts = memory_manager.search_general(txt, top_k=3)
    persona_facts = memory_manager.search_persona(txt, top_k=2)
    t_mem9 = time.perf_counter()
    print(f"[PROFILE] MemoryManager.search_general+persona: {t_mem9-t_mem8:.2f} сек")

    # Формируем дополнительный контекст для LLM
    memory_context = ""
    
    # Добавляем информацию о персонажах (если есть)
    if character_context:
        memory_context += character_context
    
    # Добавляем связанные воспоминания о персонажах
    if character_memories:
        memory_context += "\n[Связанные воспоминания о персонажах]:\n"
        memory_context += "\n".join(f"- {memory}" for memory in character_memories[:4])
    
    # Добавляем общие факты/правила
    if general_facts:
        memory_context += "\n[Факты мира/правила]:\n" + "\n".join(f"- {fact}" for fact in general_facts)
    
    # Добавляем информацию о персонаже игрока
    player_name = config.player_character_name
    if persona_facts:
        memory_context += f"\n[Черты/поведение персонажа {player_name}]:\n" + "\n".join(f"- {fact}" for fact in persona_facts)
    
    # Добавляем историю диалога
    if short_term_context:
        memory_context += "\n[Кратковременная память (история диалога)]:\n" + short_term_context

    # Расширенные ключевые слова для определения запросов на бросок кубика
    dice_keywords = [
        "бросок", "брось", "кинь", "проверь", "проверка", "бросить", "кинуть", 
        "выброси", "roll", "throw", "check", "test", "d20", "d6", "d4", "d8", "d10", "d12"
    ]
    
    # Обнаружение запроса на бросок кубика
    has_dice_request = any(kw in txt.lower() for kw in dice_keywords)
    dice_matches = list(_DICE_RE.finditer(txt))
    
    # Обрабатываем все возможные броски кубиков в запросе
    dice_results = {}
    if has_dice_request and dice_matches:
        for match in dice_matches:
            dice_expr = match.group(0)
            result = dice_roller(dice_expr)
            dice_results[dice_expr] = result
    
    # Если найдены броски кубиков, добавляем контекст к запросу для LLM
    if dice_results:
        dice_context = "Результаты бросков кубиков:\n"
        for expr, result in dice_results.items():
            dice_context += f"- {result}\n"
        
        # Подготавливаем запрос с контекстом бросков для LLM
        enhanced_prompt = f"""
{txt}
{memory_context}

{dice_context}

Включи результаты этих бросков и факты из памяти в свой ответ органично, как бы ты сам их бросал и вспоминал.
ВАЖНО: Не придумывай новых фактов о мире или событиях. Используй ТОЛЬКО те факты, которые указаны выше.
"""
        prompt_to_use = enhanced_prompt
    else:
        # Если нет бросков, но есть память — тоже подмешиваем
        if memory_context:
            enhanced_prompt = f"""
{txt}
{memory_context}

Используй факты из памяти и контекст для более точного ответа. Если речь идет о персонажах - используй информацию об их характеристиках.
ВАЖНО: Не придумывай новых фактов о мире или событиях. Используй ТОЛЬКО те факты, которые указаны выше.
Если ты не знаешь чего-то о мире или персонажах, лучше спроси собеседника, чем придумывать.
"""
            prompt_to_use = enhanced_prompt
        else:
            # Если нет ни бросков, ни памяти, но добавляем напоминание о непридумывании фактов
            prompt_to_use = f"""{txt}

ВАЖНО: Не придумывай фактов о мире или событиях. Если ты не знаешь чего-то о мире или персонажах, лучше спроси собеседника."""
    
    try:
        # Получаем актуальный системный промпт с учетом текущего имени персонажа
        current_system_prompt = get_system_prompt()
        t_llm0 = time.perf_counter()
        # Используем наш новый LLMManager для генерации ответа
        reply = await llm_manager.generate_response(
            prompt=prompt_to_use,
            system_prompt=current_system_prompt
        )
        t_llm1 = time.perf_counter()
        print(f"[PROFILE] LLMManager.generate_response: {t_llm1-t_llm0:.2f} сек")
        # Если ответ содержит результат tool call, выводим его отдельно
        if reply and reply.startswith("[TOOL RESULT]:"):
            # Можно парсить и красиво выводить результат tool, если нужно
            print(reply)
            # Оставим только текст ответа LLM после tool call
            reply = reply.split("\n", 1)[-1]
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
            audio_bytes = await tts_yandex(state.text.text)
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
                                            audio_bytes = await tts_yandex(text_message)
                                            audio_result = AudioMsg(audio_bytes, sr=48000)
                                            break
                                            
                            # Если аудио найдено, отправляем его
                            if audio_result:
                                print(f"[LOG] [WS] Отправка аудио-ответа: {len(audio_result.raw)} байт")
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
        handle, HOST, PORT, max_size=2**20,
        ping_interval=30,  # seconds between pings
        ping_timeout=30    # seconds to wait for pong
    ):
        await asyncio.Future()

async def cli_loop():
    print("\n[CLI] Sanya 2.0 — текстовый режим. Введите 'exit' для выхода.\n")
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
                # Если ответ содержит результат tool call, выводим его отдельно
                if new_state.text.text.startswith("[TOOL RESULT]:"):
                    print(new_state.text.text)
                    print("Саня:", new_state.text.text.split("\n", 1)[-1])
                else:
                    print(f"Саня: {new_state.text.text}")
        except (KeyboardInterrupt, EOFError):
            print("\n[CLI] Завершение работы.")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main_ws())
    except KeyboardInterrupt:
        print("Interrupted.")
