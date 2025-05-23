import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

# --- ENVIRONMENT & GLOBALS ---
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
# Оптимизированная модель для Orange Pi 5 Plus (самая маленькая)
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "smollm2:135m")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Оптимизация для Orange Pi 5 Plus
LOCAL_MAX_TOKENS = int(os.getenv("LOCAL_MAX_TOKENS", "64"))    # Уменьшен для скорости
LOCAL_CONTEXT    = int(os.getenv("LOCAL_CONTEXT",    "256"))   # Уменьшен контекст
LOCAL_THREADS    = int(os.getenv("LOCAL_THREADS",    "8"))      # Используем все 8 ядер
LOCAL_KEEP_ALIVE = int(os.getenv("LOCAL_KEEP_ALIVE", "60"))    # 1 минута в памяти
LOCAL_TOP_P      = float(os.getenv("LOCAL_TOP_P",   "0.9"))    # top_p
LOCAL_TOP_K      = int(os.getenv("LOCAL_TOP_K",       "20"))   # Уменьшен для скорости
# Настройки квантизации для Ollama
LOCAL_NUM_GPU    = int(os.getenv("LOCAL_NUM_GPU", "0"))           # Не используем GPU
LOCAL_LOW_VRAM   = os.getenv("LOCAL_LOW_VRAM", "true").lower() == "true"

SHOW_TEXT = os.getenv("SHOW_TEXT", "true").lower() == "true"

# --- PROVIDER INITIALIZATION ---
def _init_llm(provider: str, temperature: float) -> BaseChatModel:
    provider = provider.lower()
    model = LLM_MODEL or {
        "claude": CLAUDE_MODEL,
        "deepseek": DEEPSEEK_MODEL,
        "local": LOCAL_MODEL
    }.get(provider, LOCAL_MODEL)
    
    if provider == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temperature)
        except ImportError:
            raise ImportError("[ERROR] langchain-anthropic not installed. Run: pip install langchain-anthropic")
    elif provider == "deepseek":
        try:
            from langchain_deepseek import ChatDeepSeek
            return ChatDeepSeek(model=model, temperature=temperature)
        except ImportError:
            raise ImportError("[ERROR] langchain-deepseek not installed. Run: pip install langchain-deepseek")
    elif provider == "local":
        try:
            from langchain_ollama import ChatOllama
            # Дополнительные оптимизации для Orange Pi 5 Plus
            return ChatOllama(
                model=model, 
                temperature=temperature,
                num_predict=LOCAL_MAX_TOKENS,
                num_ctx=LOCAL_CONTEXT,
                num_thread=LOCAL_THREADS,
                keep_alive=f"{LOCAL_KEEP_ALIVE}s",
                top_p=LOCAL_TOP_P,
                top_k=LOCAL_TOP_K,
                num_gpu=LOCAL_NUM_GPU,
                low_vram=LOCAL_LOW_VRAM,
                mirostat=0,  # Отключаем для скорости
                repeat_penalty=1.0,  # Отключаем для скорости
                seed=42,  # Фиксированный seed для консистентности
            )
        except ImportError:
            raise ImportError("[ERROR] langchain-ollama not installed. Run: pip install langchain-ollama")
    else:
        print(f"[WARNING] Unknown provider '{provider}', using local Ollama")
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)

# --- LLM MANAGER ---
class LLMManager:
    """Класс для управления различными LLM провайдерами"""
    def __init__(self, provider: str = LLM_PROVIDER, temperature: float = LLM_TEMPERATURE):
        self.provider = provider.lower()
        self.temperature = temperature
        self.llm = _init_llm(self.provider, self.temperature)
        
        # Предзагрузка модели для Orange Pi
        if self.provider == "local":
            self._preload_model()
        
        if SHOW_TEXT:
            print(f"[INFO] Initialized LLM provider: {self.provider}")
            if self.provider == "local":
                model_name = LLM_MODEL or LOCAL_MODEL
                print(f"[INFO] Using model: {model_name}")
                print(f"[INFO] Optimization: context={LOCAL_CONTEXT}, max_tokens={LOCAL_MAX_TOKENS}, threads={LOCAL_THREADS}")

    def _preload_model(self):
        """Предзагружает модель в память для уменьшения первого отклика"""
        try:
            print("[INFO] Предзагрузка модели...")
            # Простой вызов для загрузки модели в память
            self.llm.invoke([HumanMessage(content="Привет")])
            print("[INFO] Модель предзагружена")
        except Exception as e:
            print(f"[WARNING] Не удалось предзагрузить модель: {e}")

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Генерирует ответ для заданного запроса.
        
        Args:
            prompt: Запрос пользователя
            system_prompt: Опциональный системный промпт
            tools: Опциональный список инструментов для LLM
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            print(f"[LOG] [LLM] Отправка запроса модели: {prompt[:50]}...")
            
            # Используем инструменты, если они предоставлены
            if tools:
                # Проверяем поддержку tools
                if not hasattr(self.llm, 'bind_tools'):
                    print("[WARNING] Модель не поддерживает bind_tools. Используйте langchain-ollama.")
                    # Fallback - обрабатываем без tools
                    response = await self.llm.ainvoke(messages)
                else:
                    model_with_tools = self.llm.bind_tools(tools)
                    response = await model_with_tools.ainvoke(messages)
            else:
                response = await self.llm.ainvoke(messages)
                
            print(f"[LOG] [LLM] Получен ответ модели: {str(response)[:100]}...")
            
            # Извлекаем текст ответа из различных форматов
            if hasattr(response, "content"):
                return response.content if isinstance(response.content, str) else str(response.content)
            return str(response)
        except Exception as e:
            print(f"[ERROR] LLM generation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Произошла ошибка при генерации ответа: {str(e)}"

    def get_provider_info(self) -> dict:
        if self.provider == "claude":
            model = LLM_MODEL or CLAUDE_MODEL
            return {"provider": "Anthropic Claude", "model": model}
        elif self.provider == "deepseek":
            model = LLM_MODEL or DEEPSEEK_MODEL
            return {"provider": "DeepSeek", "model": model}
        elif self.provider == "local":
            model = LLM_MODEL or LOCAL_MODEL
            return {"provider": "Local (Ollama)", "model": model}
        else:
            return {"provider": self.provider, "model": "unknown"} 