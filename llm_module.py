import os
import json
import asyncio
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

# --- ENVIRONMENT & GLOBALS ---
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma3:12b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

LOCAL_MAX_TOKENS = int(os.getenv("LOCAL_MAX_TOKENS", "128"))    # num_predict
LOCAL_CONTEXT    = int(os.getenv("LOCAL_CONTEXT",    "512"))   # num_ctx
LOCAL_THREADS    = int(os.getenv("LOCAL_THREADS",    "0"))      # num_thread (0 = autodetect)
LOCAL_KEEP_ALIVE = int(os.getenv("LOCAL_KEEP_ALIVE", "-1"))    # keep_alive
LOCAL_TOP_P      = float(os.getenv("LOCAL_TOP_P",   "0.9"))    # top_p
LOCAL_TOP_K      = int(os.getenv("LOCAL_TOP_K",       "40"))   # top_k

SHOW_TEXT = os.getenv("SHOW_TEXT", "true").lower() == "true"

# --- PROVIDER INITIALIZATION ---
def _init_llm(provider: str, temperature: float) -> BaseChatModel:
    provider = provider.lower()
    model = LLM_MODEL or {
        "claude": CLAUDE_MODEL,
        "local": LOCAL_MODEL
    }.get(provider, CLAUDE_MODEL)
    if provider == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temperature)
        except ImportError:
            raise ImportError("[ERROR] langchain-anthropic not installed. Run: pip install langchain-anthropic")
    elif provider == "local":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model, 
                temperature=temperature,
                num_predict=LOCAL_MAX_TOKENS,
                num_ctx=LOCAL_CONTEXT,
                num_thread=LOCAL_THREADS,
                keep_alive=LOCAL_KEEP_ALIVE,
                top_p=LOCAL_TOP_P,
                top_k=LOCAL_TOP_K,
            )
        except ImportError:
            raise ImportError("[ERROR] langchain-ollama not installed. Run: pip install langchain-ollama")
    else:
        print(f"[WARNING] Unknown provider '{provider}', using ChatAnthropic")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)

# --- LLM MANAGER ---
class LLMManager:
    """Класс для управления различными LLM провайдерами"""
    def __init__(self, provider: str = LLM_PROVIDER, temperature: float = LLM_TEMPERATURE):
        self.provider = provider.lower()
        self.temperature = temperature
        self.llm = _init_llm(self.provider, self.temperature)
        if SHOW_TEXT:
            print(f"[INFO] Initialized LLM provider: {self.provider}")

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
        elif self.provider == "local":
            model = LLM_MODEL or LOCAL_MODEL
            return {"provider": "Local (Ollama)", "model": model}
        else:
            return {"provider": self.provider, "model": "unknown"} 