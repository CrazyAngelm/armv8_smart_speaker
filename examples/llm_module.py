import os
import json
import asyncio
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import Tool
from memory import MemoryManager
from tools import (
    create_character, update_character, get_character, delete_character,
    CreateCharacterSchema, UpdateCharacterSchema, GetCharacterSchema, DeleteCharacterSchema,
    update_elemental_affinity, add_skill, add_trait, add_knowledge, add_inventory_item,
    UpdateElementalAffinitySchema, AddSkillSchema, AddTraitSchema, AddKnowledgeSchema, AddInventoryItemSchema
)

# --- ENVIRONMENT & GLOBALS ---
load_dotenv()
memory_manager = MemoryManager()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma3:12b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# --- TOOLS DEFINITION ---
tools_list = [
    Tool.from_function(
        name="create_character",
        description="Создать нового персонажа с заданными атрибутами (имя, сила, ловкость и т.д.)",
        args_schema=CreateCharacterSchema,
        func=create_character,
        coroutine=create_character
    ),
    Tool.from_function(
        name="update_character",
        description="Изменить атрибут персонажа (например, силу, ловкость и т.д.)",
        args_schema=UpdateCharacterSchema,
        func=update_character,
        coroutine=update_character
    ),
    Tool.from_function(
        name="get_character",
        description="Получить информацию о персонаже по имени",
        args_schema=GetCharacterSchema,
        func=get_character,
        coroutine=get_character
    ),
    Tool.from_function(
        name="delete_character",
        description="Удалить персонажа по имени",
        args_schema=DeleteCharacterSchema,
        func=delete_character,
        coroutine=delete_character
    ),
    Tool.from_function(
        name="update_elemental_affinity",
        description="Обновить элементальное сродство персонажа (огонь, вода, земля, воздух, свет, тьма)",
        args_schema=UpdateElementalAffinitySchema,
        func=update_elemental_affinity,
        coroutine=update_elemental_affinity
    ),
    Tool.from_function(
        name="add_skill",
        description="Добавить или обновить навык персонажа",
        args_schema=AddSkillSchema,
        func=add_skill,
        coroutine=add_skill
    ),
    Tool.from_function(
        name="add_trait",
        description="Добавить черту/особенность персонажу",
        args_schema=AddTraitSchema,
        func=add_trait,
        coroutine=add_trait
    ),
    Tool.from_function(
        name="add_knowledge",
        description="Добавить знание/язык персонажу",
        args_schema=AddKnowledgeSchema,
        func=add_knowledge,
        coroutine=add_knowledge
    ),
    Tool.from_function(
        name="add_inventory_item",
        description="Добавить предмет в инвентарь персонажа",
        args_schema=AddInventoryItemSchema,
        func=add_inventory_item,
        coroutine=add_inventory_item
    ),
]

# --- PROVIDER INITIALIZATION ---
def _init_llm(provider: str, temperature: float) -> BaseChatModel:
    provider = provider.lower()
    model = LLM_MODEL or {
        "deepseek": DEEPSEEK_MODEL,
        "claude": CLAUDE_MODEL,
        "local": LOCAL_MODEL
    }.get(provider, DEEPSEEK_MODEL)
    if provider == "deepseek":
        try:
            from langchain_deepseek import ChatDeepSeek
            return ChatDeepSeek(model=model, temperature=temperature)
        except ImportError:
            raise ImportError("[ERROR] langchain-deepseek not installed. Run: pip install langchain-deepseek")
    elif provider == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temperature)
        except ImportError:
            raise ImportError("[ERROR] langchain-anthropic not installed. Run: pip install langchain-anthropic")
    elif provider == "local":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, temperature=temperature)
        except ImportError:
            raise ImportError("[ERROR] langchain-ollama not installed. Run: pip install langchain-ollama")
    else:
        print(f"[WARNING] Unknown provider '{provider}', using DeepSeek")
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(model=DEEPSEEK_MODEL, temperature=temperature)

# --- TOOL EXECUTION HELPERS ---
def _tool_function_map():
    return {
        "create_character": create_character,
        "get_character": get_character,
        "update_character": update_character,
        "delete_character": delete_character,
        "update_elemental_affinity": update_elemental_affinity,
        "add_skill": add_skill,
        "add_trait": add_trait,
        "add_knowledge": add_knowledge,
        "add_inventory_item": add_inventory_item
    }

async def _execute_tool(tool_name, tool_args):
    tool_functions = _tool_function_map()
    if tool_name in tool_functions:
        func = tool_functions[tool_name]
        print(f"[LOG] [TOOL] Выполнение {tool_name} с аргументами: {tool_args}")
        return await func(**tool_args)
    print(f"[WARNING] [TOOL] Неизвестный инструмент: {tool_name}")
    return None

# --- LLM MANAGER ---
class LLMManager:
    """Класс для управления различными LLM провайдерами с поддержкой function calling"""
    def __init__(self, provider: str = LLM_PROVIDER, temperature: float = LLM_TEMPERATURE):
        self.provider = provider.lower()
        self.temperature = temperature
        self.llm = _init_llm(self.provider, self.temperature)
        print(f"[INFO] Initialized LLM provider: {self.provider}")
        self.tools = tools_list
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Генерирует ответ для заданного запроса с поддержкой function calling.
        Если был tool call, интегрирует результат tool в человеко-понятный ответ и обновляет память.
        LLM ВСЕГДА отвечает: результат tool + обычный ответ LLM.
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        try:
            print(f"[LOG] [LLM] Отправка запроса модели: {prompt[:50]}...")
            response = await self.llm_with_tools.ainvoke(messages)
            print(f"[LOG] [LLM] Получен ответ модели: {str(response)[:100]}...")
            if hasattr(response, "tool_calls") and response.tool_calls:
                return await self._handle_tool_calls(response)
            # Если не было tool call, просто возвращаем ответ LLM
            if hasattr(response, "content"):
                return response.content if isinstance(response.content, str) else str(response.content)
            return str(response)
        except Exception as e:
            print(f"[ERROR] LLM generation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Произошла ошибка при генерации ответа: {str(e)}"

    async def _handle_tool_calls(self, response) -> str:
        """Обработка tool call-ов и формирование финального ответа"""
        tool_outputs = []
        user_friendly = []
        for call in response.tool_calls:
            tool_name, tool_args = self._extract_tool_info(call, response)
            print(f"[LOG] [TOOL] Вызов: {tool_name}, аргументы: {tool_args}")
            tool_result = getattr(call, "function", None)
            if tool_result and hasattr(tool_result, "output"):
                tool_result = tool_result.output
            elif tool_name and tool_args:
                tool_result = await _execute_tool(tool_name, tool_args)
            user_friendly.extend(self._format_tool_result(tool_name, tool_args, tool_result))
        llm_text = self._extract_llm_text(response)
        final_response = "\n".join(user_friendly)
        if llm_text and not any(uf in llm_text for uf in user_friendly):
            if final_response:
                final_response += "\n\n"
            final_response += llm_text
        if not final_response:
            final_response = "Действие выполнено. Чем ещё могу помочь?"
        print(f"[LOG] [LLM] Сформирован финальный ответ с учётом инструментов: {final_response[:100]}...")
        return final_response

    def _extract_tool_info(self, call, response):
        tool_name = getattr(call, "name", None)
        tool_args = getattr(call, "args", None)
        if not tool_name and hasattr(call, "function") and hasattr(call.function, "name"):
            tool_name = call.function.name
        if not tool_args and hasattr(call, "arguments"):
            tool_args = call.arguments
        if not tool_args and hasattr(call, "function") and hasattr(call.function, "arguments"):
            tool_args = call.function.arguments
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except:
                    print(f"[ERROR] [LLM] Не удалось преобразовать строковые аргументы в JSON: {tool_args}")
        if isinstance(call, dict):
            tool_name = call.get('name', tool_name)
            tool_args = call.get('arguments', tool_args)
        if not tool_name and not tool_args and hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'tool_use':
                        tool_name = item.get('name')
                        tool_args = item.get('input', {})
        return tool_name, tool_args or {}

    def _format_tool_result(self, tool_name, tool_args, tool_result):
        """Формирует человеко-понятный ответ по результату tool"""
        out = []
        if tool_name == "create_character":
            name = tool_args.get("name")
            if name:
                if tool_result and isinstance(tool_result, dict) and tool_result.get("status") == "ok":
                    out.append(f"Персонаж {name} успешно создан! ID: {tool_result.get('character_id')}")
                else:
                    out.append(f"Персонаж {name} успешно создан!")
                asyncio.create_task(memory_manager.load_all_character_names())
        elif tool_name == "delete_character":
            name = tool_args.get("name")
            if name:
                out.append(f"Персонаж {name} удалён из игры.")
                asyncio.create_task(memory_manager.load_all_character_names())
        elif tool_name == "update_character":
            name = tool_args.get("name")
            attr = tool_args.get("attribute")
            value = tool_args.get("value")
            if name and attr:
                out.append(f"У персонажа {name} обновлён атрибут {attr} на {value}.")
        elif tool_name == "add_skill":
            name = tool_args.get("name")
            skill = tool_args.get("skill_name")
            level = tool_args.get("level", 1)
            if name and skill:
                out.append(f"Персонажу {name} добавлен навык: {skill} (уровень {level}).")
        elif tool_name == "add_trait":
            name = tool_args.get("name")
            trait = tool_args.get("trait_name")
            if name and trait:
                out.append(f"Персонажу {name} добавлена черта: {trait}.")
        elif tool_name == "add_knowledge":
            name = tool_args.get("name")
            knowledge = tool_args.get("knowledge_name")
            if name and knowledge:
                out.append(f"Персонажу {name} добавлено знание: {knowledge}.")
        elif tool_name == "add_inventory_item":
            name = tool_args.get("name")
            item = tool_args.get("item_name")
            if name and item:
                out.append(f"Персонажу {name} добавлен предмет: {item}.")
        elif tool_name == "get_character":
            name = tool_args.get("name")
            if name and tool_result and isinstance(tool_result, dict) and tool_result.get("status") == "ok":
                info = tool_result.get("info", {})
                char_desc = [f"Вот информация о персонаже {name}:"]
                basic = info.get("basic", {})
                if basic:
                    race = basic.get("race_name", "")
                    sex = basic.get("sex", "")
                    char_desc.append(f"Раса: {race}, Пол: {sex}, Мировоззрение: {basic.get('alignment', '')}")
                attrs = info.get("attributes", {})
                if attrs:
                    attr_str = f"Атрибуты: СИЛ {attrs.get('str', 0)}, ЛОВ {attrs.get('dex', 0)}, "
                    attr_str += f"ВЫН {attrs.get('con', 0)}, ВСП {attrs.get('per', 0)}, "
                    attr_str += f"ИНТ {attrs.get('int', 0)}, ВОЛ {attrs.get('wil', 0)}, ХАР {attrs.get('cha', 0)}"
                    char_desc.append(attr_str)
                sec_attrs = info.get("secondary_attributes", {})
                if sec_attrs:
                    hp = f"{sec_attrs.get('hp_current', 0)}/{sec_attrs.get('hp_max', 0)}"
                    mana = f"{sec_attrs.get('mana_current', 0)}/{sec_attrs.get('mana_max', 0)}"
                    char_desc.append(f"Здоровье: {hp}, Мана: {mana}")
                skills = info.get("skills", [])
                if skills:
                    skill_list = [f"{s.get('name_ru')} ({s.get('level')})" for s in skills[:5]]
                    char_desc.append(f"Навыки: {', '.join(skill_list)}")
                out.append("\n".join(char_desc))
            # Если имя не указано или отсутствует - значит запрос на всех персонажей
            elif not name and tool_result and isinstance(tool_result, dict) and tool_result.get("status") == "ok":
                # Обработка случая, когда запрошены все персонажи
                characters = tool_result.get("characters", [])
                if characters:
                    char_desc = [f"Вот список всех персонажей ({len(characters)}):\n"]
                    for char_data in characters:
                        char_name = char_data.get("name", "")
                        info = char_data.get("info", {})
                        basic = info.get("basic", {})
                        race = basic.get("race_name", "") if basic else ""
                        alignment = basic.get("alignment", "") if basic else ""
                        
                        # Основное описание персонажа
                        char_line = f"- {char_name}"
                        if race:
                            char_line += f", {race}"
                        if alignment:
                            char_line += f", {alignment}"
                        
                        # Добавление важных атрибутов, если есть
                        attrs = info.get("attributes", {})
                        if attrs:
                            char_line += f" [СИЛ: {attrs.get('str', 0)}, ИНТ: {attrs.get('int', 0)}]"
                        
                        char_desc.append(char_line)
                    
                    out.append("\n".join(char_desc))
                else:
                    out.append("В мире пока нет ни одного персонажа.")
            else:
                if name:
                    out.append(f"Информация о персонаже {name} не найдена.")
                else:
                    out.append("Не удалось получить информацию о персонажах.")
        return out

    def _extract_llm_text(self, response):
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, dict) and item.get('type') == 'text' and 'content' in item:
                    text_parts.append(item['content'])
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts)
        elif isinstance(content, dict):
            if 'text' in content:
                return content['text']
            elif 'content' in content and isinstance(content['content'], str):
                return content['content']
        elif hasattr(response, "text"):
            return response.text
        return str(content) if content else str(response)

    def get_provider_info(self) -> dict:
        if self.provider == "deepseek":
            model = LLM_MODEL or DEEPSEEK_MODEL
            return {"provider": "DeepSeek", "model": model}
        elif self.provider == "claude":
            model = LLM_MODEL or CLAUDE_MODEL
            return {"provider": "Anthropic Claude", "model": model}
        elif self.provider == "local":
            model = LLM_MODEL or LOCAL_MODEL
            return {"provider": "Local (Ollama)", "model": model}
        else:
            return {"provider": self.provider, "model": "unknown"} 