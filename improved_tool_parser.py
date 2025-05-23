# optimized_tool_parser.py
import re
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    confidence: float = 1.0

class OptimizedToolParser:
    """Оптимизированный парсер инструментов для маленьких моделей"""
    
    def __init__(self):
        # Расширенный словарь чисел (включая составные)
        self.text_numbers = {
            # Единицы
            "ноль": 0, "один": 1, "одну": 1, "одна": 1, "два": 2, "две": 2, "три": 3,
            "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
            # Десятки
            "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
            "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
            "восемнадцать": 18, "девятнадцать": 19, "двадцать": 20, "тридцать": 30,
            "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
            # Составные числа
            "двадцать пять": 25, "тридцать пять": 35, "сорок пять": 45,
            "пол": 0.5, "полчаса": 30, "четверть": 15,
        }
        
        # Улучшенные паттерны с синонимами
        self.tool_patterns = {
            "get_time": {
                "keywords": ["время", "час", "который час", "сколько времени", "текущее время", "время сейчас"],
                "patterns": [
                    r"(?:сколько|который)\s+(?:сейчас\s+)?час",
                    r"какое\s+(?:сейчас\s+)?время",
                    r"время\s+(?:сейчас|сегодня)",
                    r"current\s+time", r"what\s+time",
                    r"сколько\s+время",
                    r"только\s+время"
                ],
                "priority": 5,  # Высокий приоритет для времени
                "confidence_boost": 0.3
            },
            
            "get_weather": {
                "keywords": ["погода", "weather", "температура", "градус", "дождь", "солнце", 
                           "облачно", "ясно", "пасмурно", "снег", "ветер"],
                "patterns": [
                    r"какая\s+(?:сегодня\s+|сейчас\s+)?погода",
                    r"погода\s+(?:на\s+)?(?:сегодня|завтра|сейчас)",
                    r"температура\s+(?:на\s+)?(?:улице|сегодня|сейчас)",
                    r"(?:дождь|солнце|облачно|ясно|снег)",
                    r"weather\s+(?:today|tomorrow|now)"
                ],
                "priority": 3,  # Средний приоритет
                "confidence_boost": 0.2
            },
            
            "set_timer": {
                "keywords": ["таймер", "поставь таймер", "установи таймер", "засеки", "timer"],
                "patterns": [
                    r"(?:поставь|установи|засеки|включи|запусти)\s+таймер",
                    r"таймер\s+на\s+(\d+|\w+)",
                    r"(\d+|\w+)\s+(?:минут|мин|секунд|сек|час)",
                    r"timer\s+(?:for\s+)?(\d+)"
                ],
                "priority": 4,
                "confidence_boost": 0.2
            },
            
            "set_notification": {
                "keywords": ["напомни", "напоминание", "уведомление", "reminder", "notify", "напомни мне"],
                "patterns": [
                    r"напомни\s+(?:мне\s+)?(?:о\s+)?(.+?)(?:\s+через\s+(\d+|\w+))?",
                    r"напоминание\s+(?:о\s+)?(.+)",
                    r"remind\s+me\s+(?:to\s+)?(.+)"
                ],
                "priority": 3,
                "confidence_boost": 0.2
            },
            
            "call_contact": {
                "keywords": ["позвони", "звони", "call", "набери номер", "вызов", "звонок"],
                "patterns": [
                    r"(?:позвони|звони|набери)\s+(.+)",
                    r"call\s+(.+)",
                    r"(?:сделай\s+)?звонок\s+(.+)"
                ],
                "priority": 3,
                "confidence_boost": 0.2
            }
        }
        
        # Минимальная уверенность для выполнения инструмента
        self.min_confidence = 0.4
        
        # Простой системный промпт для LLM
        self.simple_system_prompt = """Отвечай ТОЛЬКО одним словом:
- ВРЕМЯ - если спрашивают про время
- ПОГОДА - если спрашивают про погоду  
- ТАЙМЕР - если просят поставить таймер
- НАПОМИНАНИЕ - если просят напомнить
- ЗВОНОК - если просят позвонить
- НЕТ - если это обычный разговор"""

    def parse_text_for_tools(self, text: str, use_llm_fallback: bool = True) -> Optional[List[ToolCall]]:
        """Основной метод парсинга с возможностью LLM fallback"""
        text_lower = text.lower().strip()
        
        # 1. Сначала пробуем прямой парсинг
        direct_result = self._parse_direct(text, text_lower)
        if direct_result and direct_result[0].confidence >= self.min_confidence:
            return direct_result
        
        # 2. Если уверенность низкая или нет результата, можем использовать LLM
        if use_llm_fallback and (not direct_result or direct_result[0].confidence < 0.6):
            llm_result = self._parse_with_llm_hint(text, text_lower)
            if llm_result and llm_result[0].confidence >= self.min_confidence:
                return llm_result
        
        # 3. Возвращаем прямой результат если он есть и превышает минимум
        if direct_result and direct_result[0].confidence >= self.min_confidence:
            return direct_result
            
        return None

    def _parse_direct(self, text: str, text_lower: str) -> Optional[List[ToolCall]]:
        """Прямой парсинг без LLM"""
        # Парсинг по тегам [ДЕЙСТВИЕ]
        tag_result = self._parse_action_tags(text)
        if tag_result:
            return tag_result
            
        # Парсинг по приоритетам и ключевым словам
        priority_result = self._parse_by_priority(text_lower)
        if priority_result:
            return priority_result
            
        return None

    def _parse_by_priority(self, text: str) -> Optional[List[ToolCall]]:
        """Парсинг с учетом приоритетов инструментов"""
        matches = []
        
        for tool_name, config in self.tool_patterns.items():
            confidence = 0.0
            
            # Проверяем ключевые слова
            keyword_matches = sum(1 for keyword in config["keywords"] if keyword in text)
            confidence += keyword_matches * 0.3
            
            # Проверяем паттерны
            for pattern in config["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    confidence += 0.5
                    break
            
            # Добавляем boost уверенности
            confidence += config.get("confidence_boost", 0)
            
            if confidence >= 0.3:
                args = self._extract_args(tool_name, text)
                matches.append(ToolCall(name=tool_name, args=args, confidence=confidence))
        
        if not matches:
            return None
        
        # СПЕЦИАЛЬНАЯ ЛОГИКА: Разрешение конфликтов время vs погода
        time_matches = [m for m in matches if m.name == "get_time"]
        weather_matches = [m for m in matches if m.name == "get_weather"]
        
        # Если есть и время и погода, применяем специальные правила
        if time_matches and weather_matches:
            # Ключевые слова только для времени
            time_only_words = ["только время", "сколько время", "который час", "час сейчас"]
            if any(word in text.lower() for word in time_only_words):
                return [time_matches[0]]
            
            # Ключевые слова только для погоды  
            weather_only_words = ["температура", "погода", "дождь", "солнце", "облачно", "ясно", "снег", "ветер"]
            if any(word in text.lower() for word in weather_only_words):
                return [weather_matches[0]]
            
            # Если неоднозначно, выбираем время (так как запросы времени чаще)
            return [time_matches[0]]
        
        # Обычная сортировка по приоритету и уверенности
        matches.sort(key=lambda x: (
            self.tool_patterns[x.name].get("priority", 1),
            x.confidence
        ), reverse=True)
        
        return [matches[0]]

    def _parse_with_llm_hint(self, original_text: str, text_lower: str) -> Optional[List[ToolCall]]:
        """Placeholder для LLM-помощи. Будет реализован позже"""
        # Этот метод будет вызывать LLM для получения подсказки
        # и затем применять парсинг к ответу LLM
        return None

    def _parse_action_tags(self, text: str) -> Optional[List[ToolCall]]:
        """Парсинг по тегам [ДЕЙСТВИЕ]"""
        action_map = {
            "время": "get_time", "таймер": "set_timer", 
            "напоминание": "set_notification", "погода": "get_weather",
            "звонок": "call_contact"
        }
        
        pattern = r'\[([^\]]+)\]\s*(.+)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            action = match.group(1).lower().strip()
            description = match.group(2).strip()
            
            if action in action_map:
                tool_name = action_map[action]
                args = self._extract_args(tool_name, description)
                return [ToolCall(name=tool_name, args=args, confidence=0.9)]
                
        return None

    def _extract_args(self, tool_name: str, text: str) -> Dict[str, Any]:
        """Универсальный экстрактор аргументов"""
        if tool_name == "get_time":
            return {}
        elif tool_name == "get_weather":
            return {}
        elif tool_name == "set_timer":
            return self._extract_timer_args(text)
        elif tool_name == "set_notification":
            return self._extract_notification_args(text)
        elif tool_name == "call_contact":
            return self._extract_call_args(text)
        return {}

    def _parse_number(self, text: str) -> Optional[int]:
        """Умный парсинг чисел (цифры + текст)"""
        # Сначала ищем цифры
        digit_match = re.search(r'\b(\d+)\b', text)
        if digit_match:
            return int(digit_match.group(1))
        
        # Затем ищем текстовые числа
        for word, number in self.text_numbers.items():
            if word in text.lower():
                return int(number)
        
        return None

    def _parse_text_number(self, text: str) -> int:
        """Парсит составные текстовые числа (например, 'двадцать две' -> 22)"""
        text = text.lower().strip()
        # Сначала ищем точное совпадение
        if text in self.text_numbers:
            return int(self.text_numbers[text])
        # Затем ищем комбинации десятков и единиц
        parts = text.split()
        if len(parts) == 2:
            tens, ones = parts
            if tens in self.text_numbers and ones in self.text_numbers:
                return int(self.text_numbers[tens]) + int(self.text_numbers[ones])
        return None

    def _extract_timer_args(self, text: str) -> Dict[str, Any]:
        """Улучшенный экстрактор для таймера"""
        args = {}
        # Сначала ищем "через <текстовое число> секунд"
        match = re.search(r'через\s+([а-яА-Я\s]+)\s+(?:секунд|сек)', text, re.IGNORECASE)
        if match:
            num = self._parse_text_number(match.group(1))
            if num:
                args["seconds"] = num
        # Если не нашли, ищем "через <текстовое число> минут"
        if not args:
            match = re.search(r'через\s+([а-яА-Я\s]+)\s+(?:минут|мин)', text, re.IGNORECASE)
            if match:
                num = self._parse_text_number(match.group(1))
                if num:
                    args["minutes"] = num
        # Если не нашли, ищем "через <текстовое число> час"
        if not args:
            match = re.search(r'через\s+([а-яА-Я\s]+)\s*час', text, re.IGNORECASE)
            if match:
                num = self._parse_text_number(match.group(1))
                if num:
                    args["hours"] = num
        # Если не нашли, ищем цифровые значения
        if not args:
            match = re.search(r'через\s+(\d+)\s*(секунд|сек)', text)
            if match:
                args["seconds"] = int(match.group(1))
        if not args:
            match = re.search(r'через\s+(\d+)\s*(минут|мин)', text)
            if match:
                args["minutes"] = int(match.group(1))
        if not args:
            match = re.search(r'через\s+(\d+)\s*час', text)
            if match:
                args["hours"] = int(match.group(1))
        # Если не нашли, ищем просто "<текстовое число> секунд/минут/час" без "через"
        if not args:
            match = re.search(r'([а-яА-Я\s]+)\s+(секунд|сек)', text, re.IGNORECASE)
            if match:
                num = self._parse_text_number(match.group(1))
                if num:
                    args["seconds"] = num
        if not args:
            match = re.search(r'([а-яА-Я\s]+)\s+(минут|мин)', text, re.IGNORECASE)
            if match:
                num = self._parse_text_number(match.group(1))
                if num:
                    args["minutes"] = num
        if not args:
            match = re.search(r'([а-яА-Я\s]+)\s*час', text, re.IGNORECASE)
            if match:
                num = self._parse_text_number(match.group(1))
                if num:
                    args["hours"] = num
        # Если не нашли, ищем цифровые значения без "через"
        if not args:
            match = re.search(r'(\d+)\s*(секунд|сек)', text)
            if match:
                args["seconds"] = int(match.group(1))
        if not args:
            match = re.search(r'(\d+)\s*(минут|мин)', text)
            if match:
                args["minutes"] = int(match.group(1))
        if not args:
            match = re.search(r'(\d+)\s*час', text)
            if match:
                args["hours"] = int(match.group(1))
        # По умолчанию 1 минута
        if not args:
            args["minutes"] = 1
        return args

    def _extract_notification_args(self, text: str) -> Dict[str, Any]:
        """Улучшенный экстрактор для напоминаний"""
        args = {}
        # Сначала ищем "через <текстовое число> секунд"
        match = re.search(r'через\s+([а-яА-Я\s]+)\s+(?:секунд|сек)', text, re.IGNORECASE)
        if match:
            num = self._parse_text_number(match.group(1))
            if num:
                args["seconds"] = num
        # Если не нашли, ищем "через <текстовое число> минут"
        if not args:
            match = re.search(r'через\s+([а-яА-Я\s]+)\s+(?:минут|мин)', text, re.IGNORECASE)
            if match:
                num = self._parse_text_number(match.group(1))
                if num:
                    args["minutes"] = num
        # Если не нашли, ищем "через <текстовое число> час"
        if not args:
            match = re.search(r'через\s+([а-яА-Я\s]+)\s*час', text, re.IGNORECASE)
            if match:
                num = self._parse_text_number(match.group(1))
                if num:
                    args["hours"] = num
        # Если не нашли, ищем цифровые значения
        if not args:
            match = re.search(r'через\s+(\d+)\s*(секунд|сек)', text)
            if match:
                args["seconds"] = int(match.group(1))
        if not args:
            match = re.search(r'через\s+(\d+)\s*(минут|мин)', text)
            if match:
                args["minutes"] = int(match.group(1))
        if not args:
            match = re.search(r'через\s+(\d+)\s*час', text)
            if match:
                args["hours"] = int(match.group(1))
        # Если не нашли, ищем просто "<текстовое число> секунд/минут/час" без "через"
        if not args:
            match = re.search(r'([а-яА-Я\s]+)\s+(секунд|сек)', text, re.IGNORECASE)
            if match:
                num = self._parse_text_number(match.group(1))
                if num:
                    args["seconds"] = num
        if not args:
            match = re.search(r'([а-яА-Я\s]+)\s+(минут|мин)', text, re.IGNORECASE)
            if match:
                num = self._parse_text_number(match.group(1))
                if num:
                    args["minutes"] = num
        if not args:
            match = re.search(r'([а-яА-Я\s]+)\s*час', text, re.IGNORECASE)
            if match:
                num = self._parse_text_number(match.group(1))
                if num:
                    args["hours"] = num
        # Если не нашли, ищем цифровые значения без "через"
        if not args:
            match = re.search(r'(\d+)\s*(секунд|сек)', text)
            if match:
                args["seconds"] = int(match.group(1))
        if not args:
            match = re.search(r'(\d+)\s*(минут|мин)', text)
            if match:
                args["minutes"] = int(match.group(1))
        if not args:
            match = re.search(r'(\d+)\s*час', text)
            if match:
                args["hours"] = int(match.group(1))
        # Если не нашли, по умолчанию 5 минут
        if not args:
            args["minutes"] = 5
        # Извлекаем текст напоминания - убираем команду и время
        text_clean = text
        text_clean = re.sub(r'^(?:поставь\s+)?напомни(е)?\s*(?:мне\s+)?', '', text_clean, flags=re.IGNORECASE)
        text_clean = re.sub(r'через\s+(?:\d+|[а-яА-Я\s]+)\s*(?:секунд|сек|минут|мин|час|часов)', '', text_clean, flags=re.IGNORECASE)
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        if not text_clean:
            text_clean = "Напоминание"
        args["text"] = text_clean
        return args

    def _extract_call_args(self, text: str) -> Dict[str, Any]:
        """Экстрактор для звонков"""
        args = {}
        
        call_patterns = [
            r'(?:позвони|звони|набери)\s+(.+)',
            r'call\s+(.+)',
            r'звонок\s+(.+)'
        ]
        
        for pattern in call_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                contact = match.group(1).strip()
                # Убираем лишние слова
                contact = re.sub(r'\b(?:номер|телефон)\b', '', contact).strip()
                args["contact_name"] = contact
                break
                
        if not args.get("contact_name"):
            args["contact_name"] = "неизвестный контакт"
            
        return args

    def get_simple_system_prompt(self) -> str:
        """Возвращает простой системный промпт для LLM"""
        return self.simple_system_prompt

    def set_confidence_threshold(self, threshold: float):
        """Устанавливает минимальный порог уверенности"""
        self.min_confidence = max(0.0, min(1.0, threshold))


# Фабричная функция
def create_optimized_tool_parser():
    return OptimizedToolParser()


# Тестирование
if __name__ == "__main__":
    parser = OptimizedToolParser()
    
    test_cases = [
        ("какая сейчас погода", "get_weather"),
        ("поставь таймер на пять минут", "set_timer"),
        ("напомни купить хлеб", "set_notification"),
        ("который час", "get_time"),
        ("позвони маме", "call_contact"),
        ("привет как дела", None),  # Не должно распознаваться
    ]
    
    print("🧪 Тестирование оптимизированного парсера\n")
    
    for text, expected in test_cases:
        result = parser.parse_text_for_tools(text, use_llm_fallback=False)
        actual = result[0].name if result else None
        confidence = result[0].confidence if result else 0.0
        
        status = "✅" if actual == expected else "❌"
        print(f"{status} '{text}' → {actual} (conf: {confidence:.2f})")
        
        if result and result[0].args:
            print(f"    Аргументы: {result[0].args}")