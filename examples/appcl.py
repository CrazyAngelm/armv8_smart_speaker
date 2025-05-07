import os
import json
import paho.mqtt.client as mqtt
from fastapi import FastAPI, Request
from langchain_community.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.prompts import PromptTemplate

app = FastAPI()

# Глобальные переменные: модель LLM и история диалога
llm = None

# Адрес Rhasspy
RHASSPY_URL = "http://localhost:12101"

class Category(BaseModel):
    category: Optional[str] = Field(default=None, description="The category of the message if you know")


def init_llm(model_path: str = "./models/Llama-3.2-3B-Instruct-Q8_0.gguf"):
    """
    Инициализация модели LlamaCpp с базовыми настройками.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    print("Загрузка модели...", model_path)
    callbacks = [StreamingStdOutCallbackHandler()]
    stop_sequences = [
        "<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>",
        "Human:", "User:", "Assistant:", "System:"
    ]
    
    _llm = LlamaCpp(
        model_path=model_path,
        temperature=0.2,
        max_tokens=128,
        n_ctx=1024,
        n_batch=512,
        callback_manager=callbacks,
        use_mlock=True,
        verbose=False,
        seed=42,
        stop=stop_sequences,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.15,
    )
    print("Модель загружена!")
    return _llm

@app.on_event("startup")
def startup_event():
    """
    При старте приложения инициализируется LLM.
    """
    global llm
    model_path = os.getenv("LLM_MODEL_PATH", "./models/Llama-3.2-3B-Instruct-Q8_0.gguf")
    llm = init_llm(model_path=model_path)

def speak_text_in_rhasspy(answer: str):
    """
    Отправляет текст для озвучивания через диалоговую систему Rhasspy
    """
    client = mqtt.Client()
    try:
        client.connect("localhost", 1883, 60)
        
        # Формируем сообщение для управления диалогом
        payload = {
            "siteId": "default",
            "init": {
                "type": "notification",
                "text": answer
            }
        }
        
        # Публикуем сообщение
        print(f"Отправка через MQTT диалог: {payload}")
        client.publish("hermes/dialogueManager/startSession", json.dumps(payload))
        client.disconnect()
        print("Сообщение успешно отправлено через MQTT диалог")
    except Exception as e:
        print(f"Ошибка при отправке через MQTT диалог: {str(e)}")

@app.post("/recognize")
async def recognize(request: Request):
    """
    Эндпоинт принимает JSON с полем "raw_text".
    Формируется промпт на основе системного сообщения, предыдущей истории и нового запроса.
    Затем LLM генерирует ответ, который возвращается в JSON и отправляется на озвучивание.
    
    Пример входного JSON:
    {
       "raw_text": "Как дела?"
    }
    """
    parser = PydanticOutputParser(pydantic_object=Category)

    prompt_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert assistant for classifying user messages for a smart speaker.

    AVAILABLE CATEGORIES:
    1. "weather" – For any text related to weather information, including inquiries about rain, umbrella necessity, temperature, precipitation, forecasts, climate conditions, or similar topics.
    2. "music" – For any text regarding songs, music playback, track selection, music style requests, or any other music-related inquiries.
    3. "time" – For any text concerning time, including requests about the current time, dates, days of the week, months, years, or calendar events.
    4. "about work" – For any text related to work matters, including inquiries about tasks, meetings, office schedules, colleagues, or professional-related content.
    5. "general question" – For texts that do not clearly fit into any of the above categories.

    Note: The user's message may be in any language. Base your classification solely on its meaning.

    Your task:
    Read the user's message carefully and select the single most appropriate category from the list above.
    Output ONLY a JSON object with the key "category", enclosed in triple backticks (```json ... ```). Do not include any additional text or explanation.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Determine the category for the following text:

    {text_input}

    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    ```json
    """




    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text_input"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # get input
    data = await request.json()
    raw_text = data.get("raw_text", "")
    intent = data.get("intent", {}).get("name", "")

    if not raw_text:
        return {"error": "Поле raw_text пустое"}

    print(f"Получен запрос: {raw_text}")

    # Проверяем intent
    if intent != "False":
        print("Интент не 'False', LLM не вызывается.")
        return {"status": "skipped", "reason": f"Интент '{intent}' не требует генерации ответа."}

    chain = prompt | llm | StrOutputParser() | parser

    try:
        category = chain.invoke({"text_input": raw_text})
    
        print("Ответ модели:", category)  # This will print the Category object
        
        final_response = {"status": "ok", "answer": category.model_dump()}
        print("Возвращаю из эндпоинта:", final_response)
        return final_response
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
