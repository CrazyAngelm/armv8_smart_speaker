import os
import json
import paho.mqtt.client as mqtt
from fastapi import FastAPI, Request
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = FastAPI()

# Глобальные переменные: модель LLM и история диалога
llm = None
conversation_history = []

# Адрес Rhasspy
RHASSPY_URL = "http://localhost:12101"

# Системный промпт для модели
system_prompt = """You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed.
Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user.
Speak only russian language. Write only in one language.
"""

def init_llm(model_path: str = "./models/Llama-3.2-3B-Instruct-Q8_0.gguf"):
    """
    Инициализация модели LlamaCpp с базовыми настройками.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    print("Загрузка модели...", model_path)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    stop_sequences = [
        "<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>",
        "Human:", "User:", "Assistant:", "System:"
    ]
    
    _llm = LlamaCpp(
        model_path=model_path,
        temperature=0.2,
        max_tokens=256,
        n_ctx=1024,
        n_batch=512,
        callback_manager=callback_manager,
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
    global conversation_history
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

    # Формируем полный промпт: системный промпт, история (если есть), новый ввод
    prompt = system_prompt + "\n\n"
    if conversation_history:
        prompt += "\n".join(conversation_history) + "\n"
    prompt += f"User: {raw_text}\nAssistant:"

    print("Получен запрос:", raw_text)
    try:
        response = llm.invoke(prompt)
        # Если в ответе есть стоп-маркеры, отсекаем их
        stop_markers = ["<|eot_id|>", "<|end_of_text|>"]
        for marker in stop_markers:
            if marker in response:
                response = response.split(marker)[0].strip()
        print("Ответ модели:", response)
        
        # Обновляем историю диалога (ограничиваем до 10 записей)
        conversation_history.append(f"User: {raw_text}\nAssistant: {response}")
        if len(conversation_history) > 10:
            conversation_history.pop(0)
        
        # Отправляем ответ на Rhasspy для озвучивания
        speak_text_in_rhasspy(response)
        
        return {"status": "ok", "answer": response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
