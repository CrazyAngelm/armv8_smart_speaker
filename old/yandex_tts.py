import aiohttp
import os
import asyncio
from dotenv import load_dotenv
import websockets

load_dotenv()

YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_IAM_TOKEN = os.getenv("YANDEX_IAM_TOKEN")
YANDEX_TTS_VOICE = os.getenv("YANDEX_TTS_VOICE", "alena")

async def tts_yandex(text: str, folder_id: str = None, iam_token: str = None, voice: str = None) -> bytes:
    """
    Асинхронный синтез речи через Yandex SpeechKit TTS API.
    Возвращает oggopus-байты.
    """
    folder_id = folder_id or YANDEX_FOLDER_ID
    iam_token = iam_token or YANDEX_IAM_TOKEN
    voice = voice or YANDEX_TTS_VOICE
    url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    headers = {
        "Authorization": f"Bearer {iam_token}",
    }
    data = {
        "text": text,
        "lang": "ru-RU",
        "voice": voice,
        "format": "oggopus",
        "sampleRateHertz": "48000",
        "folderId": folder_id,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data, timeout=15) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                err = await resp.text()
                raise RuntimeError(f"Yandex TTS error {resp.status}: {err}")

# --- WebSocket TTS server ---
TTS_WS_HOST = os.getenv("TTS_WS_HOST", "0.0.0.0")
TTS_WS_PORT = int(os.getenv("TTS_WS_PORT", 8777))

async def tts_ws_handler(ws):
    try:
        async for message in ws:
            if isinstance(message, str):
                try:
                    ogg_bytes = await tts_yandex(message)
                    await ws.send(ogg_bytes)
                except Exception as e:
                    await ws.send(f"ERROR: {e}")
            else:
                await ws.send("ERROR: Only text messages supported")
    except websockets.exceptions.ConnectionClosedError:
        pass
    except Exception as e:
        print(f"[TTS WS] Unexpected error: {e}")

async def main_ws():
    print(f"[TTS WS] Starting server on port {TTS_WS_PORT}")
    async with websockets.serve(tts_ws_handler, TTS_WS_HOST, TTS_WS_PORT, max_size=8*2**20, ping_interval=30, ping_timeout=30):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ws":
        asyncio.run(main_ws())
    else:
        async def main():
            text = "Привет! Это тест синтеза речи через Яндекс SpeechKit."
            try:
                ogg_bytes = await tts_yandex(text)
                with open("output.ogg", "wb") as f:
                    f.write(ogg_bytes)
                print("Аудиофайл сохранён: output.ogg")
            except Exception as e:
                print(f"Ошибка: {e}")
        asyncio.run(main()) 