import asyncio
import os
import io
import tempfile
import subprocess
import websockets
import shutil
from dotenv import load_dotenv

load_dotenv()

# === КОНФИГУРАЦИЯ ===
# Путь к .onnx-модели и конфигу
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "./models/ru_RU-denis-medium/ru_RU-denis-medium.onnx")
PIPER_SPEAKER_ID = int(os.getenv("PIPER_SPEAKER_ID", "0"))
PIPER_CONFIG_PATH = os.getenv(
    "PIPER_CONFIG_PATH",
    os.path.splitext(PIPER_MODEL_PATH)[0] + ".onnx.json"
)
# Путь к бинарнику piper (если не установлен piper-tts)
PIPER_CMD = os.getenv("PIPER_CMD", "piper")

# === Проверка наличия piper ===
def check_piper_installed():
    try:
        import piper_tts
        return True
    except ImportError:
        try:
            result = subprocess.run([PIPER_CMD, '--help'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

if not check_piper_installed():
    print("[ERROR] Piper TTS не найден. Установите его командой: pip install piper-tts")
    print("[INFO] Или скачайте бинарный файл с https://github.com/rhasspy/piper/releases и укажите путь через PIPER_CMD в .env")

# === Основная функция синтеза ===
async def tts_piper(text: str, model_path: str = None, config_path: str = None, speaker_id: int = None) -> bytes:
    """
    Асинхронный синтез речи через Piper TTS.
    Возвращает WAV-байты.
    """
    model_path = model_path or PIPER_MODEL_PATH
    config_path = config_path or PIPER_CONFIG_PATH
    speaker_id = speaker_id if speaker_id is not None else PIPER_SPEAKER_ID

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"[WARNING] Модель или конфиг не найдены: {model_path}, {config_path}")
        raise FileNotFoundError(f"Модель или конфиг не найдены: {model_path}, {config_path}")

    try:
        # 1. Пробуем использовать Python-обёртку (piper-tts)
        try:
            import piper_tts
            from piper_tts import PiperVoice
            voice = await asyncio.to_thread(PiperVoice.load, model_path, config_path=config_path)
            wav_bytes = await asyncio.to_thread(voice.synthesize_wav_bytes, text, speaker_id=speaker_id)
            return wav_bytes
        except ImportError:
            # 2. Fallback: используем внешний бинарник piper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            try:
                cmd = [
                    PIPER_CMD,
                    '--model', model_path,
                    '--output_file', temp_wav_path,
                    '--speaker', str(speaker_id)
                ]
                print(f"[DEBUG] Запускаю: {' '.join(cmd)}")
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                _, stderr = await process.communicate(input=text.encode('utf-8'))
                if process.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Неизвестная ошибка"
                    raise RuntimeError(f"Piper TTS error: {error_msg}")
                with open(temp_wav_path, 'rb') as wav_file:
                    wav_bytes = wav_file.read()
                return wav_bytes
            finally:
                if os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
    except Exception as e:
        print(f"[ERROR] Piper TTS error: {e}")
        raise

# --- WebSocket TTS server ---
TTS_WS_HOST = os.getenv("TTS_WS_HOST", "0.0.0.0")
TTS_WS_PORT = int(os.getenv("TTS_WS_PORT", 8777))

async def tts_ws_handler(ws):
    try:
        async for message in ws:
            if isinstance(message, str):
                try:
                    wav_bytes = await tts_piper(message)
                    await ws.send(wav_bytes)
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
    async with websockets.serve(tts_ws_handler, TTS_WS_HOST, TTS_WS_PORT, max_size=2**20, ping_interval=30, ping_timeout=30):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ws":
        asyncio.run(main_ws())
    else:
        async def main():
            text = "Привет! Это тест синтеза речи через Piper TTS."
            try:
                wav_bytes = await tts_piper(text)
                with open("output.wav", "wb") as f:
                    f.write(wav_bytes)
                print("Аудиофайл сохранён: output.wav")
            except Exception as e:
                print(f"Ошибка: {e}")
        asyncio.run(main())

# === Инструкции по установке и запуску ===
# 1. pip install piper-tts
# 2. или скачайте piper.exe и укажите путь через PIPER_CMD в .env
# 3. PIPER_MODEL_PATH=путь_к_модели.onnx
# 4. PIPER_CONFIG_PATH=путь_к_конфигу (опционально)
# 5. Проверить: piper --help или python -c "import piper_tts" 