import asyncio
import os
import platform
import subprocess
import tempfile
import websockets
import sys
from dotenv import load_dotenv

load_dotenv()

# === КОНФИГУРАЦИЯ ===
# Путь к .onnx-модели и конфигу
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "./models/ru_RU-denis-medium/ru_RU-denis-medium.onnx")
PIPER_SPEAKER_ID = int(os.getenv("PIPER_SPEAKER_ID", "0"))

# Путь к бинарнику piper
if platform.system() == "Windows":
    PIPER_CMD = os.getenv("PIPER_CMD", ".\\piper_windows\\piper.exe")
else:  # Linux/Unix
    PIPER_CMD = os.getenv("PIPER_CMD", "./piper/piper")

# === WebSocket TTS сервер ===
TTS_WS_HOST = os.getenv("TTS_WS_HOST", "0.0.0.0")
TTS_WS_PORT = int(os.getenv("TTS_WS_PORT", 8777))

# === Проверка наличия piper ===
def check_piper_installed():
    try:
        if platform.system() == "Windows":
            result = subprocess.run([PIPER_CMD, '--help'], capture_output=True, text=True, shell=True)
        else:
            result = subprocess.run([PIPER_CMD, '--help'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

if not check_piper_installed():
    print(f"[ERROR] Piper TTS не найден по пути: {PIPER_CMD}")
    print("[INFO] Скачайте piper с https://github.com/rhasspy/piper/releases и укажите путь через PIPER_CMD в .env")

# === Основная функция синтеза ===
async def tts_piper(text: str, model_path: str = None, speaker_id: int = None) -> bytes:
    """
    Асинхронный синтез речи через Piper TTS.
    Возвращает WAV-байты.
    """
    model_path = model_path or PIPER_MODEL_PATH
    speaker_id = speaker_id if speaker_id is not None else PIPER_SPEAKER_ID

    if not os.path.exists(model_path):
        print(f"[WARNING] Модель не найдена: {model_path}")
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav_path = temp_wav.name

    try:
        cmd = [
            PIPER_CMD,
            '--model', model_path,
            '--output_file', temp_wav_path,
        ]
        
        if speaker_id > 0:
            cmd.extend(['--speaker', str(speaker_id)])
            
        print(f"[DEBUG] Запускаю: {' '.join(cmd)}")
        
        if platform.system() == "Windows":
            # Windows: запись текста во временный файл и редирект через тип
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', encoding='utf-8', delete=False) as temp_txt:
                temp_txt_path = temp_txt.name
                temp_txt.write(text)
            
            # Формируем команду для PowerShell
            cmd_str = ' '.join(cmd)
            full_cmd = f"type \"{temp_txt_path}\" | {cmd_str}"
            
            # Запускаем процесс через PowerShell
            process = await asyncio.create_subprocess_shell(
                full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            # Удаляем временный текстовый файл
            if os.path.exists(temp_txt_path):
                os.unlink(temp_txt_path)
        else:
            # Linux/Unix использует stdin
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate(input=text.encode('utf-8'))
            
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='replace') if stderr else "Неизвестная ошибка"
            raise RuntimeError(f"Piper TTS error: {error_msg}")
            
        # Чтение созданного WAV файла
        if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
            with open(temp_wav_path, 'rb') as wav_file:
                wav_bytes = wav_file.read()
            return wav_bytes
        else:
            raise RuntimeError("Piper не создал аудиофайл или файл пустой")
    
    finally:
        # Удаление временного WAV файла
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

# === WebSocket обработчик ===
async def tts_ws_handler(ws):
    try:
        async for message in ws:
            if isinstance(message, str):
                try:
                    wav_bytes = await tts_piper(message)
                    await ws.send(wav_bytes)
                except Exception as e:
                    print(f"[ERROR] TTS ошибка: {e}")
                    await ws.send(f"ERROR: {e}")
            else:
                await ws.send("ERROR: Only text messages supported")
    except websockets.exceptions.ConnectionClosedError:
        pass
    except Exception as e:
        print(f"[TTS WS] Unexpected error: {e}")

async def main_ws():
    print(f"[TTS WS] Starting server on port {TTS_WS_PORT}")
    async with websockets.serve(
        tts_ws_handler, 
        TTS_WS_HOST, 
        TTS_WS_PORT, 
        max_size=8*2**20, 
        ping_interval=300,   # 5 минут
        ping_timeout=None):  # Без таймаута
        await asyncio.Future()  # run forever

# === Тестовый запуск ===
async def test_tts():
    text = "Привет! Это тест синтеза речи через Piper TTS."
    try:
        wav_bytes = await tts_piper(text)
        with open("output.wav", "wb") as f:
            f.write(wav_bytes)
        print("Аудиофайл сохранён: output.wav")
    except Exception as e:
        print(f"Ошибка: {e}")

# === Точка входа ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ws":
        asyncio.run(main_ws())
    else:
        asyncio.run(test_tts()) 