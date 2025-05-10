import os
import subprocess
import sys
from pathlib import Path
import time
from dotenv import load_dotenv
import asyncio
import websockets
import argparse
import threading

# Загружаем переменные окружения
load_dotenv()

WAKEWORD = os.environ.get('WAKEWORD', 'okey')

# Проверка наличия .env
if not Path('.env').exists():
    print("[ERROR] Файл .env не найден! Создайте .env с нужными переменными окружения.")
    sys.exit(1)

# Проверка наличия venv
venv_path = Path('venv')
if os.name == 'nt':  # Windows
    venv_python = venv_path / 'Scripts' / 'python.exe'
else:  # Linux/Unix
    venv_python = venv_path / 'bin' / 'python'

if not venv_python.exists():
    print("[ERROR] Виртуальное окружение не найдено. Создайте его командой: python -m venv venv")
    sys.exit(1)

# --- Предустановка зависимостей ---
REQUIRED_PACKAGES = [
    'websockets', 'python-dotenv', 'vosk', 'sounddevice', 'webrtcvad', 'soundfile',
    'numpy', 'aiohttp', 'langgraph', 'langchain', 'pocketsphinx'
]
def install_deps():
    try:
        import pkg_resources
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = [pkg for pkg in REQUIRED_PACKAGES if pkg.lower() not in installed]
        if missing:
            print(f"[INFO] Устанавливаю зависимости: {' '.join(missing)}")
            subprocess.check_call([str(venv_python), '-m', 'pip', 'install', *missing])
        else:
            print("[INFO] Все зависимости уже установлены.")
    except Exception as e:
        print(f"[ERROR] Не удалось установить зависимости: {e}")
        sys.exit(1)

install_deps()
# --- Конец блока предустановки ---

# Получаем порты из .env или используем значения по умолчанию
stt_ws_port = int(os.getenv("STT_WS_PORT", 8778))
tts_ws_port = int(os.getenv("TTS_WS_PORT", 8777))
agent_ws_port = int(os.getenv("MAGUS_WS_PORT", 8765))

# Асинхронная проверка WebSocket-сервера
async def wait_for_ws(port, timeout=30):
    uri = f"ws://localhost:{port}"
    print(f"[WAIT] Ожидание сервера на порту {port}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with websockets.connect(uri):
                return True
        except Exception:
            await asyncio.sleep(1)
    print(f"[ERROR] Сервер на порту {port} не запущен после {timeout} секунд ожидания")
    return False

def parse_args():
    parser = argparse.ArgumentParser(description="Smart Speaker")
    parser.add_argument('--cli', '-c', action='store_true', help='Тестовый CLI-режим (только текстовые команды)')
    parser.add_argument('--no-wake', action='store_true', help='Отключить режим пробуждения по ключевому слову')
    parser.add_argument('--wake-word', type=str, default=WAKEWORD, help=f'Установить ключевое слово для пробуждения (по умолчанию {WAKEWORD})')
    parser.add_argument('--device', type=int, help='Индекс устройства ввода звука')
    return parser.parse_args()

async def main():
    args = parse_args()
    
    # Set environment variables for wake word configuration
    if args.wake_word:
        os.environ["WAKEWORD"] = args.wake_word
        print(f"[CONFIG] Установлено ключевое слово: '{args.wake_word}'")
    
    # Toggle wake word mode
    os.environ["USE_WAKE_WORD"] = "false" if args.no_wake else "true"
    if args.no_wake:
        print("[CONFIG] Режим пробуждения отключен, микрофон всегда активен")
    else:
        print(f"[CONFIG] Режим пробуждения включен, ключевое слово: '{WAKEWORD}'")
    
    if args.cli:
        # CLI-режим: только агент, без микрофона и ws-сервисов
        print("[CLI MODE] Запуск агента в текстовом режиме. Введите 'exit' для выхода.")
        import agent
        await agent.cli_loop()
        return

    processes = []
    env = os.environ.copy()

    # --- Запуск backend MQTT ---
    print("[START] Запуск backend MQTT...")
    backend_proc = subprocess.Popen([str(venv_python), 'backend/mqtt_backend.py'], env=env)
    processes.append(backend_proc)
    await asyncio.sleep(2)  # Даем backend подняться

    # Запуск STT сервера
    print("[START] Запуск Vosk STT сервера...")
    stt_proc = subprocess.Popen([str(venv_python), 'vosk_stt.py', 'ws'], env=env)
    processes.append(stt_proc)
    await wait_for_ws(stt_ws_port)
    await asyncio.sleep(1)

    # Запуск TTS сервера
    print("[START] Запуск Piper TTS сервера...")
    tts_proc = subprocess.Popen([str(venv_python), 'piper_tts.py', 'ws'], env=env)
    processes.append(tts_proc)
    await wait_for_ws(tts_ws_port)
    await asyncio.sleep(1)

    # Запуск агента
    print("[START] Запуск голосового агента...")
    agent_proc = subprocess.Popen([str(venv_python), 'agent.py'], env=env)
    processes.append(agent_proc)
    await wait_for_ws(agent_ws_port)
    await asyncio.sleep(2)

    # Запуск микрофонного клиента
    print("[START] Запуск микрофонного клиента...")
    mic_cmd = [str(venv_python), 'mic_client.py']
    if args.no_wake:
        mic_cmd.append('--no-wake')
    if args.device is not None:
        mic_cmd.extend(['--device', str(args.device)])
    mic_proc = subprocess.Popen(mic_cmd, env=env)
    processes.append(mic_proc)

    print("\n[INFO] Все сервисы запущены. Для остановки нажмите Ctrl+C.\n")
    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("[STOP] Остановка сервисов...")
        for proc in processes:
            proc.terminate()
        print("[DONE] Сервисы остановлены.")

if __name__ == "__main__":
    asyncio.run(main()) 