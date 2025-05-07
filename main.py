#!/usr/bin/env python3
"""
Главный контроллер системы распознавания речи для ARMv8 (Debian).
Запускает и координирует микрофонный сервис и сервис распознавания речи.
"""
import argparse
import subprocess
import sys
import os
import signal
import time

# Конфигурация
MIC_WS_PORT = 8765
STT_WS_PORT = 8778
VOSK_MODEL_PATH = os.path.join("models", "vosk-model-small-ru-0.22")

# Глобальные переменные для управления процессами
mic_process = None
stt_process = None
should_exit = False

def check_vosk_model():
    """Проверка наличия модели Vosk"""
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"[ОШИБКА] Модель Vosk не найдена: {VOSK_MODEL_PATH}")
        print("Загрузите модель или укажите корректный путь с помощью --model")
        return False
    return True

def start_mic_service(device=None, port=MIC_WS_PORT):
    """Запуск микрофонного сервиса"""
    global mic_process
    
    cmd = [sys.executable, "mic.py"]
    if device is not None:
        cmd.extend(["--device", str(device)])
    cmd.extend(["--port", str(port)])
    
    print(f"[ИНФО] Запуск микрофонного сервиса: {' '.join(cmd)}")
    mic_process = subprocess.Popen(cmd)
    return mic_process.poll() is None  # Проверка, запустился ли процесс

def start_stt_service(model_path=VOSK_MODEL_PATH, mic_uri=None, port=STT_WS_PORT):
    """Запуск сервиса распознавания речи"""
    global stt_process
    
    cmd = [sys.executable, "stt.py"]
    cmd.extend(["--model", model_path])
    
    if mic_uri:
        cmd.extend(["--mic-uri", mic_uri])
    else:
        mic_uri = f"ws://localhost:{MIC_WS_PORT}"
        cmd.extend(["--mic-uri", mic_uri])
    
    cmd.extend(["--port", str(port)])
    
    print(f"[ИНФО] Запуск сервиса распознавания речи: {' '.join(cmd)}")
    stt_process = subprocess.Popen(cmd)
    return stt_process.poll() is None  # Проверка, запустился ли процесс

def stop_services():
    """Остановка всех сервисов"""
    global mic_process, stt_process
    
    print("[ИНФО] Остановка сервисов...")
    
    if stt_process and stt_process.poll() is None:
        print("[ИНФО] Остановка сервиса распознавания речи...")
        stt_process.terminate()
        try:
            stt_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stt_process.kill()
    
    if mic_process and mic_process.poll() is None:
        print("[ИНФО] Остановка микрофонного сервиса...")
        mic_process.terminate()
        try:
            mic_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mic_process.kill()
    
    print("[ИНФО] Все сервисы остановлены")

def signal_handler(sig, frame):
    """Обработка сигналов прерывания"""
    global should_exit
    print("\n[ИНФО] Получен сигнал прерывания, завершение работы...")
    should_exit = True
    stop_services()
    sys.exit(0)

def monitor_processes():
    """Мониторинг процессов и их перезапуск при необходимости"""
    global should_exit, mic_process, stt_process
    
    while not should_exit:
        # Проверяем микрофонный процесс
        if mic_process and mic_process.poll() is not None:
            print("[ПРЕДУПРЕЖДЕНИЕ] Микрофонный сервис неожиданно остановился, перезапуск...")
            start_mic_service()
        
        # Проверяем STT процесс
        if stt_process and stt_process.poll() is not None:
            print("[ПРЕДУПРЕЖДЕНИЕ] Сервис распознавания речи неожиданно остановился, перезапуск...")
            mic_uri = f"ws://localhost:{MIC_WS_PORT}"
            start_stt_service(mic_uri=mic_uri)
        
        time.sleep(2)  # Интервал проверки (2 секунды)

def main():
    """Главная функция запуска и контроля"""
    global MIC_WS_PORT, STT_WS_PORT, VOSK_MODEL_PATH
    
    # Разбор аргументов командной строки
    parser = argparse.ArgumentParser(description="Контроллер системы распознавания речи для ARMv8")
    parser.add_argument("--device", type=int, help="Индекс аудио устройства")
    parser.add_argument("--mic-port", type=int, default=MIC_WS_PORT, help="Порт WebSocket микрофонного сервиса")
    parser.add_argument("--stt-port", type=int, default=STT_WS_PORT, help="Порт WebSocket сервиса распознавания речи")
    parser.add_argument("--model", type=str, default=VOSK_MODEL_PATH, help="Путь к модели Vosk")
    parser.add_argument("--list-devices", action="store_true", help="Показать список аудио устройств и выйти")
    args = parser.parse_args()
    
    # Обновляем конфигурацию из аргументов
    MIC_WS_PORT = args.mic_port
    STT_WS_PORT = args.stt_port
    VOSK_MODEL_PATH = args.model
    
    # Проверяем наличие опции отображения устройств
    if args.list_devices:
        print("[ИНФО] Список аудио устройств...")
        subprocess.run([sys.executable, "mic.py", "--list-devices"])
        return
    
    # Проверяем наличие модели Vosk
    if not check_vosk_model():
        return
    
    # Настраиваем обработчики сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n[ИНФО] Запуск системы распознавания речи для ARMv8")
    print(f"[ИНФО] Модель Vosk: {VOSK_MODEL_PATH}")
    
    # Запуск микрофонного сервиса
    if not start_mic_service(device=args.device, port=MIC_WS_PORT):
        print("[ОШИБКА] Не удалось запустить микрофонный сервис")
        return
    
    # Даем время на запуск микрофонного сервиса
    time.sleep(1)
    
    # Запуск сервиса распознавания речи
    mic_uri = f"ws://localhost:{MIC_WS_PORT}"
    if not start_stt_service(model_path=VOSK_MODEL_PATH, mic_uri=mic_uri, port=STT_WS_PORT):
        print("[ОШИБКА] Не удалось запустить сервис распознавания речи")
        stop_services()
        return
    
    # Запуск мониторинга процессов
    print("[ИНФО] Все сервисы запущены")
    print("\n[ИНФО] Сервисы запущены:")
    print(f"  Микрофонный сервис: ws://localhost:{MIC_WS_PORT}")
    print(f"  Сервис распознавания: ws://localhost:{STT_WS_PORT}")
    print("\n[ИНФО] Для выхода нажмите Ctrl+C\n")
    
    try:
        # Запускаем мониторинг процессов
        monitor_processes()
    except KeyboardInterrupt:
        pass
    finally:
        stop_services()

if __name__ == "__main__":
    main() 