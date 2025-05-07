#!/usr/bin/env python3
"""
Launcher script for ARMv8 Smart Speaker.
Checks venv, dependencies and starts the main system.
"""
import os
import sys
import subprocess
import platform
import time
from pathlib import Path

def print_banner():
    """Выводит информационный баннер"""
    banner = """
╔══════════════════════════════════════════════════╗
║             ARMv8 Smart Speaker                  ║
║                                                  ║
║  Voice Recognition System for ARMv8 Devices      ║
║  (Orange Pi 5 Pro, Raspberry Pi, etc.)           ║
╚══════════════════════════════════════════════════╝    
    """
    print(banner)

def check_python_version():
    """Проверяет версию Python"""
    required_version = (3, 7)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"[ОШИБКА] Требуется Python {required_version[0]}.{required_version[1]} или выше.")
        print(f"Текущая версия: {current_version[0]}.{current_version[1]}")
        return False
    return True

def setup_venv():
    """Проверяет или создает виртуальное окружение"""
    venv_path = Path('venv')
    
    # Определяем путь к python и pip в зависимости от ОС
    if platform.system() == 'Windows':
        venv_python = venv_path / 'Scripts' / 'python.exe'
        venv_pip = venv_path / 'Scripts' / 'pip.exe'
    else:
        venv_python = venv_path / 'bin' / 'python'
        venv_pip = venv_path / 'bin' / 'pip'
    
    # Проверяем наличие venv
    if not venv_path.exists():
        print("[ИНФО] Виртуальное окружение не найдено, создаем...")
        try:
            subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
            print("[ИНФО] Виртуальное окружение создано.")
        except subprocess.CalledProcessError as e:
            print(f"[ОШИБКА] Не удалось создать виртуальное окружение: {e}")
            return None, None
    
    # Проверяем доступность python и pip в venv
    if not venv_python.exists() or not venv_pip.exists():
        print("[ОШИБКА] Виртуальное окружение повреждено или неполно.")
        return None, None
    
    return str(venv_python), str(venv_pip)

def install_dependencies(venv_pip):
    """Устанавливает зависимости из requirements.txt"""
    if not Path('requirements.txt').exists():
        print("[ОШИБКА] Файл requirements.txt не найден.")
        return False
    
    print("[ИНФО] Устанавливаем зависимости...")
    try:
        subprocess.run([venv_pip, 'install', '-r', 'requirements.txt'], check=True)
        print("[ИНФО] Зависимости установлены.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ОШИБКА] Не удалось установить зависимости: {e}")
        return False

def check_alsa():
    """Проверяет доступность ALSA (только для Linux)"""
    if platform.system() != 'Linux':
        return True  # Пропускаем для не-Linux систем
    
    try:
        proc = subprocess.run(['arecord', '--version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              text=True)
        if proc.returncode == 0:
            print("[ИНФО] ALSA установлена и доступна.")
            return True
        else:
            print("[ПРЕДУПРЕЖДЕНИЕ] ALSA недоступна. Возможно, потребуется установка:")
            print("    sudo apt install alsa-utils")
            return False
    except FileNotFoundError:
        print("[ПРЕДУПРЕЖДЕНИЕ] ALSA не найдена. Установите:")
        print("    sudo apt install alsa-utils")
        return False

def run_main(venv_python, args=None):
    """Запускает основную программу в venv"""
    cmd = [venv_python, 'main.py']
    if args:
        cmd.extend(args)
    
    print("[ИНФО] Запуск системы распознавания речи...")
    try:
        subprocess.run(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ОШИБКА] Ошибка запуска: {e}")
        return False
    except KeyboardInterrupt:
        print("\n[ИНФО] Прервано пользователем.")
        return True

def main():
    """Основная функция"""
    print_banner()
    
    # Проверяем версию Python
    if not check_python_version():
        return 1
    
    # Настраиваем venv
    venv_python, venv_pip = setup_venv()
    if not venv_python or not venv_pip:
        return 1
    
    # Устанавливаем зависимости
    if not install_dependencies(venv_pip):
        return 1
    
    # Проверяем ALSA для Linux систем
    check_alsa()
    
    # Обрабатываем аргументы командной строки
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    
    # Запускаем основную программу
    success = run_main(venv_python, args)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
