# ARMv8 Smart Speaker

Система распознавания голоса для ARMv8-устройств (Orange Pi 5 Pro, Raspberry Pi и др.) на базе Debian с использованием Vosk STT.

## Возможности

- Запись звука с микрофона через веб-интерфейс
- Распознавание речи с помощью модели Vosk
- Передача аудио и текста через WebSockets
- Простой веб-интерфейс управления
- Оптимизирован для работы на ARMv8 архитектуре

## Компоненты

Система состоит из трех основных компонентов:

1. **mic.py** - Микрофонный сервис, записывает аудио и транслирует его через WebSocket
2. **stt.py** - Сервис распознавания речи, получает аудио от mic.py и преобразует его в текст
3. **main.py** - Основной контроллер, запускает и координирует все сервисы

## Установка

### Требования
- Python 3.7+
- Debian/Ubuntu на устройстве ARMv8 (aarch64)
- ALSA для работы с аудио
- Модель Vosk для русской речи

### Установка зависимостей системы
```bash
sudo apt update
sudo apt install -y python3-venv python3-dev portaudio19-dev libatlas-base-dev
```

### Настройка Python окружения
```bash
# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt
```

### Загрузка модели Vosk
Если модель еще не загружена:

```bash
mkdir -p models
cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip vosk-model-small-ru-0.22.zip
cd ..
```

## Использование

### Запуск через основной контроллер (рекомендуется)
```bash
# Базовый запуск
python3 main.py

# Для просмотра доступных аудио устройств
python3 main.py --list-devices

# Запуск с указанием устройства и портов
python3 main.py --device 1 --mic-web-port 8080 --stt-web-port 8081
```

### Запуск компонентов по отдельности
Если нужно запустить сервисы независимо:

```bash
# Микрофонный сервис
python3 mic.py --device 0 --web-port 8080 --ws-port 8765

# Сервис STT
python3 stt.py --model models/vosk-model-small-ru-0.22 --mic-uri ws://localhost:8765
```

### Доступ к веб-интерфейсам
После запуска доступны следующие веб-интерфейсы:

- **Микрофонный сервис**: http://<IP-адрес-устройства>:8080
- **Сервис STT**: http://<IP-адрес-устройства>:8081
- **Панель управления**: http://<IP-адрес-устройства>:8082

## Настройка для Orange Pi 5 Pro

### Настройка ALSA
```bash
# Установка ALSA утилит
sudo apt install alsa-utils

# Настройка аудиоустройств
alsamixer

# Проверка микрофонов
arecord -l
```

### Разрешения для аудио
```bash
# Добавьте пользователя в audio группу
sudo usermod -a -G audio $USER
# Перезагрузка или выход/вход в систему для применения изменений
```

## Решение проблем

### Не удается найти микрофон
```bash
# Проверьте доступные устройства
python3 main.py --list-devices

# Или напрямую через ALSA
arecord -l
```

### Проблемы с распознаванием речи
```bash
# Проверьте наличие модели Vosk
ls -la models/vosk-model-small-ru-0.22

# Проверьте статус сервисов
curl http://localhost:8082
```

### Ручной перезапуск сервисов
Если сервисы зависли или работают некорректно:
```bash
# Ctrl+C для остановки main.py
# Затем запустите снова
python3 main.py
```

