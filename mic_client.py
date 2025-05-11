import asyncio
import websockets
import sounddevice as sd
import webrtcvad
import numpy as np
import threading
import time
import os
import sys
from dotenv import load_dotenv
import soundfile as sf
import queue
from wake_detector import WakeWordDetector

# --- CONFIG & GLOBALS ---
load_dotenv()
HOST = os.getenv("MAGUS_WS_HOST", "localhost")
PORT = int(os.getenv("MAGUS_WS_PORT", 8765))
URI = f"ws://{HOST}:{PORT}"
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000))
SILENCE_THRESHOLD_FRAMES = int(1000 / FRAME_DURATION_MS)  # ~1 секунда тишины
SPEECH_START_THRESHOLD = 3
WAKEWORD = os.getenv("WAKEWORD", "okey")

# Enable or disable wake word detection
USE_WAKE_WORD = os.getenv("USE_WAKE_WORD", "true").lower() in ("true", "1", "yes")

vad = webrtcvad.Vad()
vad.set_mode(3)

# Shared state between wake detection and command detection
wake_event = threading.Event()
audio_queue = queue.Queue()
wake_detector = None  # Will be initialized in main()

# --- UTILS ---
def print_available_devices():
    print("\nДоступные устройства ввода звука:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"Индекс {i}: {dev['name']} (входы: {dev['max_input_channels']}, выходы: {dev['max_output_channels']})")
    print("\nДля выбора устройства при запуске используйте параметр --device <индекс>")

def play_audio(audio_data):
    import tempfile
    
    # Определяем тип файла по заголовку WAV
    is_wav = audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE'
    suffix = '.wav' if is_wav else '.ogg'
    
    # Используем tempfile.NamedTemporaryFile для создания уникальных имен файлов
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        try:
            temp_file.write(audio_data)
            temp_file.close()
            
            try:
                data, samplerate = sf.read(temp_file.name, dtype='float32')
                print(f"[INFO] Воспроизведение аудио: {len(data)} сэмплов, {samplerate} Гц")
                sd.play(data, samplerate)
                sd.wait()
            except Exception as e:
                print(f"[ERROR] Ошибка воспроизведения аудио: {e}")
                
        except Exception as e:
            print(f"[ERROR] Не удалось воспроизвести ответ: {e}")
        finally:
            try:
                if os.path.exists(temp_file.name):
                    # Небольшая пауза перед удалением чтобы убедиться, что файл больше не используется
                    time.sleep(0.1)
                    os.remove(temp_file.name)
            except Exception as e:
                print(f"[WARNING] Не удалось удалить временный файл: {e}")

# --- WAKE WORD HANDLING ---
def on_wake_word_detected(detected_text):
    """Called when wake word is detected"""
    print(f"[WAKE] Detected wake word: '{detected_text}', now listening for command...")
    play_audio(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")  # Short beep sound
    wake_detector.stop()                # 1. остановить детектор
    wake_event.set()                    # 2. дать команду основному циклу
    threading.Timer(3.0,                # 3. перезапустить через 3 с
                    wake_detector.start
                   ).start()

# --- VAD + MIC ---
async def vad_record_and_send(device=None):
    while True:
        try:
            print(f"[INFO] Connecting to {URI} (микрофон)")
            async with websockets.connect(URI, max_size=8*2**20, 
                                         ping_interval=300, # 5 минут между пингами
                                         ping_timeout=None) as ws:  # отключаем таймаут
                await mic_stream_loop(ws, device)
        except Exception as e:
            print(f"[ERROR] Ошибка соединения: {e} (микрофон)")
            print("[INFO] Повторное подключение через 5 секунд... (микрофон)")
            await asyncio.sleep(5)

async def mic_stream_loop(ws, device=None):
    global audio_queue
    mic_queue = []
    in_speech = False
    speech_frames = 0
    silence_frames = 0
    audio_buffer = []
    processing_speech = False
    waiting_for_wake_word = USE_WAKE_WORD  # Start in wake word mode if enabled
    
    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        mic_queue.append(bytes(indata))
    
    kwargs = {
        'samplerate': SAMPLE_RATE,
        'blocksize': FRAME_SIZE,
        'dtype': 'int16',
        'channels': 1,
        'callback': callback
    }
    if device is not None:
        kwargs['device'] = device
    
    with sd.RawInputStream(**kwargs):
        print("[INFO] Слушаю микрофон...")
        if waiting_for_wake_word:
            print(f"[INFO] Ожидаю пробуждение командой '{WAKEWORD}'...")
        
        while True:
            # Check if wake word was detected
            if waiting_for_wake_word and wake_event.is_set():
                waiting_for_wake_word = False
                wake_event.clear()
                time.sleep(0.2)  # Small delay after wake word
                
                # Clear any buffered audio to start fresh for command
                audio_buffer.clear()
                mic_queue.clear()
                in_speech = False
                speech_frames = 0
                silence_frames = 0
                
                # Set a timer to return to wake word mode after some inactivity
                def reset_to_wake_word():
                    nonlocal waiting_for_wake_word
                    if USE_WAKE_WORD and not in_speech and not processing_speech:
                        waiting_for_wake_word = True
                        print("[INFO] Возвращаюсь в режим ожидания пробуждения...")
                
                # Schedule reset after 15 seconds of inactivity
                timer = threading.Timer(15.0, reset_to_wake_word)
                timer.daemon = True
                timer.start()
                
                print("[INFO] Жду команду...")
            
            # Skip audio processing in wake word mode
            if waiting_for_wake_word:
                # We still need to consume from the mic queue to keep it from growing
                if mic_queue:
                    mic_queue.pop(0)
                time.sleep(0.01)
                continue
            
            if not mic_queue:
                time.sleep(0.01)
                continue
            
            data = mic_queue.pop(0)
            is_speech_frame = vad.is_speech(data, SAMPLE_RATE)
            
            if not in_speech:
                if is_speech_frame:
                    speech_frames += 1
                    audio_buffer.append(data)
                    if speech_frames >= SPEECH_START_THRESHOLD:
                        in_speech = True
                        silence_frames = 0
                        print("[VAD] Речь обнаружена... (микрофон)")
                else:
                    speech_frames = 0
                    audio_buffer.clear()
            else:
                if is_speech_frame:
                    audio_buffer.append(data)
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames < SILENCE_THRESHOLD_FRAMES:
                        audio_buffer.append(data)
                    else:
                        in_speech = False
                        print("[VAD] Конец речи, отправка... (микрофон)")
                        combined_data = b"".join(audio_buffer)
                        audio_buffer.clear()
                        speech_frames = 0
                        silence_frames = 0
                        
                        if not processing_speech and len(combined_data) > 0:
                            processing_speech = True
                            await process_and_send(ws, combined_data)
                            processing_speech = False
                            
                            # After processing speech, return to wake word mode if enabled
                            if USE_WAKE_WORD:
                                waiting_for_wake_word = True
                                print("[INFO] Возвращаюсь в режим ожидания пробуждения...")
            
            time.sleep(0.01)

async def process_and_send(ws, combined_data):
    try:
        # Увеличиваем таймаут ожидания ответа, если используется
        await ws.send(combined_data)
        await ws.send("END")
        
        # Добавляем большой таймаут для операций recv
        response = await asyncio.wait_for(ws.recv(), timeout=3600)  # 1 час таймаут
        
        # Проверяем, начинается ли передача фрагментированного аудио
        if response == "AUDIO_CHUNKS_BEGIN":
            print("[INFO] Получаем фрагментированное аудио...")
            audio_chunks = []
            
            # Собираем все фрагменты аудио
            while True:
                chunk = await ws.recv()
                if isinstance(chunk, str) and chunk == "AUDIO_CHUNKS_END":
                    break
                if isinstance(chunk, bytes):
                    audio_chunks.append(chunk)
                    print(f"[INFO] Получен фрагмент аудио: {len(chunk)} байт")
            
            # Объединяем все фрагменты в одно аудио
            if audio_chunks:
                combined_audio = b"".join(audio_chunks)
                print(f"[INFO] Собрано аудио из {len(audio_chunks)} фрагментов, общий размер: {len(combined_audio)} байт")
                threading.Thread(target=play_audio, args=(combined_audio,)).start()
            else:
                print("[WARNING] Получены пустые фрагменты аудио")
        
        # Обычный ответ (не разбитый на части)
        elif isinstance(response, bytes):
            print(f"[INFO] Получен аудио-ответ: {len(response)} байт")
            threading.Thread(target=play_audio, args=(response,)).start()
        else:
            print(f"[INFO] Получен текстовый ответ: {response}")
    except asyncio.TimeoutError:
        print("[ERROR] Таймаут при ожидании ответа от сервера")
        # Повторное подключение или другая обработка таймаута
    except Exception as e:
        print(f"[ERROR] Ошибка при отправке/получении: {e}")

def run_wake_detector():
    """Run the wake word detector in a separate thread"""
    global wake_detector
    wake_detector = WakeWordDetector(callback=on_wake_word_detected)
    try:
        wake_detector.start()
    except Exception as e:
        print(f"[ERROR] Wake word detector error: {e}")

# --- MAIN ENTRYPOINT ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Микрофонный клиент с поддержкой VAD")
    parser.add_argument("--device", type=int, help="Индекс устройства ввода (см. --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="Показать список доступных устройств")
    parser.add_argument("--no-wake", action="store_true", help="Отключить режим wake word (всегда слушать)")
    args = parser.parse_args()
    
    if args.list_devices:
        print_available_devices()
        sys.exit(0)
    
    global USE_WAKE_WORD
    if args.no_wake:
        USE_WAKE_WORD = False
    
    # Start wake word detection if enabled
    if USE_WAKE_WORD:
        wake_thread = threading.Thread(target=run_wake_detector, daemon=True)
        wake_thread.start()
        print("[INFO] Wake word детектор запущен")
    
    # Start the main microphone listening thread
    mic_thread = threading.Thread(target=lambda: asyncio.run(vad_record_and_send(device=args.device)), daemon=True)
    mic_thread.start()
    
    print("[INFO] Клиент запущен. Для выхода нажмите Ctrl+C")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Клиент остановлен")
        if wake_detector:
            wake_detector.stop()

if __name__ == "__main__":
    main() 