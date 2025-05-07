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

vad = webrtcvad.Vad()
vad.set_mode(3)

# --- UTILS ---
def print_available_devices():
    print("\nДоступные устройства ввода звука:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"Индекс {i}: {dev['name']} (входы: {dev['max_input_channels']}, выходы: {dev['max_output_channels']})")
    print("\nДля выбора устройства при запуске используйте параметр --device <индекс>")
    print("Для прослушивания системного звука используйте --loopback (требуется PyAudio с поддержкой as_loopback)")

def play_audio(audio_data):
    import tempfile
    temp_ogg = tempfile.NamedTemporaryFile(suffix='.ogg', delete=False)
    try:
        temp_ogg.write(audio_data)
        temp_ogg.close()
        data, samplerate = sf.read(temp_ogg.name, dtype='float32')
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"[ERROR] Не удалось воспроизвести ответ: {e}")
    finally:
        try:
            if os.path.exists(temp_ogg.name):
                os.remove(temp_ogg.name)
        except Exception:
            pass

def audio_callback(indata, frames, time_info, status, audio_queue):
    if status:
        print(f"Status: {status}")
    audio_queue.append(bytes(indata))

# --- VAD + MIC ---
async def vad_record_and_send(device=None):
    while True:
        try:
            print(f"[INFO] Connecting to {URI} (микрофон)")
            async with websockets.connect(URI, max_size=2**20, ping_interval=30, ping_timeout=30) as ws:
                await mic_stream_loop(ws, device)
        except Exception as e:
            print(f"[ERROR] Ошибка соединения: {e} (микрофон)")
            print("[INFO] Повторное подключение через 5 секунд... (микрофон)")
            await asyncio.sleep(5)

async def mic_stream_loop(ws, device=None):
    audio_queue = []
    in_speech = False
    speech_frames = 0
    silence_frames = 0
    audio_buffer = []
    processing_speech = False
    def callback(indata, frames, time_info, status):
        audio_callback(indata, frames, time_info, status, audio_queue)
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
        while True:
            if not audio_queue:
                time.sleep(0.01)
                continue
            data = audio_queue.pop(0)
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
                            await process_and_send(ws, combined_data, lambda: processing_speech)
                            processing_speech = False
            time.sleep(0.01)

async def process_and_send(ws, combined_data, get_processing_flag):
    try:
        await ws.send(combined_data)
        await ws.send("END")
        response = await ws.recv()
        if isinstance(response, bytes):
            threading.Thread(target=play_audio, args=(response,)).start()
        else:
            print(f"[INFO] Получен текстовый ответ: {response}")
    except Exception as e:
        print(f"[ERROR] Ошибка при отправке/получении: {e}")

# --- LOOPBACK ---
def loopback_record_thread():
    import pyaudio
    p = pyaudio.PyAudio()
    output_device_index = None
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxOutputChannels'] > 0:
            output_device_index = i
            print(f"[INFO] Loopback: выбрано устройство вывода: [{i}] {dev['name']}")
            break
    if output_device_index is None:
        print("[ERROR] Loopback: не найдено устройств вывода для loopback")
        p.terminate()
        return
    FORMAT = pyaudio.paInt16
    RATE = SAMPLE_RATE
    CHUNK = FRAME_SIZE
    CHANNELS = 2
    vad_local = webrtcvad.Vad()
    vad_local.set_mode(3)
    def process_loopback():
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        loop.run_until_complete(loopback_ws_loop(p, output_device_index, FORMAT, CHANNELS, RATE, CHUNK, vad_local))
    threading.Thread(target=process_loopback, daemon=True).start()

def get_loopback_data(stream, vad_local, ws, RATE, CHUNK):
    in_speech = False
    speech_frames = 0
    silence_frames = 0
    audio_buffer = []
    processing_speech = False
    while stream.is_active():
        data = stream.read(CHUNK, exception_on_overflow=False)
        mono_data = np.frombuffer(data, dtype=np.int16)[::2].tobytes()
        is_speech_frame = vad_local.is_speech(mono_data, RATE)
        if not in_speech:
            if is_speech_frame:
                speech_frames += 1
                audio_buffer.append(data)
                if speech_frames >= SPEECH_START_THRESHOLD:
                    in_speech = True
                    silence_frames = 0
                    print("[VAD] Речь обнаружена... (loopback)")
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
                    print("[VAD] Конец речи, отправка... (loopback)")
                    combined_data = b"".join(audio_buffer)
                    audio_buffer.clear()
                    speech_frames = 0
                    silence_frames = 0
                    if not processing_speech and len(combined_data) > 0:
                        processing_speech = True
                        asyncio.run(process_and_send_loopback(ws, combined_data, lambda: processing_speech))
                        processing_speech = False
        time.sleep(0.01)

async def process_and_send_loopback(ws, combined_data, get_processing_flag):
    try:
        await ws.send(combined_data)
        await ws.send("END")
        response = await ws.recv()
        if isinstance(response, bytes):
            threading.Thread(target=play_audio, args=(response,)).start()
        else:
            print(f"[INFO] Получен текстовый ответ: {response}")
    except Exception as e:
        print(f"[ERROR] Ошибка при отправке/получении: {e}")

def loopback_record_main(p, output_device_index, FORMAT, CHANNELS, RATE, CHUNK, vad_local):
    import websockets
    async def loopback_ws_loop(p, output_device_index, FORMAT, CHANNELS, RATE, CHUNK, vad_local):
        while True:
            try:
                print(f"[INFO] Connecting to {URI} (loopback)")
                async with websockets.connect(URI, max_size=2**20, ping_interval=30, ping_timeout=30) as ws:
                    stream = p.open(
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=output_device_index,
                        as_loopback=True
                    )
                    print("[INFO] Слушаю системный звук (loopback)...")
                    get_loopback_data(stream, vad_local, ws, RATE, CHUNK)
                    stream.stop_stream()
                    stream.close()
            except Exception as e:
                print(f"[ERROR] Ошибка соединения: {e} (loopback)")
                print("[INFO] Повторное подключение через 5 секунд... (loopback)")
                await asyncio.sleep(5)
    return loopback_ws_loop(p, output_device_index, FORMAT, CHANNELS, RATE, CHUNK, vad_local)

# --- MAIN ENTRYPOINT ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Микрофонный клиент с поддержкой VAD и loopback")
    parser.add_argument("--device", type=int, help="Индекс устройства ввода (см. --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="Показать список доступных устройств")
    parser.add_argument("--loopback", action="store_true", help="Прослушивать системный звук (требуется PyAudio с as_loopback)")
    args = parser.parse_args()
    if args.list_devices:
        print_available_devices()
        sys.exit(0)
    threads = []
    t1 = threading.Thread(target=lambda: asyncio.run(vad_record_and_send(device=args.device)), daemon=True)
    t1.start()
    threads.append(t1)
    if args.loopback:
        loopback_record_thread()
    print("[INFO] Клиент запущен. Для выхода нажмите Ctrl+C")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Клиент остановлен")

if __name__ == "__main__":
    main() 