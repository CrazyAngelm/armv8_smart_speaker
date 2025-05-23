import os
import time
from pocketsphinx import LiveSpeech, get_model_path
from dotenv import load_dotenv
import sounddevice as sd

load_dotenv()

# Wake word configuration
KEYPHRASE = os.getenv("WAKEWORD", "okey")
KWS_THRESHOLD = float(os.getenv("KWS_THRESHOLD", "1e-20"))

# Model directory
MODEL_DIR = os.getenv("PS_MODEL_DIR", get_model_path())

class WakeWordDetector:
    def __init__(self, callback=None):
        self.callback = callback
        self.running = False
        self.device_index = self.get_input_device_index(os.getenv("WAKEWORD_DEVICE_HINT"))
        self.initialize_speech()
        self.min_interval = 2.0      # минимальный интервал между срабатываниями, сек
        self._last_ts = 0.0          # время последнего триггера
        
    def get_input_device_index(self, name_hint=None):
        """Автоматически выбирает индекс устройства ввода по имени, затем первое с входом, затем pulse/default"""
        try:
            devices = sd.query_devices()
            # 1. Поиск по имени
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0 and name_hint and name_hint.lower() in dev['name'].lower():
                    print(f"[WAKE] Используется устройство ввода по имени: {dev['name']} (index={idx})")
                    return idx
            # 2. Первый input-устройство
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    print(f"[WAKE] Используется первое доступное устройство ввода: {dev['name']} (index={idx})")
                    return idx
            # 3. Явно пробуем pulse или default
            for idx, dev in enumerate(devices):
                if dev['name'].lower() in ['pulse', 'default']:
                    print(f"[WAKE] Используется fallback устройство: {dev['name']} (index={idx})")
                    return idx
            print("[WAKE] Не найдено ни одного устройства ввода!")
            return None
        except Exception as e:
            print(f"[WAKE] Ошибка при поиске устройств ввода: {e}")
            return None
    
    def initialize_speech(self):
        """Initialize the LiveSpeech object for wake word detection"""
        try:
            self.speech = LiveSpeech(
                lm=False,
                keyphrase=KEYPHRASE,
                kws_threshold=KWS_THRESHOLD,
                hmm=os.path.join(MODEL_DIR, "en-us") if "en-us" in os.listdir(MODEL_DIR) else MODEL_DIR,
                dic=os.path.join(MODEL_DIR, "cmudict-en-us.dict"),
                #device=self.device_index
            )
            print(f"[WAKE] Initialized PocketSphinx wake word detector for '{KEYPHRASE}'")
        except Exception as e:
            print(f"[ERROR] Failed to initialize PocketSphinx: {e}")
            raise
    
    def start(self):
        """Start wake word detection in a loop"""
        # Reinitialize the speech object to ensure a fresh audio stream
        self.initialize_speech()
        
        self.running = True
        print(f"[WAKE] Listening for wake word: '{KEYPHRASE}'")
        try:
            for phrase in self.speech:
                if not self.running:
                    break
                detected_text = str(phrase)
                print(f"[WAKE] Detected: '{detected_text}'")
                
                now = time.time()
                if detected_text and now - self._last_ts >= self.min_interval:
                    self._last_ts = now
                    if self.callback:
                        self.callback(detected_text)
                        time.sleep(0.5)  # Brief pause after detection
        except Exception as e:
            print(f"[ERROR] Wake word detection error: {e}")
            self.running = False
    
    def stop(self):
        """Stop wake word detection"""
        self.running = False

# For testing the module directly
if __name__ == "__main__":
    def on_wake_word(text):
        print(f"[TEST] Wake word callback with: {text}")
    
    detector = WakeWordDetector(callback=on_wake_word)
    try:
        detector.start()
    except KeyboardInterrupt:
        print("\n[WAKE] Stopping wake word detector")
        detector.stop() 