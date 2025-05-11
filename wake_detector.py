import os
import time
from pocketsphinx import LiveSpeech, get_model_path
from dotenv import load_dotenv

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
        self.initialize_speech()
        self.min_interval = 2.0      # минимальный интервал между срабатываниями, сек
        self._last_ts = 0.0          # время последнего триггера
        
    def initialize_speech(self):
        """Initialize the LiveSpeech object for wake word detection"""
        try:
            self.speech = LiveSpeech(
                lm=False,
                keyphrase=KEYPHRASE,
                kws_threshold=KWS_THRESHOLD,
                hmm=os.path.join(MODEL_DIR, "en-us") if "en-us" in os.listdir(MODEL_DIR) else MODEL_DIR,
                dic=os.path.join(MODEL_DIR, "cmudict-en-us.dict")
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