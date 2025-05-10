import os

# MQTT Configuration
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

# MQTT Topics
RECOGNIZED_INTENT_PATH = os.getenv("RECOGNIZED_INTENT_PATH", "hermes/intent")
UNRECOGNIZED_INTENT_PATH = os.getenv("UNRECOGNIZED_INTENT_PATH", "hermes/nlu/intentNotRecognized")
TTS_SAY_PATH = os.getenv("TTS_SAY_PATH", "hermes/tts/say")
