# Industrial Smart Speaker Assistant for ARMv8

This project is a voice assistant for industrial environments, using Vosk for speech recognition, Yandex TTS for voice synthesis, and LLM (Language Model) for response generation.

## Requirements

- Python 3.8+
- ARMv8 architecture (Orange Pi 5 Pro) running Debian
- Microphone for audio input

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

```bash
# On Linux/macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file based on the template below:

```
# STT (Speech-to-Text) Settings
STT_WS_HOST=0.0.0.0
STT_WS_PORT=8778
VOSK_MODEL_PATH=models/vosk-model-small-ru-0.22

# TTS (Text-to-Speech) Settings
TTS_WS_HOST=0.0.0.0
TTS_WS_PORT=8777
YANDEX_FOLDER_ID=your_folder_id_here
YANDEX_IAM_TOKEN=your_iam_token_here
YANDEX_TTS_VOICE=alena

# Agent Settings
MAGUS_WS_HOST=0.0.0.0
MAGUS_WS_PORT=8765

# LLM Settings
LLM_PROVIDER=deepseek
LLM_MODEL=
DEEPSEEK_MODEL=deepseek-chat
CLAUDE_MODEL=claude-3-haiku-20240307
LOCAL_MODEL=gemma3:12b
LLM_TEMPERATURE=0.3
```

5. Download the Vosk model:

```bash
mkdir -p models
cd models
# Download a Russian model for Vosk
wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip vosk-model-small-ru-0.22.zip
cd ..
```

## Usage

### Running in Full Mode

To start the system with voice recognition and synthesis:

```bash
python main.py
```

This will start:
- Vosk STT (Speech-to-Text) service
- Yandex TTS (Text-to-Speech) service
- Agent service
- Microphone client

### CLI Mode

For testing without voice (text-only mode):

```bash
python main.py --cli
```

### Microphone Options

To list available audio devices:

```bash
python mic_client.py --list-devices
```

To use a specific audio device:

```bash
python mic_client.py --device <device_index>
```

To listen to system audio instead of microphone:

```bash
python mic_client.py --loopback
```

## Architecture

The system consists of several components that communicate via WebSockets:

1. **STT Service (vosk_stt.py)**: Converts audio to text using Vosk
2. **TTS Service (yandex_tts.py)**: Converts text to audio using Yandex TTS API
3. **Agent (agent.py)**: Main component that processes text input, generates responses using LLM
4. **Microphone Client (mic_client.py)**: Captures audio from microphone and sends it to the agent

## License

This project is for educational and industrial purposes. 