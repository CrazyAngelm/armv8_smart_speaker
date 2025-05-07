#!/usr/bin/env python3
"""
Speech-to-Text server for ARMv8 (Debian).
Receives audio from mic.py and converts it to text using Vosk.
Designed for Orange Pi 5 Pro and similar ARMv8 boards.
"""
import asyncio
import websockets
import json
import os
import argparse
from vosk import Model, KaldiRecognizer
from aiohttp import web

# Configuration
HOST = "0.0.0.0"  # Listen on all interfaces
WEB_PORT = 8081  # Web interface port
STT_WS_PORT = 8778  # STT WebSocket server port
MIC_WS_URI = "ws://localhost:8765"  # WebSocket URI to connect to mic.py

# Vosk model path
VOSK_MODEL_PATH = os.path.join("models", "vosk-model-small-ru-0.22")

# Global state
clients = set()
text_buffer = []
is_processing = False
mic_conn = None

# Web server routes for status page
async def index_handler(request):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ARMv8 Speech-to-Text</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .result { border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 4px; }
            .card { box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); transition: 0.3s; border-radius: 5px; 
                    padding: 20px; margin-bottom: 20px; background-color: #f9f9f9; }
            h1, h2 { color: #333; }
        </style>
    </head>
    <body>
        <h1>Speech-to-Text Server</h1>
        <div class="card">
            <h2>Status</h2>
            <p>Vosk model: <strong id="model-path">%s</strong></p>
            <p>Connection to mic.py: <strong id="mic-status">%s</strong></p>
        </div>
        <div class="card">
            <h2>Last Recognized Text</h2>
            <div class="result" id="result">
                <p>Waiting for speech...</p>
            </div>
        </div>
        
        <script>
            // WebSocket for real-time updates
            const socket = new WebSocket('ws://' + window.location.hostname + ':%d/ws');
            
            socket.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    if (data.text) {
                        document.getElementById('result').innerHTML = '<p>' + data.text + '</p>';
                    }
                    if (data.mic_status) {
                        document.getElementById('mic-status').innerText = data.mic_status;
                    }
                } catch (e) {
                    console.error('Error parsing message:', e);
                }
            };
            
            socket.onclose = function() {
                document.getElementById('mic-status').innerText = 'Disconnected';
            };
        </script>
    </body>
    </html>
    """ % (VOSK_MODEL_PATH, "Connecting...", STT_WS_PORT)
    return web.Response(text=html, content_type='text/html')

# WebSocket handler for web clients
async def ws_handler(websocket):
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)

# Send updates to web clients
async def update_clients(text=None, mic_status=None):
    if not clients:
        return
    
    update = {}
    if text is not None:
        update["text"] = text
    if mic_status is not None:
        update["mic_status"] = mic_status
    
    if update:
        message = json.dumps(update)
        await asyncio.gather(
            *[client.send_str(message) for client in clients]
        )

# Process audio with Vosk
async def process_audio(audio_data):
    global is_processing, text_buffer
    
    if is_processing:
        return
    
    try:
        is_processing = True
        
        # Check if model exists
        if not os.path.exists(VOSK_MODEL_PATH):
            print(f"[ERROR] Vosk model not found at: {VOSK_MODEL_PATH}")
            await update_clients(text="Error: Vosk model not found", mic_status="Error")
            return
        
        # Load model and create recognizer
        model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(model, 16000)
        
        # Process audio
        recognizer.AcceptWaveform(audio_data)
        result = recognizer.FinalResult()
        
        # Parse result
        result_json = json.loads(result)
        recognized_text = result_json.get("text", "")
        
        if recognized_text:
            print(f"[STT] Recognized: {recognized_text}")
            text_buffer.append(recognized_text)
            # Keep only the last 5 texts
            if len(text_buffer) > 5:
                text_buffer.pop(0)
            await update_clients(text=recognized_text)
        
    except Exception as e:
        print(f"[ERROR] STT processing error: {e}")
        await update_clients(text=f"Error processing audio: {e}")
    finally:
        is_processing = False

# Connect to microphone WebSocket
async def connect_to_mic():
    global mic_conn
    
    while True:
        try:
            print(f"[INFO] Connecting to mic.py at {MIC_WS_URI}")
            async with websockets.connect(MIC_WS_URI, max_size=2**22) as websocket:
                await update_clients(mic_status="Connected")
                mic_conn = websocket
                
                print("[INFO] Connected to mic.py")
                audio_buffer = b""
                
                async for message in websocket:
                    if isinstance(message, bytes):
                        # Process audio data
                        await process_audio(message)
                    else:
                        print(f"[INFO] Received text: {message}")
        
        except Exception as e:
            print(f"[ERROR] Connection to mic.py failed: {e}")
            await update_clients(mic_status="Disconnected")
            mic_conn = None
            await asyncio.sleep(5)  # Retry after 5 seconds

# External STT WebSocket server
async def stt_handler(websocket, path):
    try:
        # Send initial status
        status = {
            "status": "ready",
            "model": VOSK_MODEL_PATH
        }
        await websocket.send(json.dumps(status))
        
        # Process incoming messages
        async for message in websocket:
            if isinstance(message, bytes):
                # Process audio data
                try:
                    # Load model and create recognizer
                    model = Model(VOSK_MODEL_PATH)
                    recognizer = KaldiRecognizer(model, 16000)
                    
                    # Process audio
                    recognizer.AcceptWaveform(message)
                    result = recognizer.FinalResult()
                    
                    # Send back the result
                    await websocket.send(result)
                except Exception as e:
                    await websocket.send(json.dumps({"error": str(e)}))
            else:
                # Handle command
                try:
                    cmd = json.loads(message)
                    if cmd.get("command") == "get_last_texts":
                        await websocket.send(json.dumps({"texts": text_buffer}))
                except:
                    await websocket.send(json.dumps({"error": "Invalid command"}))
    
    except websockets.exceptions.ConnectionClosedError:
        print("[INFO] STT client disconnected")
    except Exception as e:
        print(f"[ERROR] STT WebSocket error: {e}")

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Speech-to-Text server for ARMv8")
    parser.add_argument("--model", type=str, default=VOSK_MODEL_PATH, help="Path to Vosk model")
    parser.add_argument("--mic-uri", type=str, default=MIC_WS_URI, help="URI of mic.py WebSocket")
    parser.add_argument("--web-port", type=int, default=WEB_PORT, help="Web interface port")
    parser.add_argument("--stt-port", type=int, default=STT_WS_PORT, help="STT WebSocket port")
    args = parser.parse_args()
    
    # Update global settings
    global VOSK_MODEL_PATH, MIC_WS_URI
    VOSK_MODEL_PATH = args.model
    MIC_WS_URI = args.mic_uri
    
    print(f"[INFO] Using Vosk model: {VOSK_MODEL_PATH}")
    print(f"[INFO] Connecting to mic.py at: {MIC_WS_URI}")
    
    # Setup web application
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', web.get(ws_handler))
    
    # Start web server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, args.web_port)
    await site.start()
    print(f"[INFO] Web interface running at http://{HOST}:{args.web_port}")
    
    # Start STT WebSocket server
    stt_server = await websockets.serve(stt_handler, HOST, args.stt_port, max_size=2**22)
    print(f"[INFO] STT WebSocket server running at ws://{HOST}:{args.stt_port}")
    
    # Connect to mic.py
    mic_task = asyncio.create_task(connect_to_mic())
    
    # Keep everything running
    try:
        await asyncio.gather(mic_task)
    finally:
        await runner.cleanup()
        stt_server.close()
        await stt_server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main()) 