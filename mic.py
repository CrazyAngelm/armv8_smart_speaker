#!/usr/bin/env python3
"""
Simple microphone listening script with web interface for ARMv8 (Debian).
Designed for Orange Pi 5 Pro and similar ARMv8 boards.
"""
import asyncio
import websockets
import numpy as np
import sounddevice as sd
import argparse
from aiohttp import web

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
HOST = "0.0.0.0"  # Listen on all interfaces
WEB_PORT = 8080
WS_PORT = 8765

# Global state
audio_buffer = []
clients = set()
is_recording = False

# Web server routes
async def index_handler(request):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ARMv8 Smart Speaker</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            button { padding: 12px 20px; margin: 10px 0; background-color: #4CAF50; color: white; 
                     border: none; cursor: pointer; border-radius: 4px; font-size: 16px; }
            button:hover { background-color: #45a049; }
            button:disabled { background-color: #cccccc; }
            #status { margin-top: 20px; padding: 10px; border-radius: 4px; }
            .recording { background-color: #ffcccc; }
            .stopped { background-color: #ccffcc; }
        </style>
    </head>
    <body>
        <h1>ARMv8 Smart Speaker</h1>
        <button id="startBtn">Start Listening</button>
        <button id="stopBtn" disabled>Stop Listening</button>
        <div id="status" class="stopped">Status: Not recording</div>
        
        <script>
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            
            startBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch('/start', {method: 'POST'});
                    const result = await response.json();
                    if (result.success) {
                        status.textContent = 'Status: Recording';
                        status.className = 'recording';
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                    } else {
                        alert('Failed to start recording: ' + result.error);
                    }
                } catch (error) {
                    alert('Error: ' + error);
                }
            });
            
            stopBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch('/stop', {method: 'POST'});
                    const result = await response.json();
                    if (result.success) {
                        status.textContent = 'Status: Not recording';
                        status.className = 'stopped';
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    } else {
                        alert('Failed to stop recording: ' + result.error);
                    }
                } catch (error) {
                    alert('Error: ' + error);
                }
            });
        </script>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')

async def start_recording_handler(request):
    global is_recording
    if not is_recording:
        is_recording = True
        print("[INFO] Recording started")
        return web.json_response({"success": True})
    return web.json_response({"success": False, "error": "Already recording"})

async def stop_recording_handler(request):
    global is_recording
    if is_recording:
        is_recording = False
        print("[INFO] Recording stopped")
        return web.json_response({"success": True})
    return web.json_response({"success": False, "error": "Not recording"})

# Audio callback function
def audio_callback(indata, frames, time, status):
    if status:
        print(f"[WARNING] {status}")
    if is_recording:
        audio_data = indata.copy()
        audio_buffer.append(audio_data)
        if len(audio_buffer) > 10:  # Keep buffer size reasonable
            asyncio.run_coroutine_threadsafe(broadcast_audio(audio_buffer.pop(0)), asyncio.get_event_loop())

# WebSocket functions
async def ws_handler(websocket, path):
    clients.add(websocket)
    print(f"[INFO] Client connected: {websocket.remote_address}")
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)
        print(f"[INFO] Client disconnected: {websocket.remote_address}")

async def broadcast_audio(audio_data):
    if not clients:
        return
    
    # Convert to bytes for transmission
    data_bytes = audio_data.tobytes()
    
    # Send to all connected clients
    disconnected = set()
    for client in clients:
        try:
            await client.send(data_bytes)
        except websockets.exceptions.ConnectionClosed:
            disconnected.add(client)
    
    # Remove disconnected clients
    for client in disconnected:
        clients.remove(client)

# Setup and run main tasks
async def start_microphone(device=None):
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=audio_callback,
            blocksize=CHUNK,
            dtype='float32',
            device=device
        )
        with stream:
            print(f"[INFO] Microphone stream started on device {device if device else 'default'}")
            while True:
                await asyncio.sleep(1)
    except Exception as e:
        print(f"[ERROR] Error with microphone: {e}")
        print("[INFO] Available audio devices:")
        print(sd.query_devices())
        raise

def list_audio_devices():
    """Print available audio input devices."""
    print("\nAvailable audio input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"Index {i}: {dev['name']} (inputs: {dev['max_input_channels']})")

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simple microphone server for ARMv8")
    parser.add_argument("--device", type=int, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices")
    parser.add_argument("--web-port", type=int, default=WEB_PORT, help="Web server port")
    parser.add_argument("--ws-port", type=int, default=WS_PORT, help="WebSocket server port")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # Setup web server
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_post('/start', start_recording_handler)
    app.router.add_post('/stop', stop_recording_handler)
    
    # Start web server in a separate task
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, args.web_port)
    await site.start()
    print(f"[INFO] Web interface available at http://{HOST}:{args.web_port}")
    
    # Start WebSocket server in a separate task
    ws_server = await websockets.serve(ws_handler, HOST, args.ws_port)
    print(f"[INFO] WebSocket server running at ws://{HOST}:{args.ws_port}")
    
    # Start microphone processing
    mic_task = asyncio.create_task(start_microphone(args.device))
    
    # Keep everything running
    try:
        await asyncio.gather(mic_task)
    finally:
        await runner.cleanup()
        ws_server.close()
        await ws_server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main()) 