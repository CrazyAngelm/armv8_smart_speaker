#!/usr/bin/env python3
"""
Main controller script for ARMv8 Smart Speaker.
Orchestrates microphone input and speech-to-text services.
Designed for Orange Pi 5 Pro and similar ARMv8 boards running Debian.
"""
import argparse
import subprocess
import sys
import os
import time
import signal
import threading
import webbrowser
import asyncio
import websockets
import json

# Default configuration
HOST = "0.0.0.0"  # Listen on all interfaces
MIC_WEB_PORT = 8080
MIC_WS_PORT = 8765
STT_WEB_PORT = 8081
STT_WS_PORT = 8778
VOSK_MODEL_PATH = os.path.join("models", "vosk-model-small-ru-0.22")

# Global variables for process management
mic_process = None
stt_process = None
should_exit = False
local_ip = None

def get_local_ip():
    """Get the local IP address of this machine"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def check_vosk_model():
    """Check if the Vosk model exists"""
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"[ERROR] Vosk model not found at: {VOSK_MODEL_PATH}")
        print("Please download the model or specify correct path with --model")
        return False
    return True

def start_mic_service(device=None, web_port=MIC_WEB_PORT, ws_port=MIC_WS_PORT):
    """Start the microphone service"""
    global mic_process
    
    cmd = [sys.executable, "mic.py"]
    if device is not None:
        cmd.extend(["--device", str(device)])
    cmd.extend(["--web-port", str(web_port), "--ws-port", str(ws_port)])
    
    print(f"[INFO] Starting microphone service: {' '.join(cmd)}")
    mic_process = subprocess.Popen(cmd)
    return mic_process.poll() is None  # Check if process started

def start_stt_service(model_path=VOSK_MODEL_PATH, mic_uri=None, web_port=STT_WEB_PORT, stt_port=STT_WS_PORT):
    """Start the STT service"""
    global stt_process
    
    cmd = [sys.executable, "stt.py"]
    cmd.extend(["--model", model_path])
    
    if mic_uri:
        cmd.extend(["--mic-uri", mic_uri])
    else:
        mic_uri = f"ws://localhost:{MIC_WS_PORT}"
        cmd.extend(["--mic-uri", mic_uri])
    
    cmd.extend(["--web-port", str(web_port), "--stt-port", str(stt_port)])
    
    print(f"[INFO] Starting STT service: {' '.join(cmd)}")
    stt_process = subprocess.Popen(cmd)
    return stt_process.poll() is None  # Check if process started

def stop_services():
    """Stop all running services"""
    global mic_process, stt_process
    
    print("[INFO] Stopping services...")
    
    if stt_process and stt_process.poll() is None:
        print("[INFO] Stopping STT service...")
        stt_process.terminate()
        try:
            stt_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stt_process.kill()
    
    if mic_process and mic_process.poll() is None:
        print("[INFO] Stopping microphone service...")
        mic_process.terminate()
        try:
            mic_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mic_process.kill()
    
    print("[INFO] All services stopped")

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    global should_exit
    print("\n[INFO] Interrupt received, shutting down...")
    should_exit = True
    stop_services()
    sys.exit(0)

def monitor_processes():
    """Monitor subprocess health"""
    global should_exit, mic_process, stt_process
    
    while not should_exit:
        # Check microphone process
        if mic_process and mic_process.poll() is not None:
            print("[WARNING] Microphone service stopped unexpectedly, restarting...")
            start_mic_service()
        
        # Check STT process
        if stt_process and stt_process.poll() is not None:
            print("[WARNING] STT service stopped unexpectedly, restarting...")
            mic_uri = f"ws://localhost:{MIC_WS_PORT}"
            start_stt_service(mic_uri=mic_uri)
        
        time.sleep(5)  # Check every 5 seconds

def open_web_interfaces():
    """Open web interfaces in browser"""
    global local_ip
    
    if not local_ip:
        local_ip = get_local_ip()
    
    mic_url = f"http://{local_ip}:{MIC_WEB_PORT}"
    stt_url = f"http://{local_ip}:{STT_WEB_PORT}"
    
    print(f"[INFO] Opening web interfaces in browser:")
    print(f"  Microphone: {mic_url}")
    print(f"  STT: {stt_url}")
    
    try:
        webbrowser.open(mic_url)
        time.sleep(1)  # Small delay between opening browsers
        webbrowser.open(stt_url)
    except Exception as e:
        print(f"[WARNING] Couldn't open browser: {e}")
        print(f"[INFO] Please manually open: {mic_url} and {stt_url}")

async def status_server(host=HOST, port=8082):
    """Run a simple status server to monitor services"""
    
    async def handler(websocket, path):
        """WebSocket handler for status updates"""
        try:
            while True:
                # Prepare status info
                status = {
                    "mic_running": mic_process and mic_process.poll() is None,
                    "stt_running": stt_process and stt_process.poll() is None,
                    "mic_web_port": MIC_WEB_PORT,
                    "mic_ws_port": MIC_WS_PORT,
                    "stt_web_port": STT_WEB_PORT,
                    "stt_ws_port": STT_WS_PORT,
                    "vosk_model": VOSK_MODEL_PATH
                }
                
                await websocket.send(json.dumps(status))
                await asyncio.sleep(2)  # Update every 2 seconds
        except websockets.exceptions.ConnectionClosed:
            pass
    
    async def http_handler(request):
        """Simple HTTP handler for status page"""
        status_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ARMv8 Smart Speaker Control</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .card {{ box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); border-radius: 5px; 
                         padding: 20px; margin-bottom: 20px; background-color: #f9f9f9; }}
                .running {{ color: green; }}
                .stopped {{ color: red; }}
                button {{ padding: 10px; margin: 5px; cursor: pointer; }}
            </style>
        </head>
        <body>
            <h1>ARMv8 Smart Speaker Control</h1>
            
            <div class="card">
                <h2>Services Status</h2>
                <p>Microphone Service: <span id="mic-status">Checking...</span></p>
                <p>STT Service: <span id="stt-status">Checking...</span></p>
            </div>
            
            <div class="card">
                <h2>Web Interfaces</h2>
                <p><a href="http://{local_ip}:{MIC_WEB_PORT}" target="_blank">Microphone Interface</a></p>
                <p><a href="http://{local_ip}:{STT_WEB_PORT}" target="_blank">STT Interface</a></p>
            </div>
            
            <script>
                const ws = new WebSocket('ws://' + window.location.hostname + ':{port}/ws');
                
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    
                    // Update microphone status
                    const micStatus = document.getElementById('mic-status');
                    if (data.mic_running) {{
                        micStatus.textContent = 'Running';
                        micStatus.className = 'running';
                    }} else {{
                        micStatus.textContent = 'Stopped';
                        micStatus.className = 'stopped';
                    }}
                    
                    // Update STT status
                    const sttStatus = document.getElementById('stt-status');
                    if (data.stt_running) {{
                        sttStatus.textContent = 'Running';
                        sttStatus.className = 'running';
                    }} else {{
                        sttStatus.textContent = 'Stopped';
                        sttStatus.className = 'stopped';
                    }}
                }};
                
                ws.onclose = function() {{
                    document.getElementById('mic-status').textContent = 'Unknown (Connection Lost)';
                    document.getElementById('stt-status').textContent = 'Unknown (Connection Lost)';
                }};
            </script>
        </body>
        </html>
        """
        return web.Response(text=status_html, content_type='text/html')
    
    # Set up web server
    from aiohttp import web
    app = web.Application()
    app.router.add_get('/', http_handler)
    app.router.add_get('/ws', lambda request: web.WebSocketResponse())
    
    # Create a server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    
    # Start the server
    await site.start()
    print(f"[INFO] Status page available at http://{host}:{port}")
    
    # Start WebSocket server
    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # Run forever

def main():
    """Main entry point"""
    global local_ip, MIC_WEB_PORT, MIC_WS_PORT, STT_WEB_PORT, STT_WS_PORT, VOSK_MODEL_PATH
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="ARMv8 Smart Speaker Controller")
    parser.add_argument("--device", type=int, help="Audio input device index")
    parser.add_argument("--mic-web-port", type=int, default=MIC_WEB_PORT, help="Microphone web interface port")
    parser.add_argument("--mic-ws-port", type=int, default=MIC_WS_PORT, help="Microphone WebSocket port")
    parser.add_argument("--stt-web-port", type=int, default=STT_WEB_PORT, help="STT web interface port")
    parser.add_argument("--stt-ws-port", type=int, default=STT_WS_PORT, help="STT WebSocket port")
    parser.add_argument("--model", type=str, default=VOSK_MODEL_PATH, help="Path to Vosk model")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--no-browser", action="store_true", help="Don't open web interfaces in browser")
    parser.add_argument("--status-port", type=int, default=8082, help="Status web interface port")
    
    args = parser.parse_args()
    
    # Update configuration from arguments
    MIC_WEB_PORT = args.mic_web_port
    MIC_WS_PORT = args.mic_ws_port
    STT_WEB_PORT = args.stt_web_port
    STT_WS_PORT = args.stt_ws_port
    VOSK_MODEL_PATH = args.model
    
    # Get local IP
    local_ip = get_local_ip()
    print(f"[INFO] Local IP: {local_ip}")
    
    # Check for list devices option
    if args.list_devices:
        print("[INFO] Listing audio devices...")
        subprocess.run([sys.executable, "mic.py", "--list-devices"])
        return
    
    # Check Vosk model
    if not check_vosk_model():
        return
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start microphone service
    if not start_mic_service(device=args.device, web_port=MIC_WEB_PORT, ws_port=MIC_WS_PORT):
        print("[ERROR] Failed to start microphone service")
        return
    
    # Give a little time for mic service to start
    time.sleep(2)
    
    # Start STT service
    mic_uri = f"ws://localhost:{MIC_WS_PORT}"
    if not start_stt_service(model_path=VOSK_MODEL_PATH, mic_uri=mic_uri, 
                            web_port=STT_WEB_PORT, stt_port=STT_WS_PORT):
        print("[ERROR] Failed to start STT service")
        stop_services()
        return
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_processes, daemon=True)
    monitor_thread.start()
    
    # Wait a moment for services to start
    time.sleep(3)
    
    # Open web interfaces if requested
    if not args.no_browser:
        open_web_interfaces()
    
    # Print service info
    print("\n[INFO] Services running:")
    print(f"  Microphone web interface: http://{local_ip}:{MIC_WEB_PORT}")
    print(f"  Microphone WebSocket: ws://{local_ip}:{MIC_WS_PORT}")
    print(f"  STT web interface: http://{local_ip}:{STT_WEB_PORT}")
    print(f"  STT WebSocket: ws://{local_ip}:{STT_WS_PORT}")
    print(f"  Status web interface: http://{local_ip}:{args.status_port}")
    print("\n[INFO] Press Ctrl+C to exit")
    
    # Run status server
    try:
        asyncio.run(status_server(host=HOST, port=args.status_port))
    except KeyboardInterrupt:
        pass
    finally:
        stop_services()

if __name__ == "__main__":
    main() 