#!/bin/bash
# AllTalk TTS - Start Script for macOS Apple Silicon

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if already running
if lsof -ti:7851 > /dev/null 2>&1; then
    echo "AllTalk is already running on port 7851"
    echo "Use ./stop_alltalk.sh to stop it first"
    exit 1
fi

# Get local IP address
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "127.0.0.1")

# Read ports from config (with defaults)
API_PORT=$(python3 -c "import json; print(json.load(open('confignew.json')).get('api_def',{}).get('api_port_number', 7851))" 2>/dev/null || echo "7851")
GRADIO_PORT=$(python3 -c "import json; print(json.load(open('confignew.json')).get('gradio_port_number', 7852))" 2>/dev/null || echo "7852")

# Activate conda environment and start server
echo "========================================"
echo "  AllTalk TTS - Starting Server"
echo "========================================"
echo ""
echo "  Local IP:     $LOCAL_IP"
echo "  API Port:     $API_PORT"
echo "  Gradio Port:  $GRADIO_PORT"
echo ""
echo "  API URL:      http://$LOCAL_IP:$API_PORT"
echo "  Web UI:       http://$LOCAL_IP:$GRADIO_PORT"
echo "  Localhost:    http://127.0.0.1:$GRADIO_PORT"
echo ""
echo "========================================"
echo ""

source ./alltalk_environment/conda/bin/activate ./env

# Open Firefox in private window after server starts (in background)
(sleep 5 && open -a Firefox --args -private-window "http://127.0.0.1:$API_PORT") &

# Start the server (script.py launches tts_server.py as subprocess + Gradio)
python script.py "$@"
