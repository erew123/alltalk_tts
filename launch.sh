#!/bin/sh
python modeldownload.py
uvicorn tts_server:app --host 0.0.0.0 --port 7851 --workers 1 --proxy-headers &
sleep 5
python script.py