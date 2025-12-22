#!/bin/bash
# AllTalk TTS - Stop Script for macOS Apple Silicon

echo "Stopping AllTalk TTS..."

# Find and kill process on port 7851
PID=$(lsof -ti:7851 2>/dev/null)

if [ -z "$PID" ]; then
    echo "AllTalk is not running"
    exit 0
fi

kill -9 $PID 2>/dev/null
sleep 1

# Verify it stopped
if lsof -ti:7851 > /dev/null 2>&1; then
    echo "Failed to stop AllTalk"
    exit 1
else
    echo "AllTalk stopped successfully"
fi
