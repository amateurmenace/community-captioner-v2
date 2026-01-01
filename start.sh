#!/bin/bash
# Community Captioner - Start Script
cd "$(dirname "$0")"

# Check Python
if command -v python3 &> /dev/null; then
    python3 start-server.py
elif command -v python &> /dev/null; then
    python start-server.py
else
    echo "Python not found. Please install Python 3."
    exit 1
fi
