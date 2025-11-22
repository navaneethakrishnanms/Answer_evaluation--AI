#!/bin/bash

echo "========================================"
echo "  AI Answer Evaluation System"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run setup.sh first to install dependencies."
    echo ""
    exit 1
fi

echo "[1/3] Activating virtual environment..."
source venv/bin/activate

echo "[2/3] Checking GPU availability..."
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

echo ""
echo "[3/3] Starting Flask server..."
echo ""
echo "========================================"
echo "Server will start on: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

python app.py
