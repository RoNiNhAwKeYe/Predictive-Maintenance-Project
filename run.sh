#!/bin/bash

# Predictive Maintenance Project - Setup Script (macOS/Linux)
# This script sets up the Python environment and runs the server

set -e

cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "Predictive Maintenance - Setup & Run"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

echo "[1/4] Python found:"
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "[2/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[2/4] Virtual environment already exists"
fi

# Activate virtual environment
echo "[3/4] Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "[4/4] Installing dependencies..."
pip install -r backend/requirements.txt

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "Starting Flask server..."
echo ""
echo "The dashboard will be available at:"
echo "  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "============================================================"
echo ""

# Start the Flask server
cd backend
python app.py
