#!/bin/bash

echo "========================================================================"
echo "           HINGLISH TEXT ANALYZER - QUICK START"
echo "========================================================================"
echo ""

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found! Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $PYTHON_CMD"
echo ""

# Check if lid.176.bin exists
if [ ! -f "lid.176.bin" ]; then
    echo "❌ FastText model (lid.176.bin) not found!"
    echo "📥 Downloading... (126 MB)"
    curl -o lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    echo "✓ Download complete!"
    echo ""
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
$PYTHON_CMD -c "import fasttext, transformers, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Dependencies not installed. Installing now..."
    echo ""
    pip install -r requirements.txt
    echo ""
    echo "✓ Dependencies installed!"
else
    echo "✓ Dependencies already installed!"
fi

echo ""
echo "========================================================================"
echo "🚀 Starting Hinglish Analyzer GUI..."
echo "========================================================================"
echo ""

$PYTHON_CMD hinglish_analyzer_gui.py
