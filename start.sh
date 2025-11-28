#!/bin/bash

echo "🧠 Starting Concept Explorer..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Find an available port starting from 5059
PORT=5059
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; do
    echo "Port $PORT is in use, trying next port..."
    PORT=$((PORT+1))
done

echo "Starting server on port $PORT"
echo "Open your browser to: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python concept_explorer.py --port $PORT
