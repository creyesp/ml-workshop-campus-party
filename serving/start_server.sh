#!/bin/bash

# MLServer Start Script
# This script starts the MLServer with the custom handler for the flood-it-churned model

echo "🚀 Starting MLServer for flood-it-churned model..."
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "model-settings.json" ]; then
    echo "❌ Error: model-settings.json not found!"
    echo "Please run this script from the serving/ directory"
    exit 1
fi

# Check if model file exists
MODEL_PATH="../notebooks/models/xgb_model_full.joblib"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found at $MODEL_PATH"
    echo "Please ensure the model has been trained and saved"
    exit 1
fi

# Check if custom handler exists
if [ ! -f "custom_sklearn_handler.py" ]; then
    echo "❌ Error: custom_sklearn_handler.py not found!"
    echo "Please ensure the custom handler file is in this directory"
    exit 1
fi

# Kill any existing MLServer processes
echo "🔄 Stopping any existing MLServer processes..."
pkill -f mlserver 2>/dev/null || true
sleep 2

# Start MLServer
echo "▶️  Starting MLServer..."
echo "📍 Server will be available at:"
echo "   HTTP API: http://localhost:8080"
echo "   gRPC API: http://localhost:8081"
echo "   Metrics:  http://localhost:8082"
echo ""
echo "📝 Logs will be saved to mlserver.log"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start server with logging
MLSERVER_PARALLEL_WORKERS=0 mlserver start . 2>&1 | tee mlserver.log