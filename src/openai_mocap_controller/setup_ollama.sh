#!/bin/bash

echo "Setting up Ollama Mocap Controller..."
echo "======================================"

# Make Python scripts executable
echo "1. Making Python scripts executable..."
chmod +x src/openai_mocap_controller/scripts/ollama_mocap_controller.py
chmod +x src/openai_mocap_controller/test_ollama.py

echo "✅ Scripts are now executable"

# Get server configuration from launch file or use defaults
OLLAMA_SERVER_IP=${1:-"192.168.137.54"}
OLLAMA_SERVER_PORT=${2:-11434}

echo ""
echo "2. Checking Ollama server connection..."
echo "   Server: $OLLAMA_SERVER_IP:$OLLAMA_SERVER_PORT"

if [ "$OLLAMA_SERVER_IP" = "localhost" ] || [ "$OLLAMA_SERVER_IP" = "127.0.0.1" ]; then
    echo ""
    echo "🏠 LOCAL SERVER MODE"
    echo "   Checking if Ollama is installed locally..."
    
    if command -v ollama &> /dev/null; then
        echo "✅ Ollama is installed locally"
        
        # Check if local Ollama server is running
        if curl -s http://localhost:$OLLAMA_SERVER_PORT/api/tags &> /dev/null; then
            echo "✅ Local Ollama server is running"
            
            # List available models
            echo ""
            echo "Available models:"
            ollama list
            
        else
            echo "⚠️  Local Ollama server is not running"
            echo "   Start it with: ollama serve"
            echo "   Or with custom port: OLLAMA_HOST=0.0.0.0:$OLLAMA_SERVER_PORT ollama serve"
        fi
        
    else
        echo "❌ Ollama is not installed locally"
        echo ""
        echo "For local installation:"
        echo "  curl -fsSL https://ollama.ai/install.sh | sh"
        echo "  ollama serve"
        echo "  ollama pull llama3.2"
    fi
    
else
    echo ""
    echo "🌐 REMOTE SERVER MODE"
    echo "   Using remote Ollama server at $OLLAMA_SERVER_IP:$OLLAMA_SERVER_PORT"
    echo "   No local Ollama installation required!"
    
    # Test remote connection
    if curl -s http://$OLLAMA_SERVER_IP:$OLLAMA_SERVER_PORT/api/tags &> /dev/null; then
        echo "✅ Remote Ollama server is accessible"
    else
        echo "❌ Cannot connect to remote Ollama server"
        echo ""
        echo "Make sure:"
        echo "  1. Ollama is running on the server: ollama serve"
        echo "  2. Server allows network access: OLLAMA_HOST=0.0.0.0 ollama serve"
        echo "  3. Firewall allows port $OLLAMA_SERVER_PORT"
        echo "  4. Server IP $OLLAMA_SERVER_IP is correct"
    fi
fi

echo ""
echo "3. Testing connection..."
python3 src/openai_mocap_controller/test_ollama.py $OLLAMA_SERVER_IP $OLLAMA_SERVER_PORT

echo ""
echo "Setup complete!"
echo ""
echo "Configuration options:"
echo ""
echo "🏠 For LOCAL Ollama server:"
echo "  roslaunch openai_mocap_controller ollama_mocap_controller.launch"
echo ""
echo "🌐 For your configured Ollama server ($OLLAMA_SERVER_IP):"
echo "  roslaunch openai_mocap_controller ollama_mocap_controller.launch"
echo ""
echo "📊 For comparison with OpenAI:"
echo "  roslaunch openai_mocap_controller openai_mocap_controller.launch"