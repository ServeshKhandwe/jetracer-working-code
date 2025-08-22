#!/bin/bash

# Setup script for OpenAI Mocap Controller

echo "Setting up OpenAI Mocap Controller environment..."

# Install Python dependencies
# echo "Installing Python dependencies..."
# pip3 install openai websockets numpy matplotlib

# Make scripts executable
chmod +x src/openai_mocap_controller/scripts/openai_mocap_controller.py
chmod +x src/openai_mocap_controller/scripts/performance_analyzer.py
chmod +x src/openai_mocap_controller/run_controller.sh

# Create log directory
mkdir -p /tmp/openai_mocap_logs

echo "✓ Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'"
echo "2. Run the controller: ./src/openai_mocap_controller/run_controller.sh"
echo "3. Analyze results: python3 src/openai_mocap_controller/scripts/performance_analyzer.py"