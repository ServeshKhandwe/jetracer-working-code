# Ollama Mocap Controller

This is a local LLM version of the OpenAI Mocap Controller that uses **Ollama** to run language models locally instead of using the OpenAI API. This allows for:

- **Privacy**: All processing happens locally, no data sent to external APIs
- **Cost-effective**: No API costs for model inference
- **Customization**: Use any model supported by Ollama
- **Network independence**: Works without internet connection (after model download)

## Deployment Scenarios

### 🏠 Scenario 1: Local Deployment
- **Robot machine**: Runs both ROS controller AND Ollama server
- **Pros**: Simple setup, no network dependencies
- **Cons**: Uses robot's computational resources
- **Best for**: Testing, single robot setups

### 🌐 Scenario 2: Remote Server (Recommended)
- **Robot machine**: Runs only ROS controller (lightweight)
- **Server machine**: Runs Ollama server with powerful hardware
- **Pros**: Dedicated GPU/CPU for AI, multiple robots can share server
- **Cons**: Network dependency
- **Best for**: Production, multiple robots, limited robot hardware

### ☁️ Scenario 3: Hybrid
- **Local fallback**: Basic navigation when server unavailable
- **Remote primary**: Use server when network is good
- **Best for**: Critical applications requiring reliability

## Key Differences from OpenAI Version

| Feature | OpenAI Version | Ollama Version |
|---------|----------------|----------------|
| **LLM Provider** | OpenAI API (cloud) | Ollama (local/remote) |
| **Models** | GPT-3.5, GPT-4, etc. | Llama, Mistral, CodeLlama, etc. |
| **Cost** | Pay per API call | Free after setup |
| **Privacy** | Data sent to OpenAI | Data stays in your network |
| **Internet** | Required | Not required (after setup) |
| **Setup** | API key only | Ollama server + model |
| **Hardware** | None (cloud) | Server needs GPU/CPU for AI |

## Prerequisites

You have **two deployment options**:

### Option 1: Local Ollama (Same Machine)

Install Ollama on the same machine as the robot controller:

```bash
# Install Ollama locally
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

### Option 2: Remote Ollama Server (Recommended)

Use a separate server machine for Ollama (no installation needed on robot):

**On the server machine:**
```bash
# Install Ollama on the server
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama with network access
OLLAMA_HOST=0.0.0.0 ollama serve
```

**On the robot/client machine:**
- No Ollama installation required!
- Just configure the server IP in the launch file

### 2. Download a Model

```bash
# Download a recommended model (choose one)
ollama pull llama3.2          # Fast, good for robotics
ollama pull llama3.2:3b       # Smaller, faster
ollama pull mistral           # Alternative option
ollama pull codellama         # Good for structured output

# List downloaded models
ollama list
```

### 3. Test Connection

```bash
# Test local connection
python3 src/openai_mocap_controller/test_ollama.py

# Test your configured server connection
python3 src/openai_mocap_controller/test_ollama.py

# Test with different parameters if needed
python3 src/openai_mocap_controller/test_ollama.py 192.168.137.54 11434 llama3.2

# Or use the setup script (uses your configured server)
./src/openai_mocap_controller/setup_ollama.sh
```

## Configuration

### Launch File Parameters

Edit `src/openai_mocap_controller/launch/ollama_mocap_controller.launch`:

```xml
<!-- Ollama Server Configuration -->
<param name="ollama_server_ip" value="localhost" />        <!-- Server IP -->
<param name="ollama_server_port" value="11434" />          <!-- Server port -->
<param name="ollama_model" value="llama3.2" />             <!-- Model name -->
<param name="request_timeout" value="10.0" />              <!-- Request timeout -->
```

### Common Configurations

**Local Ollama (if running on robot):**
```xml
<param name="ollama_server_ip" value="localhost" />        <!-- Only if local -->
<param name="ollama_server_port" value="11434" />
```

**Remote Ollama (configured for your setup):**
```xml
<param name="ollama_server_ip" value="192.168.137.54" />   <!-- Your Ollama server -->
<param name="ollama_server_port" value="11434" />
```

**Different models:**
```xml
<param name="ollama_model" value="llama3.2" />      <!-- Fast, balanced -->
<param name="ollama_model" value="llama3.2:3b" />   <!-- Smaller, faster -->
<param name="ollama_model" value="mistral" />       <!-- Alternative -->
<param name="ollama_model" value="codellama" />     <!-- Good for JSON -->
```

## Usage

### 1. Setup and Test

```bash
# For local Ollama server
./src/openai_mocap_controller/setup_ollama.sh

# For remote Ollama server (replace with actual IP)
./src/openai_mocap_controller/setup_ollama.sh 192.168.1.100 11434
```

### 2. Launch Controller

**For local Ollama:**
```bash
# Build workspace first
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# Launch with default settings (localhost)
roslaunch openai_mocap_controller ollama_mocap_controller.launch
```

**For your configured Ollama server (192.168.137.54):**
```bash
# Launch with default configuration (already set to your server)
roslaunch openai_mocap_controller ollama_mocap_controller.launch

# Or explicitly specify (same result)
roslaunch openai_mocap_controller ollama_mocap_controller.launch \
    ollama_server_ip:=192.168.137.54 \
    ollama_model:=llama3.2
```

### 3. Advanced Configuration

```bash
# Custom model and settings (server IP already configured)
roslaunch openai_mocap_controller ollama_mocap_controller.launch \
    ollama_model:=mistral \
    decision_frequency:=2.0 \
    request_timeout:=15.0
```

## Model Recommendations

### For Robot Navigation

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **llama3.2:3b** | 2GB | Fast | Good | Real-time navigation |
| **llama3.2** | 4.7GB | Medium | Better | Balanced performance |
| **mistral** | 4.1GB | Medium | Good | Alternative option |
| **codellama** | 3.8GB | Medium | Excellent | Structured JSON output |

### Performance Tips

- **Faster inference**: Use smaller models (3b versions)
- **Better reasoning**: Use larger models (7b+ versions)
- **JSON reliability**: CodeLlama is excellent for structured output
- **Memory usage**: Check available RAM before choosing large models

## Troubleshooting

### Common Issues

**1. "Connection refused" error:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

**2. "Model not found" error:**
```bash
# List available models
ollama list

# Download the model
ollama pull llama3.2
```

**3. Slow response times:**
```bash
# Use a smaller model
ollama pull llama3.2:3b

# Update launch file
<param name="ollama_model" value="llama3.2:3b" />
```

**4. Network access issues:**
```bash
# Start Ollama with network access
OLLAMA_HOST=0.0.0.0 ollama serve

# Check firewall settings
sudo ufw allow 11434
```

**5. JSON parsing errors:**
```bash
# Try CodeLlama for better structured output
ollama pull codellama
<param name="ollama_model" value="codellama" />
```

### Debug Commands

```bash
# Test Ollama directly
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "prompt": "Hello", "stream": false}'

# Monitor ROS topics
rostopic echo /cmd_vel
rostopic echo /performance_data

# Check logs
roslaunch openai_mocap_controller ollama_mocap_controller.launch --screen
```

## Performance Comparison

The Ollama version provides similar navigation performance to the OpenAI version with these characteristics:

### Advantages
- **No API costs** - Free after initial setup
- **Privacy** - All data stays local
- **Customizable** - Use any Ollama-supported model
- **Offline capable** - Works without internet

### Considerations
- **Setup complexity** - Requires Ollama installation and model download
- **Hardware requirements** - Needs sufficient RAM and CPU
- **Response time** - May be slower than cloud APIs depending on hardware
- **Model quality** - Varies by chosen model

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB (for 3b models)
- **CPU**: 4 cores
- **Storage**: 5GB for model storage

### Recommended Requirements
- **RAM**: 16GB+ (for 7b+ models)
- **CPU**: 8+ cores or GPU acceleration
- **Storage**: 10GB+ for multiple models

### GPU Acceleration (Optional)
```bash
# Install CUDA support for faster inference
# Follow Ollama GPU setup guide for your system
```

## File Structure

```
src/openai_mocap_controller/
├── scripts/
│   ├── openai_mocap_controller.py      # Original OpenAI version
│   └── ollama_mocap_controller.py      # New Ollama version
├── launch/
│   ├── openai_mocap_controller.launch  # Original launch file
│   └── ollama_mocap_controller.launch  # New Ollama launch file
├── test_ollama.py                      # Ollama connection test
├── README.md                           # Original documentation
└── README_OLLAMA.md                    # This file
```

## Example Usage Session

```bash
# 1. Start Ollama server
ollama serve

# 2. Download model (first time only)
ollama pull llama3.2

# 3. Test connection
python3 src/openai_mocap_controller/test_ollama.py

# 4. Launch robot controller
roslaunch openai_mocap_controller ollama_mocap_controller.launch

# 5. Monitor performance
rostopic echo /performance_data
```

## Contributing

When contributing to the Ollama version:

1. Test with multiple models (llama3.2, mistral, codellama)
2. Verify JSON parsing reliability
3. Check performance on different hardware
4. Update documentation for new features
5. Maintain compatibility with the original OpenAI version structure

## License

Same as the original OpenAI Mocap Controller - MIT License.