# OpenAI Mocap Controller - Project Summary

## Overview

This project implements a new navigation system for the Jetracer robot that uses Large Language Models (LLMs) to make movement decisions based on motion capture data. Unlike the existing SafeLLMRA system, this controller operates without safety constraints and focuses on comparing different LLM performances.

The project now includes **two versions**:
1. **OpenAI Version**: Uses OpenAI API (GPT models) - cloud-based
2. **Ollama Version**: Uses local LLMs via Ollama server - privacy-focused and cost-free

## Key Features

### 1. Motion Capture Integration
- Connects to mocap system via WebSocket at `192.168.64.147:8765`
- Receives real-time robot position data in JSON format
- Converts position data from millimeters to meters
- Extracts yaw angle from quaternion data

### 2. OpenAI-Powered Navigation
- Uses OpenAI API for movement decisions
- Supports multiple models: GPT-4, GPT-3.5-turbo, GPT-4-turbo
- Provides context-aware prompts with robot state and obstacles
- Returns structured JSON responses with velocities and reasoning

### 3. Model Comparison System
- Automatically cycles through different OpenAI models
- Tests each model for configurable duration (default: 60 seconds)
- Tracks performance metrics for each model
- Generates comparative analysis reports

### 4. Performance Metrics
- Distance to goal over time
- Total distance traveled
- Minimum distance achieved
- Success rate (reaching within 0.1m of goal)
- Efficiency ratio (distance traveled vs. progress)

## Project Structure

```
src/openai_mocap_controller/
├── CMakeLists.txt                      # CMake build configuration
├── package.xml                         # ROS package metadata
├── PROJECT_SUMMARY.md                  # This file
├── README.md                           # Original OpenAI documentation
├── README_OLLAMA.md                    # Ollama version documentation
├── setup_environment.sh               # Environment setup script
├── setup_ollama.sh                     # Ollama setup script
├── run_controller.sh                   # Launch script
├── test_ollama.py                      # Ollama connection test
├── compare_versions.py                 # Version comparison tool
├── config/
│   └── controller_config.yaml         # Configuration parameters
├── launch/
│   ├── openai_mocap_controller.launch # OpenAI version launch file
│   └── ollama_mocap_controller.launch # Ollama version launch file
└── scripts/
    ├── openai_mocap_controller.py     # OpenAI controller node
    ├── ollama_mocap_controller.py     # Ollama controller node
    └── performance_analyzer.py        # Performance analysis tool
```

## How It Works

### 1. Data Flow
```
Mocap System → WebSocket → Robot Position
LiDAR → Obstacle Detection
OpenAI API → Movement Decisions
Robot → Velocity Commands
```

### 2. Control Loop
1. Receive mocap position data (10 Hz)
2. Process LiDAR for obstacles
3. Query OpenAI for movement decision (2 Hz)
4. Publish velocity commands to `/cmd_vel`
5. Log performance data

### 3. OpenAI Integration
The system sends contextual prompts to OpenAI including:
- Current robot position and heading
- Distance and angle to goal (0,0)
- Obstacle information from LiDAR
- Current velocities
- Movement constraints

OpenAI responds with:
- Linear velocity (-0.3 to 0.3 m/s)
- Angular velocity (-0.8 to 0.8 rad/s)
- Reasoning for the decision

## Differences from Existing System

| Aspect | SafeLLMRA System | OpenAI Mocap System |
|--------|------------------|---------------------|
| Safety | Zonotope-based safety filters | No safety constraints |
| Decision Making | LLM + Safety controller | Pure OpenAI API |
| Goal | Dynamic waypoints | Fixed goal at (0,0,0) |
| Purpose | Safe navigation | Model comparison research |
| Complexity | High (multiple components) | Simple (single node) |

## Usage Instructions

### 1. Setup
```bash
# Run setup script
./src/openai_mocap_controller/setup_environment.sh

# Set OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 2. Launch Controller
```bash
# Using launch script
./src/openai_mocap_controller/run_controller.sh

# Or directly with ROS
roslaunch openai_mocap_controller openai_mocap_controller.launch
```

### 3. Monitor Performance
```bash
# Watch real-time performance data
rostopic echo /performance_data

# Analyze results after test
python3 src/openai_mocap_controller/scripts/performance_analyzer.py
```

## Configuration Options

### Launch Parameters
- `models_to_test`: Comma-separated list of OpenAI models
- `test_duration`: Duration to test each model (seconds)
- `max_linear_velocity`: Maximum forward/backward speed
- `max_angular_velocity`: Maximum rotation speed
- `decision_frequency`: How often to query OpenAI (Hz)

### Example Custom Launch
```bash
roslaunch openai_mocap_controller openai_mocap_controller.launch \
    models_to_test:="gpt-4,gpt-3.5-turbo" \
    test_duration:=120 \
    max_linear_velocity:=0.2
```

## Expected Mocap Data Format

The system expects JSON data from the mocap system:
```json
{
  "timestamp": 1750861745.7979083,
  "objects": {
    "Player": {
      "id": 0,
      "x": -854.1630249023438,    // mm
      "y": 338.8022155761719,     // mm  
      "z": 77.5765609741211,      // mm
      "qx": -0.0015750379534438252,
      "qy": 0.0010788951767608523,
      "qz": 0.003049813909456134,
      "qw": 0.9999935030937195
    }
  }
}
```

## Performance Analysis

The system generates comprehensive performance reports including:

### Per-Model Metrics
- Average distance to goal
- Minimum distance achieved
- Final distance at test end
- Total distance traveled
- Success rate (< 0.1m from goal)
- Efficiency ratio

### Visualizations
- Distance to goal over time
- Robot trajectory plots
- Performance comparison charts
- Statistical distributions

## Research Applications

This system is designed for:
- Comparing OpenAI model performance in robotics
- Studying LLM decision-making in navigation tasks
- Benchmarking different AI models for robot control
- Research into AI-powered autonomous navigation

## Safety Considerations

⚠️ **Important**: This system operates without safety constraints and is intended for research in controlled environments only. Ensure:
- Robot operates in safe, enclosed area
- Emergency stop is readily available
- Reasonable velocity limits are set
- Proper supervision during operation

## Commands to Run

### OpenAI Version (Original)
```bash
export OPENAI_API_KEY="your-openai-api-key"
roslaunch openai_mocap_controller openai_mocap_controller.launch
```

### Ollama Version (New Local LLM)
```bash
# Start Ollama server
ollama serve

# Download a model (first time only)
ollama pull llama3.2

# Launch controller
roslaunch openai_mocap_controller ollama_mocap_controller.launch
```

Both replace the existing system:
```bash
roslaunch my_robot_controller safellmra_llm_controller.launch
```

## Future Enhancements

Potential improvements:
- Add more OpenAI models (Claude, Gemini)
- Implement dynamic goal positioning
- Add obstacle avoidance strategies comparison
- Include learning from previous attempts
- Real-time model switching based on performance