# Installation Guide - OpenAI Mocap Controller

## Quick Setup

### 1. Build the Package

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 2. Install Python Dependencies

```bash
pip3 install openai websockets numpy matplotlib
```

### 3. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 4. Test Mocap Connection

```bash
python3 src/openai_mocap_controller/scripts/test_mocap_connection.py
```

### 5. Launch the Controller

```bash
roslaunch openai_mocap_controller openai_mocap_controller.launch
```

## Detailed Setup

### Prerequisites

- Ubuntu 18.04+ with ROS Melodic/Noetic
- Python 3.7+
- Jetracer robot with LiDAR
- Motion capture system running at 192.168.64.147:8765

### Environment Setup

1. **Create Python virtual environment (recommended):**
```bash
python3 -m venv ~/openai_env
source ~/openai_env/bin/activate
pip install openai websockets numpy matplotlib
```

2. **Set environment variables:**
```bash
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.bashrc
echo 'source ~/openai_env/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

### Configuration

1. **Edit launch file parameters:**
```bash
nano src/openai_mocap_controller/launch/openai_mocap_controller.launch
```

2. **Key parameters to adjust:**
- `models_to_test`: Which OpenAI models to compare
- `test_duration`: How long to test each model (seconds)
- `max_linear_velocity`: Maximum forward speed
- `max_angular_velocity`: Maximum rotation speed
- `mocap_server_ip`: IP address of mocap server

### Testing

1. **Test mocap connection:**
```bash
cd ~/catkin_ws
python3 src/openai_mocap_controller/scripts/test_mocap_connection.py
```

2. **Test OpenAI API:**
```bash
python3 -c "
import openai
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=5
)
print('OpenAI API working:', response.choices[0].message.content)
"
```

3. **Launch with debug output:**
```bash
roslaunch openai_mocap_controller openai_mocap_controller.launch --screen
```

### Troubleshooting

**Common Issues:**

1. **"No module named 'openai'"**
   - Install: `pip3 install openai`

2. **"Connection refused" to mocap server**
   - Check mocap server is running
   - Verify IP address: `ping 192.168.64.147`

3. **OpenAI API errors**
   - Check API key: `echo $OPENAI_API_KEY`
   - Verify internet connection
   - Check API quota/billing

4. **Robot not moving**
   - Check `/cmd_vel` topic: `rostopic echo /cmd_vel`
   - Verify jetracer launch files are working
   - Check motor controller connections

**Debug Commands:**

```bash
# Check ROS topics
rostopic list

# Monitor velocity commands
rostopic echo /cmd_vel

# Check mocap data reception
rostopic echo /performance_data

# View ROS logs
roslog
```

### Performance Analysis

After running tests, analyze results:

```bash
python3 src/openai_mocap_controller/scripts/performance_analyzer.py
```

This will show:
- Model comparison metrics
- Performance plots
- Success rates
- Efficiency analysis

## System Architecture

```
Motion Capture System (192.168.64.147:8765)
    ↓ WebSocket JSON
OpenAI Mocap Controller
    ↓ /cmd_vel
Jetracer Robot
    ↓ /scan, /odom
LiDAR & Odometry Feedback
```

The controller:
1. Receives robot position from mocap via WebSocket
2. Gets LiDAR data for obstacle detection
3. Queries OpenAI API for movement decisions
4. Publishes velocity commands to robot
5. Logs performance data for analysis

## Next Steps

1. Run initial tests with default settings
2. Adjust parameters based on robot performance
3. Add custom OpenAI models to test
4. Analyze results and optimize prompts
5. Compare with existing SafeLLMRA controller