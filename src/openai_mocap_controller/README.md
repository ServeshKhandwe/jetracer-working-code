# OpenAI Mocap Controller

This package provides an OpenAI-based navigation controller for the Jetracer robot using motion capture data. The system compares different OpenAI models (GPT-4, GPT-3.5-turbo, etc.) for robot navigation performance.

## Features

- **Motion Capture Integration**: Receives real-time position data via WebSocket
- **OpenAI-Powered Navigation**: Uses OpenAI API for movement decisions
- **Model Comparison**: Tests multiple OpenAI models and compares performance
- **Performance Logging**: Tracks and logs navigation metrics
- **No Safety Constraints**: Direct control without safety filters (for research purposes)

## System Overview

The robot receives its position from a motion capture system and uses OpenAI models to make navigation decisions to reach the goal at (0,0,0). The system:

1. Connects to mocap system via WebSocket at `192.168.64.147:8765`
2. Receives robot position data in JSON format
3. Processes LiDAR data for obstacle detection
4. Queries OpenAI API for movement decisions
5. Publishes velocity commands to `/cmd_vel`
6. Logs performance metrics for model comparison

## Prerequisites

- ROS (Robot Operating System)
- Python 3.7+
- OpenAI Python library (`pip install openai`)
- WebSocket client (`pip install websockets`)

## Installation

1. Clone this package into your catkin workspace:
```bash
cd ~/catkin_ws/src
# Package is already in the workspace
```

2. Install Python dependencies:
```bash
pip install openai websockets numpy
```

3. Build the workspace:
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Configuration

### Environment Variables

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### Launch Parameters

The launch file accepts several parameters:

- `openai_api_key`: OpenAI API key (or use environment variable)
- `mocap_server_ip`: Motion capture server IP (default: 192.168.64.147)
- `mocap_server_port`: Motion capture server port (default: 8765)
- `models_to_test`: Comma-separated list of OpenAI models to test
- `test_duration`: Duration to test each model (seconds)
- `max_linear_velocity`: Maximum forward/backward speed (m/s)
- `max_angular_velocity`: Maximum rotation speed (rad/s)

## Usage

### Basic Launch

```bash
roslaunch openai_mocap_controller openai_mocap_controller.launch
```

### Custom Configuration

```bash
roslaunch openai_mocap_controller openai_mocap_controller.launch \
    models_to_test:="gpt-4,gpt-3.5-turbo,gpt-4-turbo" \
    test_duration:=120 \
    max_linear_velocity:=0.2
```

## Motion Capture Data Format

The system expects JSON data in this format:
```json
{
  "timestamp": 1750861745.7979083,
  "objects": {
    "Player": {
      "id": 0,
      "x": -854.1630249023438,
      "y": 338.8022155761719,
      "z": 77.5765609741211,
      "qx": -0.0015750379534438252,
      "qy": 0.0010788951767608523,
      "qz": 0.003049813909456134,
      "qw": 0.9999935030937195
    }
  }
}
```

Position values are assumed to be in millimeters and are converted to meters using the `position_scale_factor`.

## Performance Metrics

The system tracks several performance metrics for each model:

- **Average Distance to Goal**: Mean distance from goal during test period
- **Minimum Distance to Goal**: Closest approach to goal
- **Final Distance to Goal**: Distance at end of test period
- **Total Distance Traveled**: Total path length
- **Efficiency Ratio**: Distance traveled vs. progress toward goal

Performance data is published to `/performance_data` topic and logged for analysis.

## Topics

### Published Topics

- `/cmd_vel` (geometry_msgs/Twist): Velocity commands for robot
- `/performance_data` (std_msgs/String): Real-time performance data (JSON)

### Subscribed Topics

- `/scan` (sensor_msgs/LaserScan): LiDAR data for obstacle detection
- `/odom` (nav_msgs/Odometry): Odometry data (fallback if mocap unavailable)

## Model Comparison

The system automatically cycles through specified OpenAI models, testing each for the configured duration. After testing all models, it publishes a comparison report showing:

- Which model achieved the closest approach to goal
- Which model was most efficient (least distance traveled per unit progress)
- Average performance metrics for each model

## Safety Considerations

⚠️ **Warning**: This controller operates without safety constraints and is intended for research purposes in controlled environments. Ensure:

- Robot is operated in a safe, enclosed area
- Emergency stop is readily available
- Obstacles are properly detected by LiDAR
- Maximum velocities are set to safe values

## Troubleshooting

### Common Issues

1. **No mocap data**: Check WebSocket connection to `192.168.64.147:8765`
2. **OpenAI API errors**: Verify API key and internet connection
3. **Robot not moving**: Check `/cmd_vel` topic and motor controllers
4. **Poor performance**: Adjust decision frequency and velocity limits

### Debug Information

Enable debug logging:
```bash
roslaunch openai_mocap_controller openai_mocap_controller.launch --screen
```

Monitor topics:
```bash
# Watch velocity commands
rostopic echo /cmd_vel

# Monitor performance data
rostopic echo /performance_data
```

## File Structure

```
src/openai_mocap_controller/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
│   └── controller_config.yaml
├── launch/
│   └── openai_mocap_controller.launch
└── scripts/
    └── openai_mocap_controller.py
```

## Contributing

This is a research project for comparing OpenAI model performance in robot navigation. Contributions for additional models, metrics, or analysis tools are welcome.

## License

MIT License - See package.xml for details.