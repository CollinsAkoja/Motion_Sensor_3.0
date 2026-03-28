# AI Security System v3.0

An intelligent motion detection and security system built with Python that uses computer vision to detect motion, recognize people, and send real-time alerts via Telegram.

## Features

- **Motion Detection**: Uses OpenCV to detect movement in the camera feed
- **Person Recognition**: Integrates YOLOv8 for accurate person detection
- **Day/Night Detection**: Automatically distinguishes between daytime and nighttime based on frame brightness
- **Smart Alerts**: Sends photo alerts to Telegram only when motion + person is detected at night
- **Video Recording**: Records short video clips when alerts are triggered
- **Light Simulation**: Simulates turning lights on/off based on activity
- **Cooldown System**: Prevents spam alerts with configurable cooldown periods

## Requirements

- Python 3.7+
- Webcam/Camera
- Internet connection (for Telegram alerts)

## Dependencies

- opencv-python
- numpy
- ultralytics (YOLOv8)
- requests

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CollinsAkoja/Motion_Sensor_3.0.git
cd Motion_Sensor_3.0
```

2. Install dependencies:
```bash
pip install opencv-python numpy ultralytics requests
```

3. Download YOLOv8 model (automatically handled by the script)

## Configuration

Edit the following variables in `motion_sensor_3.0.py`:

- `BOT_TOKEN`: Your Telegram bot token
- `CHAT_ID`: Your Telegram chat ID
- `LIGHT_TIMEOUT`: Seconds to keep light on after motion (default: 5)
- `NIGHT_THRESHOLD`: Brightness threshold for night detection (default: 60)
- `RECORD_SECONDS`: Duration of video recording in seconds (default: 10)
- `ALERT_COOLDOWN`: Minimum seconds between alerts (default: 10)

## Usage

1. Ensure your camera is connected and accessible
2. Configure your Telegram bot token and chat ID
3. Run the script:
```bash
python motion_sensor_3.0.py
```

4. The system will start monitoring. Press 'x' to exit.

## How It Works

1. **Motion Detection**: Compares consecutive frames to detect changes
2. **Person Verification**: Uses YOLOv8 to confirm if detected motion involves a person
3. **Night Check**: Only triggers alerts during nighttime (low brightness)
4. **Alert System**: Captures image and sends to Telegram, starts video recording
5. **Light Control**: Simulates light activation during detected activity

## Output

- Images saved in `captures/` directory with timestamp
- Video recordings saved as AVI files in `captures/` directory
- Real-time status display showing light state
- Telegram notifications with captured images

## Troubleshooting

- **Camera not accessible**: Check camera permissions and connections
- **Telegram errors**: Verify bot token and chat ID are correct
- **YOLO model download**: Ensure internet connection for initial model download

## License

This project is open source. Feel free to modify and distribute.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
