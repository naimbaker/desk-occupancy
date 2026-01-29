# Desk Occupancy Detection System

A real-time desk occupancy detection system using YOLO and Flask.

## Requirements
- Python 3.8 or higher
- Camo Studio (or OBS) for phone camera connection
- macOS or Linux (Windows should work but untested)

## Setup Instructions

### 1. Clone or Extract Project
Extract the zip file to your desired location.

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Connect Camera
- Install Camo Studio on your phone and Mac
- Connect your phone via Camo
- Run `find_camera.py` to find your camera index
- Update `camera_index` in `app.py` with the correct number

### 5. Run the Application
```bash
python app.py
```

### 6. Open in Browser
Navigate to: `http://localhost:5001`

## Troubleshooting

**Can't find camera?**
- Run `python find_camera.py` to detect available cameras
- Make sure Camo is running on both phone and Mac
- Try camera index 0, 1, or 2 in `app.py`

**Port already in use?**
- Change port in `app.py` from 5001 to another number like 5002
- Or disable AirPlay Receiver in System Settings

**Slow detection?**
- Reduce frame size in `app.py`
- Use `yolov8n.pt` (nano model, fastest)

## File Structure
```
desk-occupancy/
├── app.py              # Main Flask application
├── find_camera.py      # Camera detection utility
├── templates/
│   └── index.html      # Web interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Adjusting Desk Zones

In `app.py`, modify the `desk_zones` dictionary to match your camera angle:
```python
desk_zones = {
    'Desk 1': [0.0, 0.0, 0.5, 0.5],   # [x_start%, y_start%, x_end%, y_end%]
    'Desk 2': [0.5, 0.0, 1.0, 0.5],
    'Desk 3': [0.0, 0.5, 0.5, 1.0],
    'Desk 4': [0.5, 0.5, 1.0, 1.0],
}
```

Values are percentages from 0.0 to 1.0 of the frame dimensions.
```
