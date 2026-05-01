# Desk Occupancy Detection System

A real-time, multi-camera desk occupancy detection system using YOLO, Flask, and WebSockets. This system maps physical desk locations to a digital SVG floorplan, providing live status updates for university study spaces.

## Key Features
* Multi-Camera Support: Manages multiple concurrent camera streams using a persistent threading manager.
* Dynamic Floorplan Integration: Maps detection results directly to interactive SVG elements for Library and MERIC floors.
* Live Updates: Uses Socket.IO to push occupancy changes to the frontend without requiring page refreshes.
* Admin Dashboard: Secure login for staff to view live video feeds and individual camera metrics.

## Requirements
* Python 3.8 or higher
* Raspberry Pi (3, 4, or 5) with Pi Camera or USB Webcam
* macOS, Linux, Windows

## Setup Instructions

1. Clone or Extract Project
   Extract the zip file to your desired location.

2. Create Virtual Environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
   pip install -r requirements.txt

4. Run the Application
   python app.py

5. Open in Browser
   Navigate to: http://localhost:5001

## Configuration (app.py)
* CAMERAS: Map camera IDs to their hardware indices (e.g., 0, 1, or 2).
* FLOORPLAN_CAMERAS: Group cameras by their physical floor location
* desk_zones: Define coordinate percentages [x_start, y_start, x_end, y_end] for each desk per camera.

## Troubleshooting

Can't find camera?
* Run 'python find_camera.py' to detect available cameras.

Port already in use?
* Change port in 'app.py' from 5001 to another number like 5002.

Slow detection?
* The system currently runs YOLO inference every 2 seconds. Reduce this interval in 'generate_frames' if needed.
* Use 'yolov8n.pt' (nano model) for maximum speed.

## File Structure
* app.py              # Main Flask application and YOLO processing
* find_camera.py      # Camera detection utility
* templates/          # UI templates (landing, room selection, floorplans)
* requirements.txt    # Python dependencies (ultralytics, flask-socketio, opencv)
* README.md           # Documentation
