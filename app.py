from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLO model
model = YOLO('yolov8n.pt')

# Camera setup - change this to your Camo camera index
camera_index = 1
cap = cv2.VideoCapture(camera_index)

# Get frame dimensions
ret, frame = cap.read()
if ret:
    frame_height, frame_width = frame.shape[:2]
else:
    frame_height, frame_width = 480, 640

# Define desk zones (as percentages of frame dimensions)
# Format: [x_start%, y_start%, x_end%, y_end%]
# You can adjust these based on your camera angle
desk_zones = {
    'Desk 1': [0.0, 0.0, 0.5, 0.5],      # Top-left quadrant
    'Desk 2': [0.5, 0.0, 1.0, 0.5],      # Top-right quadrant
    'Desk 3': [0.0, 0.5, 0.5, 1.0],      # Bottom-left quadrant
    'Desk 4': [0.5, 0.5, 1.0, 1.0],      # Bottom-right quadrant
}

# Convert percentage zones to pixel coordinates
def get_zone_coordinates(zone_percentages):
    x1 = int(zone_percentages[0] * frame_width)
    y1 = int(zone_percentages[1] * frame_height)
    x2 = int(zone_percentages[2] * frame_width)
    y2 = int(zone_percentages[3] * frame_height)
    return (x1, y1, x2, y2)

# Check if a bounding box center is within a zone
def is_person_in_zone(box, zone_coords):
    # Get center of bounding box
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Check if center is within zone
    zone_x1, zone_y1, zone_x2, zone_y2 = zone_coords
    return (zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2)

# Global occupancy data
occupancy_data = {
    'desks': {},
    'total_people': 0,
    'last_updated': None
}

def detect_desk_occupancy(frame):
    """Run YOLO detection and determine desk occupancy"""
    # Detect only people (class 0)
    results = model(frame, conf=0.25, classes=[0], verbose=False)
    
    # Initialize desk status
    desk_status = {}
    for desk_name in desk_zones.keys():
        desk_status[desk_name] = {
            'occupied': False,
            'people_count': 0
        }
    
    # Check each detected person
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Check which zone this person is in
            for desk_name, zone_percentages in desk_zones.items():
                zone_coords = get_zone_coordinates(zone_percentages)
                if is_person_in_zone((x1, y1, x2, y2), zone_coords):
                    desk_status[desk_name]['occupied'] = True
                    desk_status[desk_name]['people_count'] += 1
    
    # Update global occupancy data
    occupancy_data['desks'] = desk_status
    occupancy_data['total_people'] = len(results[0].boxes)
    occupancy_data['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Draw zones and annotations on frame
    annotated_frame = draw_zones_and_detections(frame, results, desk_status)
    
    return annotated_frame

def draw_zones_and_detections(frame, results, desk_status):
    """Draw zones, detections, and labels on frame"""
    annotated_frame = results[0].plot()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.7, 2
    
    # Draw desk zones
    for desk_name, zone_percentages in desk_zones.items():
        x1, y1, x2, y2 = get_zone_coordinates(zone_percentages)
        occupied = desk_status[desk_name]['occupied']
        color = (0, 0, 255) if occupied else (0, 255, 0)  # Red or Green
        status_text = "OCCUPIED" if occupied else "AVAILABLE"
        
        # Draw zone rectangle and label
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        label = f"{desk_name}: {status_text}"
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), 
                     (x1 + text_width + 10, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), 
                   font, font_scale, (255, 255, 255), thickness)
    
    return annotated_frame

def generate_frames():
    """Generate video frames with detection"""
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Run detection and get annotated frame
        annotated_frame = detect_desk_occupancy(frame)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # Emit occupancy update via WebSocket
        socketio.emit('occupancy_update', occupancy_data)
        
        # Yield frame for video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/occupancy')
def get_occupancy():
    """API endpoint to get current occupancy status"""
    return jsonify(occupancy_data)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)