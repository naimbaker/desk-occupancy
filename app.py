from flask import Flask, render_template, Response, jsonify, session, redirect, url_for, request
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import time

app = Flask(__name__)
app.secret_key = 'cheese' # required for sessions
socketio = SocketIO(app, cors_allowed_origins="*")

#hardcoded credentials for now
admin_username = "admin"
admin_password = "password"

# Load YOLO model
model = YOLO('yolov8x.pt')

# Camera URL (Raspberry Pi MJPEG stream)
camera_url = "rtsp://10.5.3.210:8554/webcam"

# Define desk zones (percentages of frame)
desk_zones = {
    'Desk 1': [0.0, 0.0, 0.5, 0.5],  # Top-left
    'Desk 2': [0.5, 0.0, 1.0, 0.5],  # Top-right
    'Desk 3': [0.0, 0.5, 0.5, 1.0],  # Bottom-left
    'Desk 4': [0.5, 0.5, 1.0, 1.0],  # Bottom-right
}

# Global occupancy data
occupancy_data = {
    'desks': {},
    'total_people': 0,
    'last_updated': None
}

# Initialize desk status
initial_desk_status = {desk: {'occupied': False, 'people_count': 0} for desk in desk_zones.keys()}
occupancy_data['desks'] = initial_desk_status.copy()

# Helper functions
def get_zone_coordinates(zone_percentages, frame_width, frame_height):
    x1 = int(zone_percentages[0] * frame_width)
    y1 = int(zone_percentages[1] * frame_height)
    x2 = int(zone_percentages[2] * frame_width)
    y2 = int(zone_percentages[3] * frame_height)
    return x1, y1, x2, y2

def is_person_in_zone(box, zone_coords):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    zx1, zy1, zx2, zy2 = zone_coords
    return zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2

def draw_zones_and_detections(frame, results, desk_status):
    annotated_frame = results[0].plot()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.7, 2
    frame_height, frame_width = frame.shape[:2]
    
    for desk_name, zone_percentages in desk_zones.items():
        x1, y1, x2, y2 = get_zone_coordinates(zone_percentages, frame_width, frame_height)
        occupied = desk_status[desk_name]['occupied']
        color = (0, 0, 255) if occupied else (0, 255, 0)
        status_text = "OCCUPIED" if occupied else "AVAILABLE"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(f"{desk_name}: {status_text}", font, font_scale, thickness)
        cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated_frame, f"{desk_name}: {status_text}", (x1 + 5, y1 - 5),
                    font, font_scale, (255, 255, 255), thickness)
    return annotated_frame

def draw_zones(frame, desk_status):
    annotated_frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.7, 2
    frame_height, frame_width = frame.shape[:2]
    
    for desk_name, zone_percentages in desk_zones.items():
        x1, y1, x2, y2 = get_zone_coordinates(zone_percentages, frame_width, frame_height)
        occupied = desk_status[desk_name]['occupied']
        color = (0, 0, 255) if occupied else (0, 255, 0)
        status_text = "OCCUPIED" if occupied else "AVAILABLE"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(f"{desk_name}: {status_text}", font, font_scale, thickness)
        cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated_frame, f"{desk_name}: {status_text}", (x1 + 5, y1 - 5),
                    font, font_scale, (255, 255, 255), thickness)
    return annotated_frame

# --- Core: generate frames inside generator ---
def generate_frames():
    cap = cv2.VideoCapture(camera_url)  # Open camera here
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera stream: {camera_url}")
    
    ret, frame = cap.read()
    if ret:
        frame_height, frame_width = frame.shape[:2]
    else:
        frame_height, frame_width = 480, 640

    last_update = 0
    last_desk_status = initial_desk_status.copy()
    
    while True:
        success, frame = cap.read()
        if not success:
            continue
        
        current_time = time.time()
        if current_time - last_update >= 3:
            # Run YOLO detection
            results = model(frame, conf=0.15, classes=[0], verbose=False)
            
            desk_status = {desk: {'occupied': False, 'people_count': 0} for desk in desk_zones.keys()}
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    for desk_name, zone_percentages in desk_zones.items():
                        zone_coords = get_zone_coordinates(zone_percentages, frame_width, frame_height)
                        if is_person_in_zone((x1, y1, x2, y2), zone_coords):
                            desk_status[desk_name]['occupied'] = True
                            desk_status[desk_name]['people_count'] += 1
            
            last_desk_status = desk_status
            occupancy_data['desks'] = desk_status
            occupancy_data['total_people'] = len(results[0].boxes)
            occupancy_data['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            socketio.emit('occupancy_update', occupancy_data)
            
            last_update = current_time
            annotated_frame = draw_zones_and_detections(frame, results, desk_status)
        else:
            annotated_frame = draw_zones(frame, last_desk_status)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Flask routes ---
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/library_floor2')
def library_floor2():
    return render_template('library_floor2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/occupancy')
def get_occupancy():
    return jsonify(occupancy_data)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/room_selection')
def room_selection():
    role = session.get('role', 'guest') # Default to guest if no role in session
    return render_template('room_selection.html', role=role)

@app.route('/continue_as_guest')
def continue_as_guest():
    return redirect(url_for('room_selection'))


@app.route('/login', methods=['POST'])
def login():
    password = request.form.get('password') # Get password from the input field
    
    if password == "password": # Check against your hardcoded password
        session['role'] = 'admin' # Give them the Admin "ID Card"
        return redirect(url_for('room_selection'))
    else:
        # If password fails, you could redirect back or show an error
        return redirect(url_for('landing'))

@app.route('/guest_login')
def guest_login():
    session['role'] = 'guest' # Give them a Guest "ID Card"
    return redirect(url_for('room_selection'))


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, use_reloader=False)