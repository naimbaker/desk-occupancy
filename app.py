from flask import Flask, render_template, Response, jsonify, session, redirect, url_for, request
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import time

app = Flask(__name__)
app.secret_key = 'cheese'
socketio = SocketIO(app, cors_allowed_origins="*")

# Credentials
secret_key = "cheese"
admin_password = "password"

model = YOLO('yolov8n.pt') 

# --- 1. CAMERA CONFIG ---
CAMERAS = {
    'cam1': "rtsp://10.5.3.210:8554/webcam",
    'cam2': "rtsp://10.5.3.211:8554/webcam"
}

# --- 2. DESK ZONES ---
desk_zones = {
    'cam1': {
        'Desk 1': [0.0, 0.0, 0.5, 0.5],
        'Desk 2': [0.5, 0.0, 1.0, 0.5],
        'Desk 3': [0.0, 0.5, 0.5, 1.0],
        'Desk 4': [0.5, 0.5, 1.0, 1.0],
    },
    'cam2': {
        'Desk 1': [0.0, 0.0, 0.5, 0.5],
        'Desk 2': [0.5, 0.0, 1.0, 0.5],
        'Desk 3': [0.0, 0.5, 0.5, 1.0],
        'Desk 4': [0.5, 0.5, 1.0, 1.0],
    },
}

# --- 3. GLOBAL DATA STORAGE ---
occupancy_data = {
    'cam1': {
        'desks': {desk: {'occupied': False} for desk in desk_zones['cam1']},
        'total_people': 0,
        'last_updated': None,
    },
    'cam2': {
        'desks': {desk: {'occupied': False} for desk in desk_zones['cam2']},
        'total_people': 0,
        'last_updated': None,
    },
}

# --- 4. THE GENERATOR FUNCTION ---
def generate_frames(cam_id):
    camera_source = CAMERAS.get(cam_id)
    cap = cv2.VideoCapture(camera_source)
    
    last_yolo_time = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(1)
            cap = cv2.VideoCapture(camera_source)
            continue

        h, w = frame.shape[:2]
        current_time = time.time()

        # Run YOLO every 2 seconds to keep the server fast
        if current_time - last_yolo_time >= 2:
            results = model(frame, conf=0.15, classes=[0], verbose=False)
            
            # Reset status
            new_status = {desk: {'occupied': False} for desk in desk_zones[cam_id]}
            
            # Detection logic
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy()
                cx, cy = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2

                for desk_name, zone_pct in desk_zones[cam_id].items():
                    zx1, zy1 = int(zone_pct[0] * w), int(zone_pct[1] * h)
                    zx2, zy2 = int(zone_pct[2] * w), int(zone_pct[3] * h)

                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        new_status[desk_name]['occupied'] = True

            # Update Global Data
            occupancy_data[cam_id]['desks'] = new_status
            occupancy_data[cam_id]['total_people'] = len(results[0].boxes)
            occupancy_data[cam_id]['last_updated'] = datetime.now().strftime("%H:%M:%S")
            
            socketio.emit(f'occupancy_update_{cam_id}', occupancy_data[cam_id])
            last_yolo_time = current_time

          

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- 5. ROUTES ---
@app.route('/')
def landing(): return render_template('landing.html')

@app.route('/login', methods=['POST'])
def login():
    password = request.form.get('password')
    if password == admin_password:
        session['role'] = 'admin'
        return redirect(url_for('room_selection'))
    else:
        return redirect(url_for('landing'))

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/guest_login')
def guest_login():
    session['role'] = 'guest' # Give them a Guest "ID Card"
    return redirect(url_for('room_selection'))

@app.route('/room_selection')
def room_selection():
    return render_template('room_selection.html', role=session.get('role', 'guest'))

@app.route('/library_floor2')
def library_floor2(): return render_template('library_floor2.html')

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/occupancy')
def get_occupancy():
    return jsonify(occupancy_data)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)