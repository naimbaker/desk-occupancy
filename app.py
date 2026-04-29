from flask import Flask, render_template, Response, jsonify, session, redirect, url_for, request
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import time
import threading

app = Flask(__name__)
app.secret_key = 'cheese'
socketio = SocketIO(app, cors_allowed_origins="*")

# Credentials
secret_key = "cheese"
admin_password = "password"

model = YOLO('yolo26x.pt') 

# --- 1. CAMERA CONFIG ---
CAMERAS = {
    'cam1': "tcp://127.0.0.1:5002",
    'cam2': "tcp://127.0.0.1:8080",
    'cam3':  "tcp://127.0.0.1:5003"
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
        'Desk 10': [0.65, 0.05, 0.95, 0.50],
        'Desk 11': [0.38, 0.05, 0.63, 0.50],
        'Desk 12': [0.05, 0.05, 0.36, 0.50],
        'Desk 13': [0.65, 0.50, 0.95, 0.95],
        'Desk 14': [0.38, 0.50, 0.63, 0.95],
        'Desk 15': [0.05, 0.50, 0.36, 0.95],
    },
    'cam3': {
        'Desk 21': [0.0, 0.0, 0.5, 0.5],
        'Desk 22': [0.5, 0.0, 1.0, 0.5],
        'Desk 23': [0.0, 0.5, 0.5, 1.0],
        'Desk 24': [0.5, 0.5, 1.0, 1.0],
    }
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
    'cam3': {
        'desks': {desk: {'occupied': False} for desk in desk_zones['cam3']},
        'total_people': 0,
        'last_updated': None,
    },
}

# --- 4. PERSISTENT CAMERA MANAGER ---
class CameraStream:
    def __init__(self, source):
        self.source = source
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        cap = None
        while self.running:
            if cap is None or not cap.isOpened():
                print(f"[CameraStream] Connecting to {self.source}...")
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    print(f"[CameraStream] Failed to connect to {self.source}, retrying in 2s...")
                    time.sleep(2)
                    continue
                print(f"[CameraStream] Connected to {self.source}")

            success, frame = cap.read()
            if not success:
                print(f"[CameraStream] Lost connection to {self.source}, retrying...")
                cap.release()
                cap = None
                time.sleep(1)
                continue

            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False


# Start all streams at app launch — TCP handshake happens ONCE on startup
camera_streams = {
    cam_id: CameraStream(source)
    for cam_id, source in CAMERAS.items()
}

for stream in camera_streams.values():
    stream.start()


# --- 5. THE GENERATOR FUNCTION ---
def generate_frames(cam_id):
    stream = camera_streams[cam_id]

    last_yolo_time = 0
    current_status = {}

    while True:
        frame = stream.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        current_time = time.time()

        if current_time - last_yolo_time >= 2:
            results = model(frame, conf=0.15, classes=[0], verbose=False)
            new_status = {desk: {'occupied': False} for desk in desk_zones[cam_id]}

            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy()
                cx, cy = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2
                for desk_name, zone_pct in desk_zones[cam_id].items():
                    zx1, zy1, zx2, zy2 = int(zone_pct[0]*w), int(zone_pct[1]*h), int(zone_pct[2]*w), int(zone_pct[3]*h)
                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        new_status[desk_name]['occupied'] = True

            occupancy_data[cam_id]['desks'] = new_status
            occupancy_data[cam_id]['total_people'] = len(results[0].boxes)
            occupancy_data[cam_id]['last_updated'] = datetime.now().strftime("%H:%M:%S")
            socketio.emit('occupancy_update', occupancy_data[cam_id])

            current_status = new_status
            last_yolo_time = current_time

        for desk_name, zone_pct in desk_zones[cam_id].items():
            zx1, zy1 = int(zone_pct[0] * w), int(zone_pct[1] * h)
            zx2, zy2 = int(zone_pct[2] * w), int(zone_pct[3] * h)

            is_occ = current_status.get(desk_name, {}).get('occupied', False)
            color = (0, 0, 255) if is_occ else (0, 255, 0)

            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 2)
            cv2.putText(frame, desk_name, (zx1 + 5, zy1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# --- 6. ROUTES ---
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
    session['role'] = 'guest'
    return redirect(url_for('room_selection'))

@app.route('/room_selection')
def room_selection():
    return render_template('room_selection.html', role=session.get('role', 'guest'))

@app.route('/library_floor2')
def library_floor2(): 
    return render_template('library_floor2.html')

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/occupancy')
def get_occupancy():
    return jsonify(occupancy_data)

@app.route('/meric_floor3')
def meric_floor3():
    return render_template('MERIC_floor3.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)