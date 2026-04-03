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

# Hardcoded credentials 
admin_username = "admin"
admin_password = "password"

# Load YOLO model — swap to 'yolov8n.pt' if two feeds are too slow
model = YOLO('yolov8x.pt')

# --- Camera config ---
# 0 = built-in webcam (floor 2), 1 = USB webcam (floor 3)
# Swap these for RTSP URLs when you move to the Raspberry Pis
CAMERAS = {
    'cam1': "rtsp://10.5.3.210:8554/webcam"
}

# --- Desk zones per floor (percentages of frame) ---
# Tune these to match each camera's actual view of the desks
desk_zones = {
    'floor2': {
        'Desk 1': [0.0, 0.0, 0.5, 0.5],
        'Desk 2': [0.5, 0.0, 1.0, 0.5],
        'Desk 3': [0.0, 0.5, 0.5, 1.0],
        'Desk 4': [0.5, 0.5, 1.0, 1.0],
    },
    'floor3': {
        'Desk 1': [0.0, 0.0, 0.5, 0.5],
        'Desk 2': [0.5, 0.0, 1.0, 0.5],
        'Desk 3': [0.0, 0.5, 0.5, 1.0],
        'Desk 4': [0.5, 0.5, 1.0, 1.0],
    },
}

# --- Global occupancy data per floor ---
occupancy_data = {
    'floor2': {
        'desks': {desk: {'occupied': False, 'people_count': 0} for desk in desk_zones['floor2']},
        'total_people': 0,
        'last_updated': None,
    },
    'floor3': {
        'desks': {desk: {'occupied': False, 'people_count': 0} for desk in desk_zones['floor3']},
        'total_people': 0,
        'last_updated': None,
    },
}

# --- Helper functions ---
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

def draw_zones_and_detections(frame, results, desk_status, floor):
    annotated_frame = results[0].plot()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.7, 2
    frame_height, frame_width = frame.shape[:2]

    for desk_name, zone_percentages in desk_zones[floor].items():
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

def draw_zones(frame, desk_status, floor):
    annotated_frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.7, 2
    frame_height, frame_width = frame.shape[:2]

    for desk_name, zone_percentages in desk_zones[floor].items():
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

# --- Core frame generator (floor-aware) ---
def generate_frames(floor):
    camera_source = CAMERAS[floor]
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera for {floor}: {camera_source}")

    ret, frame = cap.read()
    if ret:
        frame_height, frame_width = frame.shape[:2]
    else:
        frame_height, frame_width = 480, 640

    last_update = 0
    last_desk_status = {desk: {'occupied': False, 'people_count': 0} for desk in desk_zones[floor]}

    while True:
        success, frame = cap.read()
        if not success:
            continue

        current_time = time.time()
        if current_time - last_update >= 3:
            results = model(frame, conf=0.15, classes=[0], verbose=False)

            desk_status = {desk: {'occupied': False, 'people_count': 0} for desk in desk_zones[floor]}

            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    for desk_name, zone_percentages in desk_zones[floor].items():
                        zone_coords = get_zone_coordinates(zone_percentages, frame_width, frame_height)
                        if is_person_in_zone((x1, y1, x2, y2), zone_coords):
                            desk_status[desk_name]['occupied'] = True
                            desk_status[desk_name]['people_count'] += 1

            last_desk_status = desk_status
            occupancy_data[floor]['desks'] = desk_status
            occupancy_data[floor]['total_people'] = len(results[0].boxes)
            occupancy_data[floor]['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Emit floor-specific event so each page only reacts to its own data
            socketio.emit(f'occupancy_update_{floor}', occupancy_data[floor])

            last_update = current_time
            annotated_frame = draw_zones_and_detections(frame, results, desk_status, floor)
        else:
            annotated_frame = draw_zones(frame, last_desk_status, floor)

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

@app.route('/library_floor3')
def library_floor3():
    return render_template('library_floor3.html')

# Separate video feed routes per floor
@app.route('/video_feed/floor2')
def video_feed_floor2():
    return Response(generate_frames('floor2'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/floor3')
def video_feed_floor3():
    return Response(generate_frames('floor3'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Occupancy API — returns all floors, or a specific floor with ?floor=floor2
@app.route('/api/occupancy')
def get_occupancy():
    floor = request.args.get('floor')
    if floor and floor in occupancy_data:
        return jsonify(occupancy_data[floor])
    return jsonify(occupancy_data)

# Endpoint for Pi-based detectors to push results (future use)
@app.route('/api/update_occupancy', methods=['POST'])
def update_occupancy():
    data = request.get_json()
    floor = data.get('floor')
    if not floor or floor not in occupancy_data:
        return jsonify({'error': 'invalid floor'}), 400
    occupancy_data[floor]['desks'] = data['desks']
    occupancy_data[floor]['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    socketio.emit(f'occupancy_update_{floor}', occupancy_data[floor])
    return jsonify({'status': 'ok'})

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/room_selection')
def room_selection():
    role = session.get('role', 'guest')
    return render_template('room_selection.html', role=role)

@app.route('/continue_as_guest')
def continue_as_guest():
    return redirect(url_for('room_selection'))

@app.route('/login', methods=['POST'])
def login():
    password = request.form.get('password')
    if password == "password":
        session['role'] = 'admin'
        return redirect(url_for('room_selection'))
    else:
        return redirect(url_for('landing'))

@app.route('/guest_login')
def guest_login():
    session['role'] = 'guest'
    return redirect(url_for('room_selection'))

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, use_reloader=False)