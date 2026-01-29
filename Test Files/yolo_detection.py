from ultralytics import YOLO
import cv2

# Load YOLO model
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')

# Use the camera index you found (probably 1 or 2 for Camo)
camera_index = 0 # Change this to whatever worked in the test

print(f"Connecting to Camo camera (index {camera_index})...")
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not connect to camera")
    print("Try a different camera_index (0, 1, 2, etc.)")
    exit()

print("Camera connected! Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Run YOLO detection
    results = model(frame)
    
    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()
    
    # Display
    cv2.imshow('YOLO Detection - Press Q to Quit', annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")