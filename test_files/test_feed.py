import cv2
import time

cap = cv2.VideoCapture(0)

# Give it time to initialize
time.sleep(2)

# Flush first frames
for _ in range(5):
    cap.read()

print("Reading from camera...")
for i in range(30):
    ret, frame = cap.read()
    print(f"Frame {i}: Success={ret}, Frame shape={frame.shape if ret else 'None'}")
    if ret:
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()