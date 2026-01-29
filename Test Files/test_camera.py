import cv2

# Try camera indices 0-5
for i in range(6):
    print(f"Trying camera {i}...")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera {i} works! Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cv2.imshow(f'Camera {i} - Press any key', frame)
            cv2.waitKey(2000)  # Show for 2 seconds
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap.release()
    else:
        print(f"✗ Camera {i} not available")