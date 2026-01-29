import cv2

print("Searching for available cameras...")
print("Make sure Camo is running on both your phone and Mac!\n")

available_cameras = []

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            print(f"✓ Camera {i} WORKS! Resolution: {width}x{height}")
            available_cameras.append(i)
            
            # Show preview for 2 seconds
            cv2.imshow(f'Camera {i} - Close window to continue', frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"✗ Camera {i} not available")

print(f"\n{'='*50}")
if available_cameras:
    print(f"Found {len(available_cameras)} working camera(s): {available_cameras}")
    print(f"\nUse one of these numbers in your app.py:")
    print(f"camera_index = {available_cameras[0]}  # or try others if this doesn't work")
else:
    print("No cameras found!")
    print("\nTroubleshooting:")
    print("1. Make sure Camo is running on BOTH phone and Mac")
    print("2. Check System Settings → Privacy & Security → Camera")
    print("3. Make sure Python/Terminal has camera permissions")
print(f"{'='*50}")