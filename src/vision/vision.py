import cv2
import base64

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read from camera")
    return frame

# Convert frame to base64

def frame_to_base64(frame):
    _, buffer = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    return base64.b64encode(buffer).decode("utf-8")