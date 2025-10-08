import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera!")
    exit()

prev_center = None

def detect_gesture(prev, curr, threshold=30):
    if prev is None or curr is None:
        return None
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    if abs(dx) > abs(dy):
        if dx > threshold:
            return "right"
        elif dx < -threshold:
            return "left"
    else:
        if dy > threshold:
            return "down"
        elif dy < -threshold:
            return "up"
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 500:
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            center = (center_x, center_y)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    gesture = detect_gesture(prev_center, center)
    if gesture:
        print(f"Gesture Detected: {gesture.capitalize()}")

    prev_center = center

    cv2.imshow("Original Frame", frame)
    cv2.imshow("filtered Frame", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()