import cv2
import numpy as np

def detect_arrow_direction(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        if len(approx) == 7:
            x, y, w, h = cv2.boundingRect(approx)
            # Analyze aspect ratio and direction
            if w > h:
                direction = "Right" if x < frame.shape[1] // 2 else "Left"
            else:
                direction = "Up" if y < frame.shape[0] // 2 else "Down"

            # Draw contour and direction
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
            cv2.putText(frame, f"Direction: {direction}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break

    return frame
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = detect_arrow_direction(frame)
    cv2.imshow("Arrow Detection", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()