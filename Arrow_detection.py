import cv2
import numpy as np

def detect_arrows_from_video():
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny Edge Detection before thresholding for better accuracy
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours from the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        arrow_detected = False
        direction = ""

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            epsilon = 0.02 * cv2.arcLength(contour, True)  # Tighter approximation
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if 7 <= len(approx) <= 12:  # Arrows typically have 7-12 vertices
                arrow_detected = True

                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                # Determine direction based on aspect ratio & position
                if aspect_ratio > 1.3:  
                    direction = "Left" if x < frame.shape[1] // 2 else "Right"
                elif aspect_ratio < 0.75:  
                    direction = "Up" if y < frame.shape[0] // 2 else "Down"
                break  

        if arrow_detected:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)

        cv2.imshow("Arrow Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
detect_arrows_from_video()
