import cv2
import numpy as np

# Define HSV color ranges
color_ranges = {
    "yellow": ([20, 100, 100], [30, 255, 255]),
    "green": ([40, 70, 70], [80, 255, 255]),
    "blue": ([100, 150, 0], [140, 255, 255]),
    "red1": ([0, 120, 70], [10, 255, 255]),
    "red2": ([170, 120, 70], [180, 255, 255])
}

# BGR for drawing
color_bgr = {
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "red": (0, 0, 255)
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Handle red separately since it wraps around hue=0
    lower_red1 = np.array(color_ranges["red1"][0])
    upper_red1 = np.array(color_ranges["red1"][1])
    lower_red2 = np.array(color_ranges["red2"][0])
    upper_red2 = np.array(color_ranges["red2"][1])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.add(mask_red1, mask_red2)

    # Clean red mask and detect contours
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr["red"], 2)
            cv2.putText(frame, "red", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr["red"], 2)

    # Handle other colors
    for color_name in ["yellow", "green", "blue"]:
        lower = np.array(color_ranges[color_name][0])
        upper = np.array(color_ranges[color_name][1])
        mask = cv2.inRange(hsv, lower, upper)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr[color_name], 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color_name], 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
