import cv2
import numpy as np

KNOWN_WIDTH_CM = 5.0
FOCAL_LENGTH = 500

green_lower = np.array([40, 70, 70])
green_upper = np.array([80, 255, 255])
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([179, 255, 255])

cap = cv2.VideoCapture(0)
frame_width = 640

def estimate_distance(perceived_width_pixels):
    if perceived_width_pixels == 0:
        return None
    return (KNOWN_WIDTH_CM * FOCAL_LENGTH) / perceived_width_pixels

def decide_turn(color_name, cx, distance):
    side = "left"
    if cx >= frame_width // 2:
        side = "right"

    turn = "NO ACTION"
    is_red = color_name == "Red"
    is_green = color_name == "Green"

    # Pööramisloogika ilma elif/else
    turn = turn * (not is_red and not is_green)  # Jätab "NO ACTION", kui pole Red ega Green

    # Kui on Red
    turn = "TURN RIGHT" * (is_red and side == "left") + \
           "TURN LEFT"  * (is_red and side == "right") + \
           turn * (not is_red)

    # Kui on Green
    turn = "TURN LEFT"  * (is_green and side == "right") + \
           "TURN RIGHT" * (is_green and side == "left") + \
           turn * (not is_green)

    return {
        "color": color_name,
        "distance": distance,
        "side": side,
        "turn": turn
    }

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, 480))
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    detected = []

    for mask, name in [
        (green_mask, "Green"),
        (red_mask, "Red")
    ]:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        max_area = 0
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500 and area > max_area:
                best_cnt = cnt
                max_area = area
        if best_cnt is not None:
            x, y, w, h = cv2.boundingRect(best_cnt)
            cx = x + w // 2
            distance = estimate_distance(w)
            if distance is not None:
                decision = decide_turn(name, cx, distance)
                detected.append(decision)

    if len(detected) > 0:
        closest = min(detected, key=lambda d: d["distance"])
        print(f"{closest['color']} detected at {closest['distance']:.1f} cm on the {closest['side']} side → {closest['turn']}")

    cv2.imshow("Camera View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
