import cv2
import numpy as np
import time
import os
import logging
import requests
from ultralytics import YOLO

# configurations

LIGHT_TIMEOUT = 5
NIGHT_THRESHOLD = 60
RECORD_SECONDS = 10

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

BOT_TOKEN = "8438222375:AAE6NwAzTG5rPVmKYTzaEsipORtEhQ0-KRY"
CHAT_ID = "collinsakojabot"

# logs
logging.basicConfig(level=logging.INFO)

# yolo model
model = YOLO("yolov8n.pt")  # lightweight model

# functions
def is_night(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < NIGHT_THRESHOLD

def send_telegram_image(image_path):
    try:
        url = f"https://api.telegram.org/collinsakojabot{8438222375:AAE6NwAzTG5rPVmKYTzaEsipORtEhQ0-KRY}/sendPhoto"
        with open(image_path, "rb") as img:
            requests.post(url, data={"chat_id": CHAT_ID}, files={"photo": img})
        logging.info("Alert sent to Telegram")
    except Exception as e:
        logging.warning(f"Telegram error: {e}")

def detect_person(frame):
    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                return True
    return False


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logging.error("Camera not accessible")
        return

    prev_frame = None
    light_on = False
    last_motion_time = 0

    recording = False
    video_writer = None
    record_start_time = 0

    last_alert_time = 0
    ALERT_COOLDOWN = 10  # seconds

    logging.info("AI Security System v3 Started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        # ================= MOTION =================
        frame_diff = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = any(cv2.contourArea(c) > MIN_AREA for c in contours)

        # ai detection 
        person_detected = detect_person(frame)
        night = is_night(frame)

        # Trigger section
        if motion_detected and person_detected and night:
            light_on = True
            last_motion_time = time.time()

            timestamp = int(time.time())
            image_path = os.path.join(SAVE_DIR, f"alert_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)

            # Telegram alert (cooldown)
            if time.time() - last_alert_time > ALERT_COOLDOWN:
                send_telegram_image(image_path)
                last_alert_time = time.time()

            # Start recording
            if not recording:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_path = os.path.join(SAVE_DIR, f"video_{timestamp}.avi")
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0,
                                               (frame.shape[1], frame.shape[0]))
                recording = True
                record_start_time = time.time()

        # recording
        if recording:
            video_writer.write(frame)
            if time.time() - record_start_time > RECORD_SECONDS:
                recording = False
                video_writer.release()

    #    day_light
        if light_on and (time.time() - last_motion_time > LIGHT_TIMEOUT):
            light_on = False
    # display_ui
        status = "LIGHT ON 💡" if light_on else "LIGHT OFF 🌙"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if light_on else (100, 100, 100), 2)

        cv2.imshow("AI Security System v3", frame)

        prev_frame = gray

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()