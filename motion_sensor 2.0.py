import cv2
import numpy as np
import time
import os
import logging


MIN_AREA = 1200
LIGHT_TIMEOUT = 5
BLUR_SIZE = (21, 21)
NIGHT_THRESHOLD = 60   # Lower = darker
RECORD_SECONDS = 10

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)


logging.basicConfig(level=logging.INFO)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def is_night(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < NIGHT_THRESHOLD

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logging.error("Camera not accessible")
        return

    logging.info("Smart Motion Security System Started...")

    prev_frame = None
    light_on = False
    last_motion_time = 0

    recording = False
    video_writer = None
    record_start_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_SIZE, 0)

        
        if prev_frame is None:
            prev_frame = gray
            continue

        
        frame_diff = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = any(
            cv2.contourArea(c) > MIN_AREA for c in contours
        )

        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_detected = len(faces) > 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 0), 2)

        # check if its night or day
        night = is_night(frame)

        # trigger section
        if motion_detected and face_detected and night:
            light_on = True
            last_motion_time = time.time()

            # Snapshot
            filename = os.path.join(SAVE_DIR, f"snapshot_{int(time.time())}.jpg")
            cv2.imwrite(filename, frame)
            logging.info(f"Snapshot saved: {filename}")

            # Start recording
            if not recording:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_filename = os.path.join(SAVE_DIR, f"video_{int(time.time())}.avi")
                video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0,
                                               (frame.shape[1], frame.shape[0]))
                recording = True
                record_start_time = time.time()
                logging.info("Recording started")

        
        if recording:
            video_writer.write(frame)

            if time.time() - record_start_time > RECORD_SECONDS:
                recording = False
                video_writer.release()
                logging.info("Recording stopped")

        
        if light_on and (time.time() - last_motion_time > LIGHT_TIMEOUT):
            light_on = False
            logging.info("Light OFF")

       # output
        status = "LIGHT ON 💡" if light_on else "LIGHT OFF 🌙"
        night_text = "NIGHT 🌙" if night else "DAY ☀️"

        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if light_on else (100, 100, 100), 2)

        cv2.putText(frame, night_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)

        cv2.imshow("Smart Security System v2", frame)

        prev_frame = gray

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()