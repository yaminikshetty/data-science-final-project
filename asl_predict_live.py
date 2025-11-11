# asl_predict_live.py
# ===============================================
# Real-time ASL Prediction using improved CNN
# ===============================================

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import os
import sys

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "improved_asl_model.h5"
CONF_THRESH = 0.70
STABLE_FRAMES = 8
COOLDOWN_FRAMES = 10
MIN_CONTOUR_AREA = 2000
ROI_SIZE = 200
ROI_X_OFFSET = 300
ROI_Y_OFFSET = 100

labels = {i: chr(ord('A') + i) for i in range(26)}

# -----------------------------
# Load model
# -----------------------------
def load_model_file(path):
    if not os.path.exists(path):
        print(f"❌ Model not found: {path}")
        sys.exit(1)
    print(f"🧠 Loading model: {path}")
    return load_model(path, compile=False)

# -----------------------------
# Preprocess frame
# -----------------------------
def preprocess_frame(frame):
    img = cv2.resize(frame, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1,28,28,1)
    return img

# -----------------------------
# Detect hand contour
# -----------------------------
def hand_present(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0
    area = max(cv2.contourArea(c) for c in contours)
    return area >= MIN_CONTOUR_AREA, int(area)

# -----------------------------
# Prediction
# -----------------------------
def make_prediction(model, roi):
    img = preprocess_frame(roi)
    probs = model.predict(img, verbose=0)[0]
    class_idx = np.argmax(probs)
    conf = np.max(probs)
    return class_idx, conf

# -----------------------------
# Main
# -----------------------------
def main():
    model = load_model_file(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera error.")
        sys.exit(1)

    print("\n🎥 Ready — show signs inside the yellow box (ROI)!")
    print("Press 'q' to quit | Press 'c' to calibrate (print contour area)\n")

    pred_queue = deque(maxlen=STABLE_FRAMES)
    last_confirmed = None
    cooldown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        x, y = ROI_X_OFFSET, ROI_Y_OFFSET
        w, h = ROI_SIZE, ROI_SIZE
        roi = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        present, area = hand_present(roi)
        display_text = "No hand detected"

        if present and cooldown == 0:
            idx, conf = make_prediction(model, roi)
            pred_queue.append((idx, conf))

            if len(pred_queue) == STABLE_FRAMES:
                classes = [p[0] for p in pred_queue]
                confs = [p[1] for p in pred_queue]
                most_common = max(set(classes), key=classes.count)
                avg_conf = np.mean([c for (i, c) in pred_queue if i == most_common])

                if avg_conf > CONF_THRESH:
                    last_confirmed = most_common
                    letter = labels.get(most_common, "?")
                    display_text = f"Predicted: {letter} ({avg_conf*100:.1f}%)"
                    cooldown = COOLDOWN_FRAMES
                    pred_queue.clear()

                    # ✅ Print detection info in terminal
                    print(f"🖐️ Area: {area} | 🔤 Detected: {letter} ({avg_conf*100:.2f}%)")

        else:
            if cooldown > 0:
                cooldown -= 1
            if last_confirmed is not None:
                letter = labels.get(last_confirmed, "?")
                display_text = f"Held: {letter}"

        # Draw text overlays on frame
        cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Contour Area: {area}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("ASL Recognition", frame)

        # Keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print(f"📏 Current Contour Area: {area} (minimum required: {MIN_CONTOUR_AREA})")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
