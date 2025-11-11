# asl_mediapipe_ultra.py
# ============================================================
# Ultra-Accurate ASL Recognition with MediaPipe + CNN + Calibration
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os, sys, time
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "improved_asl_model.h5"
TRAIN_CSV = "sign_mnist_train.csv"
TEST_CSV = "sign_mnist_test.csv"
IMG_SIZE = 28
CONF_THRESH = 0.75
STABLE_FRAMES = 8
COOLDOWN_FRAMES = 10
ROI_PADDING = 0.25
MIN_BBOX_AREA = 800
MAX_HANDS = 1
DRAW_LANDMARKS = True

labels = {i: chr(ord('A') + i) for i in range(26)}

# ============================================================
# 1️⃣ DATA + MODEL
# ============================================================
def load_data():
    print("📦 Loading Sign-MNIST dataset...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    X_train = train.drop("label", axis=1).values.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test  = test.drop("label", axis=1).values.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = to_categorical(train["label"], 26)
    y_test  = to_categorical(test["label"], 26)
    print(f"✅ Loaded {len(X_train)} train / {len(X_test)} test samples.")
    return X_train, X_test, y_train, y_test

def build_model():
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)), Dropout(0.25),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)), Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_or_load_model():
    if os.path.exists(MODEL_PATH):
        print(f"🧠 Loading existing model: {MODEL_PATH}")
        return load_model(MODEL_PATH, compile=False)
    X_train, X_test, y_train, y_test = load_data()
    model = build_model()
    print("🏋️ Training CNN...")
    model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_test, y_test))
    model.save(MODEL_PATH)
    print(f"✅ Model saved as {MODEL_PATH}")
    return model

# ============================================================
# 2️⃣ IMAGE PREPROCESSING
# ============================================================
def preprocess_roi(img):
    """Enhance ROI robustness under lighting variations."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = gray.astype("float32") / 255.0
    return np.expand_dims(gray, axis=(0, -1))

# ============================================================
# 3️⃣ MEDIAPIPE HAND ROI
# ============================================================
def bbox_from_landmarks(landmarks, w, h, pad=ROI_PADDING):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = int(max(0, min(xs))), int(min(w-1, max(xs)))
    y1, y2 = int(max(0, min(ys))), int(min(h-1, max(ys)))
    bw, bh = x2-x1, y2-y1
    size = max(bw, bh, 1)
    pad_px = int(size * pad)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    x1, y1 = max(0, cx - (size//2 + pad_px)), max(0, cy - (size//2 + pad_px))
    x2, y2 = min(w-1, cx + (size//2 + pad_px)), min(h-1, cy + (size//2 + pad_px))
    return x1, y1, x2, y2, (x2-x1)*(y2-y1)

# ============================================================
# 4️⃣ LIVE DETECTION
# ============================================================
def live_detection(model):
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=MAX_HANDS, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Camera not accessible.")
        sys.exit(1)
    print("\n🎥 Camera started — perform ASL gestures.")
    print("Press 'q' to quit | 'c' for area calibration\n")

    pred_queue = deque(maxlen=STABLE_FRAMES)
    last_confirmed = None
    cooldown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame read error.")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        display_text = "No hand detected"
        bbox_area = 0
        conf_disp = 0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            if DRAW_LANDMARKS:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            x1,y1,x2,y2,bbox_area = bbox_from_landmarks(hand.landmark, w, h)
            if bbox_area > MIN_BBOX_AREA:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    x = preprocess_roi(roi)
                    probs = model.predict(x, verbose=0)[0]
                    idx = np.argmax(probs)
                    conf = np.max(probs)
                    conf_disp = conf
                    pred_queue.append((idx, conf))
                    if len(pred_queue) == STABLE_FRAMES:
                        cls = [p[0] for p in pred_queue]
                        best = max(set(cls), key=cls.count)
                        avg_conf = np.mean([c for (i, c) in pred_queue if i == best])
                        if avg_conf > CONF_THRESH:
                            letter = labels.get(best, "?")
                            print(f"🖐️ Area: {bbox_area} | 🔤 {letter} ({avg_conf*100:.2f}%)")
                            display_text = f"{letter} ({avg_conf*100:.1f}%)"
                            last_confirmed = best
                            cooldown = COOLDOWN_FRAMES
                            pred_queue.clear()
                        else:
                            display_text = "Detecting..."
            else:
                display_text = "Hand too small"
        else:
            if cooldown>0: cooldown-=1
            if last_confirmed is not None:
                display_text = f"Held: {labels[last_confirmed]}"
            pred_queue.clear()

        if bbox_area>0:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,display_text,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(frame,f"Area: {bbox_area}",(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        cv2.putText(frame,f"Conf: {conf_disp*100:.1f}%",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        cv2.imshow("ASL Ultra MediaPipe", frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key==ord('c'):
            print(f"📏 Area calibration: {bbox_area} (min {MIN_BBOX_AREA})")

    cap.release(); cv2.destroyAllWindows(); hands.close()
    print("\n👋 Session ended.")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    model = train_or_load_model()
    live_detection(model)
