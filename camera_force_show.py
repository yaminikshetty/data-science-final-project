import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# -------------------------------
# Load model
# -------------------------------
print("🔄 Loading model...")
try:
    model = load_model("model.h5", compile=False)
    print("✅ Model loaded successfully.")
    print("📐 Input shape:", model.input_shape)
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# -------------------------------
# Helper functions
# -------------------------------
def preprocess_frame(frame):
    """Resize ROI to 28x28, grayscale, normalize, flatten -> (1,1,784)"""
    roi = cv2.resize(frame, (28, 28))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = roi.astype("float32") / 255.0
    roi = roi.flatten().reshape(1, 1, 784)  # match model input (1,1,784)
    return roi

def keras_predict(model, image):
    try:
        preds = model.predict(image, verbose=0)
        pred_class = np.argmax(preds[0, -1])  # take last timestep prediction
        confidence = np.max(preds[0, -1])
        return pred_class, confidence
    except Exception as e:
        print("⚠️ Prediction error:", e)
        print("🔹 Image shape:", image.shape)
        return None, None

# -------------------------------
# Main
# -------------------------------
def main():
    print("🎥 Starting webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Camera not found.")
        return
    print("✅ Camera opened successfully.")
    print("🖐 Place your hand inside the yellow box.")
    print("Press 'q' to quit.")

    prediction_history = deque(maxlen=10)  # smooth predictions

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not captured.")
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        x, y, box_w, box_h = w//2 - 100, h//2 - 100, 200, 200

        # Draw region box
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 255), 2)
        roi = frame[y:y + box_h, x:x + box_w]

        # Preprocess ROI
        processed = preprocess_frame(roi)

        # Predict
        pred_class, confidence = keras_predict(model, processed)

        # Draw info
        if pred_class is not None and confidence > 0.75:
            prediction_history.append(pred_class)
            most_common = max(set(prediction_history), key=prediction_history.count)
            text = f"Prediction: {most_common} ({confidence:.2f})"
            color = (0, 255, 0)
        else:
            text = "No confident prediction"
            color = (0, 0, 255)

        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Camera closed successfully.")

# -------------------------------
if __name__ == "__main__":
    main()
