import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# ========================
# LOAD MODEL
# ========================
MODEL_PATH = "model/sign_language_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

print("\n=== MODEL SUMMARY ===")
model.summary() #test commit
print("\nModel Input Shape:", model.input_shape)

print("\n=== DETAIL LAYER ===")
for i, layer in enumerate(model.layers):
    print(f"{i}. {layer.name} | {layer.__class__.__name__} | Output: {layer.output_shape}")

# LABELS 
LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# ========================
# MEDIAPIPE SETUP
# ========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ========================
# STABILIZER (VOTING)
# ========================
prediction_buffer = deque(maxlen=5)

# ========================
# METRICS (REAL-TIME ESTIMATION)
# ========================
total_predictions = 0
stable_predictions = 0
high_conf_predictions = 0

# ========================
# WEBCAM
# ========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera tidak terdeteksi")
    exit()

print("✅ Tekan 'Q' untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    letter = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # ========================
            # LANDMARK LIST (WAJIB)
            # ========================
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            # ========================
            # BOUNDING BOX (SQUARE + DYNAMIC PADDING)
            # ========================
            x_min = int(min(x_list) * w)
            x_max = int(max(x_list) * w)
            y_min = int(min(y_list) * h)
            y_max = int(max(y_list) * h)

            box_width = x_max - x_min
            box_height = y_max - y_min
            box_size = max(box_width, box_height)
            box_size = max(box_size, 10)  # safety

            pad = int(0.25 * box_size)

            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2

            x_min = max(0, x_center - box_size // 2 - pad)
            x_max = min(w, x_center + box_size // 2 + pad)
            y_min = max(0, y_center - box_size // 2 - pad)
            y_max = min(h, y_center + box_size // 2 + pad)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                # ========================
                # PREPROCESS
                # ========================
                IMG_SIZE = 128
                resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                normalized = rgb.astype(np.float32) / 255.0
                input_img = np.expand_dims(normalized, axis=0)

                # ========================
                # PREDICTION
                # ========================
                prediction = model.predict(input_img, verbose=0)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)

                total_predictions += 1

                if confidence > 0.75 and class_id < len(LABELS):
                    prediction_buffer.append(class_id)

                    most_common = max(
                        set(prediction_buffer),
                        key=prediction_buffer.count
                    )

                    if prediction_buffer.count(most_common) >= 3:
                        stable_predictions += 1

                    if confidence > 0.85:
                        high_conf_predictions += 1

                    letter = f"{LABELS[most_common]} ({confidence:.2f})"

                else:
                    prediction_buffer.clear()

            # ========================
            # DRAW
            # ========================
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ========================
    # DISPLAY TEXT
    # ========================
    cv2.putText(
        frame,
        letter,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    # ========================
    # DISPLAY METRICS
    # ========================
    if total_predictions > 0:
        stability_rate = (stable_predictions / total_predictions) * 100
        high_conf_rate = (high_conf_predictions / total_predictions) * 100

        cv2.putText(
            frame,
            f"Stability: {stability_rate:.1f}%",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"High Conf: {high_conf_rate:.1f}%",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    cv2.imshow("Real-Time Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()
