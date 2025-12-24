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
            # BOUNDING BOX
            # ========================
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_list) * w)
            x_max = int(max(x_list) * w)
            y_min = int(min(y_list) * h)
            y_max = int(max(y_list) * h)

            pad = 20
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                # ========================
                # PREPROCESS
                # ========================
                resized = cv2.resize(hand_img, (64, 64))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                normalized = rgb / 255.0
                input_img = np.expand_dims(normalized, axis=0)

                # ========================
                # PREDICTION
                # ========================
                prediction = model.predict(input_img, verbose=0)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)

                if confidence > 0.75 and class_id < len(LABELS):
                    prediction_buffer.append(class_id)
                    most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                    letter = f"{LABELS[most_common]} ({confidence:.2f})"
                else:
                    prediction_buffer.clear()

            # DRAW
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

    cv2.imshow("Real-Time Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()
