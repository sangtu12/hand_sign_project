import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import os
import time

# =========================
# LOAD MODEL & ENCODER
# =========================
model = tf.keras.models.load_model("model/hand_landmark_model.h5")
encoder = joblib.load("model/label_encoder.pkl")

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

# =========================
# SCREENSHOT SETUP
# =========================
os.makedirs("screenshots", exist_ok=True)
ss_count = 0

# =========================
# PREDICTION SETTINGS
# =========================
history = []
HISTORY_LEN = 7
CONF_THRESHOLD = 0.7

# =========================
# FPS COUNTER
# =========================
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    letter = "-"
    confidence = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            features = np.array(features).reshape(1, -1)

            prediction = model.predict(features, verbose=0)
            class_id = np.argmax(prediction)
            confidence = float(np.max(prediction))

            if confidence > CONF_THRESHOLD:
                history.append(class_id)
                history = history[-HISTORY_LEN:]
                class_id = max(set(history), key=history.count)
                letter = encoder.inverse_transform([class_id])[0]

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # =========================
    # FPS CALCULATION
    # =========================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # =========================
    # UI DISPLAY
    # =========================
    
    overlay = frame.copy()

    cv2.rectangle(
    overlay,
    (0, 0),
    (420, 160),
    (0, 0, 0),   # warna tetap hitam
    -1
)

    alpha = 0.5  # transparansi (0.0 - 1.0)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


    cv2.putText(frame, f"Sign       : {letter}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence : {confidence:.2f}", (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    cv2.putText(frame, f"FPS        : {int(fps)}", (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.putText(frame, f"History    : {len(history)}/{HISTORY_LEN}", (15, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # =========================
    # HELP TEXT
    # =========================
    cv2.putText(frame, "[S] Screenshot  |  [Q] Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Real-Time Sign Language (Landmark)", frame)

    # =========================
    # KEY CONTROL
    # =========================
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = f"screenshots/ss_{ss_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot disimpan: {filename}")
        ss_count += 1

    elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
