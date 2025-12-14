import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# =========================
# LOAD MODEL & LABEL
# =========================
model = tf.keras.models.load_model("model/hand_landmark_model.h5")

LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # tanpa J & Z

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

# Untuk smoothing prediksi
history = []
HISTORY_LEN = 7
CONF_THRESHOLD = 0.75

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    letter = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            features = np.array(features).reshape(1, -1)

            prediction = model.predict(features, verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > CONF_THRESHOLD:
                history.append(class_id)
                if len(history) > HISTORY_LEN:
                    history.pop(0)

                # Majority voting
                class_id = max(set(history), key=history.count)
                letter = LABELS[class_id]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # =========================
    # DISPLAY
    # =========================
    cv2.rectangle(frame, (0, 0), (300, 80), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Sign: {letter}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Real-Time Sign Language (Landmark)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
