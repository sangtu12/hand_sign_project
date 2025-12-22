import cv2
import mediapipe as mp
import csv
import os

LABEL = "G"  # GANTI SETIAP HURUF
SAVE_PATH = f"dataset/landmarks/{LABEL}.csv"

os.makedirs("dataset/landmarks", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

file = open(SAVE_PATH, "a", newline="")
writer = csv.writer(file)

count = 0

print("▶ Fokus ke window kamera")
print("▶ Tekan 'S' untuk simpan | 'Q' untuk keluar")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    ready = False

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        ready = True

        cv2.putText(frame, "READY", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 3)

    cv2.putText(frame, f"Label: {LABEL}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(frame, f"Saved: {count}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    cv2.imshow("Collect Landmark", frame)

    key = cv2.waitKey(30) & 0xFF  # ⬅ lebih stabil

    if key == ord("s") and ready:
        writer.writerow(landmarks)
        count += 1
        print(f"✅ Saved {count}")

    elif key == ord("q"):
        break

file.close()
cap.release()
cv2.destroyAllWindows()
