import cv2
import mediapipe as mp
import csv
import os

LABEL = "C"  # GANTI HURUF SETIAP KALI
SAVE_PATH = f"dataset/landmarks/{LABEL}.csv"

os.makedirs("dataset/landmarks", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

file = open(SAVE_PATH, "a", newline="")
writer = csv.writer(file)

print("Tekan 's' untuk simpan, 'q' untuk keluar")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            cv2.putText(frame, LABEL, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Collect Landmark", frame)
    key = cv2.waitKey(1)

    if key == ord("s") and result.multi_hand_landmarks:
        writer.writerow(landmarks)
        print("Saved")

    if key == ord("q"):
        break

file.close()
cap.release()
cv2.destroyAllWindows()
