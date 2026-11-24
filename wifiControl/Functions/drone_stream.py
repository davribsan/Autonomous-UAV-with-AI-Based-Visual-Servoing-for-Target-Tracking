import cv2
import os

OUT_A     = "/tmp/droneA.jpg"
OUT_B     = "/tmp/droneB.jpg"
TMP_A     = "/tmp/droneA_tmp.jpg"
TMP_B     = "/tmp/droneB_tmp.jpg"
SLOT_FILE = "/tmp/drone_slot.txt"

cap = cv2.VideoCapture("tcp://192.168.1.1:5555")
if not cap.isOpened():
    print("Error: no se pudo conectar al dron")
    exit(1)

slot = 'A'
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if slot == 'A':
            cv2.imwrite(TMP_A, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            os.replace(TMP_A, OUT_A)          # rename atómico
            with open(SLOT_FILE, 'w') as f: f.write('A')
            slot = 'B'
        else:
            cv2.imwrite(TMP_B, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            os.replace(TMP_B, OUT_B)          # rename atómico
            with open(SLOT_FILE, 'w') as f: f.write('B')
            slot = 'A'
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    print("Stream cerrado.")