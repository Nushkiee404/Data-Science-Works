import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import imageio
import time
import os

# ========== INIT ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
minVol, maxVol = vol_range[0], vol_range[1]

# Create screenshots folder if not exists
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# GIF settings
frames = []
gif_duration_sec = 5
fps = 10
max_frames = gif_duration_sec * fps

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Width for GitHub-friendly resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height

prev_brightness = sbc.get_brightness(display=0)[0]
prev_vol_perc = 0
frame_count = 0
start_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = img.shape
    vol_bar = 400
    vol_perc = prev_vol_perc
    brightness_val = prev_brightness

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Hand center for label positioning
            x_coords = [pt[0] for pt in lmList]
            y_coords = [pt[1] for pt in lmList]
            cx, cy = int(np.mean(x_coords)), int(np.min(y_coords)) - 20

            if len(lmList) >= 21:
                # ---------- RIGHT HAND: Volume ----------
                if hand_label == "Right":
                    x1, y1 = lmList[4]
                    x2, y2 = lmList[8]
                    length = np.hypot(x2 - x1, y2 - y1)
                    vol = np.interp(length, [20, 200], [minVol, maxVol])
                    volume.SetMasterVolumeLevel(vol, None)
                    vol_perc = np.interp(length, [20, 200], [0, 100])
                    vol_bar = np.interp(length, [20, 200], [400, 150])

                    # Dynamic label color
                    color = (0, 0, 255) if abs(prev_vol_perc - vol_perc) > 1 else (255, 255, 255)
                    prev_vol_perc = vol_perc

                    cv2.putText(img, "Right Hand - Volume", (cx - 100, cy),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

                # ---------- LEFT HAND: Brightness ----------
                elif hand_label == "Left":
                    finger_tips = [8, 12, 16, 20]
                    fingers_open = 0
                    for tip in finger_tips:
                        if lmList[tip][1] < lmList[tip - 2][1]:
                            fingers_open += 1

                    if fingers_open >= 3:
                        brightness_val = min(prev_brightness + 2, 100)
                        sbc.set_brightness(brightness_val)
                    elif fingers_open == 0:
                        brightness_val = max(prev_brightness - 2, 0)
                        sbc.set_brightness(brightness_val)
                    prev_brightness = brightness_val

                    # Dynamic label color
                    color = (0, 0, 255) if fingers_open >= 3 or fingers_open == 0 else (255, 255, 255)

                    cv2.putText(img, "Left Hand - Brightness", (cx - 120, cy),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

    # ---------- Volume Bar ----------
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'Vol: {int(vol_perc)}%', (40, 450),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # ---------- Brightness Bar ----------
    cv2.rectangle(img, (550, 150), (585, 400), (255, 255, 0), 3)
    bright_bar = np.interp(brightness_val, [0, 100], [400, 150])
    cv2.rectangle(img, (550, int(bright_bar)), (585, 400), (255, 255, 0), cv2.FILLED)
    cv2.putText(img, f'Bright: {int(brightness_val)}%', (520, 450),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)

    # ---------- Display ----------
    cv2.imshow("Gesture Volume + Brightness Control", img)

    # Save frames for GIF (up to max_frames)
    if frame_count < max_frames:
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        frame_count += 1

    # ---------- Key Handling ----------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

# ---------- Save GIF ----------
if frames:
    gif_path = "screenshots/demo.gif"
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"Demo GIF saved: {gif_path}")

