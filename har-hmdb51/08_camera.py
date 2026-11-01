"""8. Rozpoznávanie činností z kamery alebo videa v reálnom čase.

Model: R(2+1)D (príznaky klipu) + SVM uložený v notebooku 6 (models/svm_r21d.joblib).

Postup:
  - snímky z posledných --window-sec sekúnd sa držia v pamäti,
  - každých --every-sec sekúnd sa z nich rovnomerne vyberie 16 snímok (ako pri trénovaní),
  - snímky prejdú cez R(2+1)D -> 512 príznakov -> SVM -> skóre,
  - skóre sa spriemeruje s poslednými predikciami; ak je blízko nuly,
    zobrazí sa „neiste“ (model pozná iba triedy z CLASSES).

Spustenie:
  python 08_camera.py                         # webkamera
  python 08_camera.py --source video.mp4      # videosúbor
  python 08_camera.py --source video.mp4 --out vystup.mp4 --no-window
Ukončenie: klávesa q.
"""
import argparse
import time
from collections import deque

import cv2
import joblib
import numpy as np
import torch
from torch import nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

from har.config import CLASSES, N_FRAMES, SIZE
from har.data import sample_frames, to_video_tensor

COLORS = [(0, 200, 0), (0, 140, 255)]          # farba textu pre triedu 0 a 1 (BGR)
GRAY = (200, 200, 200)

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0", help="číslo kamery (0) alebo cesta k videu")
parser.add_argument("--model", default="models/svm_r21d.joblib")
parser.add_argument("--window-sec", type=float, default=3.0, help="dĺžka analyzovaného úseku (s)")
parser.add_argument("--every-sec", type=float, default=0.5, help="ako často predikovať (s)")
parser.add_argument("--threshold", type=float, default=0.5, help="|skóre| pod touto hodnotou = neisté")
parser.add_argument("--smooth", type=int, default=3, help="počet predikcií na priemerovanie")
parser.add_argument("--out", default=None, help="uložiť video s výsledkom")
parser.add_argument("--no-window", action="store_true", help="nezobrazovať okno")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# rovnaká sieť a predspracovanie ako v notebooku 3
weights = R2Plus1D_18_Weights.KINETICS400_V1
backbone = r2plus1d_18(weights=weights)
backbone.fc = nn.Identity()
backbone.eval().to(device)
video_transform = weights.transforms()
svm = joblib.load(args.model)

is_camera = args.source.isdigit()
cap = cv2.VideoCapture(int(args.source) if is_camera else args.source)
if not cap.isOpened():
    raise SystemExit(f"Nepodarilo sa otvoriť zdroj: {args.source}")
if is_camera:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # čítať najnovšiu snímku, nie staré z fronty
fps = cap.get(cv2.CAP_PROP_FPS) or 30          # kamera niekedy vráti 0

buffer = deque()                               # (čas v sekundách, zmenšená RGB snímka)
scores = deque(maxlen=args.smooth)
writer = None
text, color = "...", GRAY
frame_no = 0
last_pred = -np.inf

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1
    # čas snímky: pri kamere hodiny, pri videosúbore poradie snímky / fps
    t = time.monotonic() if is_camera else frame_no / fps
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (SIZE, SIZE))
    buffer.append((t, small))
    while t - buffer[0][0] > args.window_sec:
        buffer.popleft()

    buffer_full = t - buffer[0][0] >= 0.9 * args.window_sec and len(buffer) >= N_FRAMES
    if buffer_full and t - last_pred >= args.every_sec:
        last_pred = t
        clip = sample_frames([f for _, f in buffer])              # (16, 224, 224, 3)
        with torch.no_grad():
            x = to_video_tensor(clip, video_transform).to(device)
            feat = backbone(x).cpu().numpy()                      # (1, 512)
        scores.append(svm.decision_function(feat)[0])             # > 0 -> CLASSES[1], < 0 -> CLASSES[0]
        s = float(np.mean(scores))
        if abs(s) < args.threshold:
            text, color = f"neiste ({s:+.2f})", GRAY
        else:
            label = int(s > 0)
            text, color = f"{CLASSES[label]} ({s:+.2f})", COLORS[label]

    cv2.rectangle(frame, (0, 0), (330, 50), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if args.out:
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
    if not args.no_window:
        cv2.imshow(f"HAR: {' / '.join(CLASSES)}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
if writer:
    writer.release()
if not args.no_window:
    cv2.destroyAllWindows()
