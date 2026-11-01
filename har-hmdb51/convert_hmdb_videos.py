"""Prevedie pôvodné videá HMDB51 (*.avi) tried z CLASSES na priečinky so snímkami.

Výsledná štruktúra: HMDB51/<trieda>/<klip>/<snímka>.jpg (všetky snímky videa).

Predtým (v priečinku projektu), pre každú triedu z CLASSES:
    sudo apt install unrar
    unrar x hmdb51_org.rar <trieda>.rar           # z veľkého archívu len potrebná trieda
    unrar x <trieda>.rar videos/                  # -> videos/<trieda>/*.avi

Spustenie:  python convert_hmdb_videos.py
"""
from pathlib import Path

import cv2

from har.config import CLASSES, DATA_DIR

VIDEOS = Path("videos")       # videos/<trieda>/*.avi

for cls in CLASSES:
    for video in sorted((VIDEOS / cls).glob("*.avi")):
        out_dir = Path(DATA_DIR) / cls / video.stem
        if out_dir.exists():
            continue                                   # už prevedené
        out_dir.mkdir(parents=True)
        cap = cv2.VideoCapture(str(video))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), frame)
            i += 1
        cap.release()
        print(f"{cls}/{video.stem}: {i} snímok")
