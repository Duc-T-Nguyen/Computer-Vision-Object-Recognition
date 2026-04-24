"""
test_hoi.py
===========
One script that does everything:

  Step 1 — test detection immediately (no training needed):
    python test_hoi.py --live
    python test_hoi.py --folder

  Step 2 — collect labelled data:
    python data_collector.py --label throwing --clips 20
    python data_collector.py --label catching --clips 20
    python data_collector.py --label holding  --clips 15
    python data_collector.py --label no_interaction --clips 15

  Step 3 — train the HOI head:
    python test_hoi.py --train

  Step 4 — run with trained weights:
    python test_hoi.py --live   --checkpoint checkpoints/hoi_head.pt
    python test_hoi.py --folder --checkpoint checkpoints/hoi_head.pt

"""

import argparse
import glob
import os
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch

from hoi_model import (
    HOIDetector, draw_results, train_hoi_head,
    HOI_CLASSES, HOIDataset,
)


# ─────────────────────────────────────────────────────────────
# FPS counter
# ─────────────────────────────────────────────────────────────
class FPS:
    def __init__(self, n=30):
        self._t = deque(maxlen=n)
    def tick(self):
        self._t.append(time.monotonic())
    @property
    def fps(self):
        if len(self._t) < 2: return 0.0
        return (len(self._t)-1) / (self._t[-1]-self._t[0])


# ─────────────────────────────────────────────────────────────
# Live webcam
# ─────────────────────────────────────────────────────────────
def run_live(detector, save_path=None):
    """
    Mirrors frames_capture.py exactly:
      Camera 0, 650×480, Q to quit.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  650)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera 0. Try changing the device index.")

    writer = None
    if save_path:
        writer = cv2.VideoWriter(save_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 15, (650, 480))

    fps    = FPS()
    events = []
    print("[Live] Running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result    = detector.predict(frame, debug=getattr(args,'debug',False))
        fps.tick()
        annotated = draw_results(frame, result, fps.fps)

        cv2.putText(annotated, "Press Q to quit.",
                    (10, annotated.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.imshow("HOI Live", annotated)
        if writer: writer.write(annotated)

        if result.hoi_class != "no_interaction":
            events.append(result.hoi_class)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    _print_summary(events)


# ─────────────────────────────────────────────────────────────
# Saved frames folder
# ─────────────────────────────────────────────────────────────
def run_folder(detector, folder_path=None, save_path=None):
    """
    Runs on every JPEG inside a testing_frames/frame_folder_vid_X/ folder.
    Auto-picks the most recent folder if no path given.
    Press Q to quit, SPACE to pause.
    """
    if folder_path is None:
        candidates = sorted(glob.glob("testing_frames/frame_folder_vid_*"))
        if not candidates:
            print("[Folder] No frame folders found in ./testing_frames/")
            print("[Folder] Run frames_capture.py first.")
            return
        folder_path = candidates[-1]
        print(f"[Folder] Using most recent: {folder_path}")

    jpgs = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    if not jpgs:
        print(f"[Folder] No .jpg files in {folder_path}")
        return

    print(f"[Folder] {len(jpgs)} frames — Q to quit, SPACE to pause")

    H_fr, W_fr = cv2.imread(jpgs[0]).shape[:2]
    writer = None
    if save_path:
        writer = cv2.VideoWriter(save_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 15, (W_fr, H_fr))

    fps    = FPS()
    events = []
    paused = False

    for i, path in enumerate(jpgs):
        frame = cv2.imread(path)
        if frame is None: continue

        result    = detector.predict(frame, debug=getattr(args,'debug',False))
        fps.tick()
        annotated = draw_results(frame, result, fps.fps)

        cv2.putText(annotated,
                    f"Frame {i+1}/{len(jpgs)}  |  {os.path.basename(path)}",
                    (10, annotated.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

        cv2.imshow("HOI Folder", annotated)
        if writer: writer.write(annotated)

        if result.hoi_class != "no_interaction":
            events.append(result.hoi_class)

        if (i+1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(jpgs)}  "
                  f"last={result.hoi_class}  conf={result.confidence:.2f}")

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'): break
        if key == ord(' '): paused = not paused

    if writer: writer.release()
    cv2.destroyAllWindows()
    _print_summary(events)


def _print_summary(events):
    if not events: return
    total = len(events)
    print(f"\nInteraction summary ({total} active frames):")
    for cls, n in Counter(events).most_common():
        bar = "█" * int(30 * n / total)
        print(f"  {cls:<20} {n:>4}  {bar}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="HOI detector — test / train / live")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--live",   action="store_true", help="Live webcam")
    mode.add_argument("--folder", nargs="?", const=None, metavar="PATH",
                      help="Saved frames folder (omit for most recent)")
    mode.add_argument("--train",  action="store_true",
                      help="Train the HOI head on data/ directory")

    p.add_argument("--checkpoint", default=None,
                   help="HOI head .pt checkpoint path")
    p.add_argument("--data",       default="data/",
                   help="Data root for --train")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=8,  dest="batch_size")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--conf",       type=float, default=0.40,
                   help="YOLO detection confidence threshold")
    # Auto-select best device: MPS (Apple M-series) > CUDA > CPU
    _best_device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    p.add_argument("--device",     default=_best_device,
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--debug",  action="store_true",
                   help="Print YOLO detections every 10 frames")
    p.add_argument("--save",       default=None, help="Save annotated output .mp4")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.train:
        # Train only the HOI head — no detector needed
        train_hoi_head(
            data_root=args.data,
            save_path=args.checkpoint or "checkpoints/hoi_head.pt",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    else:
        # Load detector (YOLO always works, HOI head optional)
        detector = HOIDetector(
            hoi_head_path=args.checkpoint,
            conf=args.conf,
            device=args.device,
        )

        if args.folder is not None:
            run_folder(detector, folder_path=args.folder, save_path=args.save)
        else:
            # Default = live
            run_live(detector, save_path=args.save)