"""
data_collector.py
=================
Data collection tool for HOI training data (proposal §3).

Records labelled video clips or individual frames from the webcam,
organised by interaction type. Supports all interaction classes:
  throwing_overhand | throwing_underhand | kicking |
  catching | holding | no_interaction

Usage
-----
  python data_collector.py                # interactive session
  python data_collector.py --label throwing --clips 10
  python data_collector.py --frames-only  # save individual frames, not clips

Output structure:
  data/
    throwing_overhand/
      clip_001.mp4
      clip_001_frames/
        frame_0001.jpg  ...
    throwing_underhand/
      ...
    metadata.json
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from hoi_model import HOI_CLASSES

DATA_ROOT = Path("data")
W, H      = 640, 480
FPS       = 30
CLIP_SECS = 3      # seconds per clip


def ensure_dirs():
    for lbl in HOI_CLASSES:
        (DATA_ROOT / lbl).mkdir(parents=True, exist_ok=True)


def count_existing(label: str) -> int:
    d = DATA_ROOT / label
    return len(list(d.glob("clip_*.mp4")))


def record_clip(
    cap: cv2.VideoCapture,
    label: str,
    clip_id: int,
    frames_only: bool = False,
) -> dict:
    """Record one clip or frame burst for the given label."""
    clip_dir  = DATA_ROOT / label
    clip_path = clip_dir / f"clip_{clip_id:04d}.mp4"
    frames_dir = clip_dir / f"clip_{clip_id:04d}_frames"
    frames_dir.mkdir(exist_ok=True)

    writer = None
    if not frames_only:
        writer = cv2.VideoWriter(
            str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
        )

    n_frames   = int(CLIP_SECS * FPS)
    saved      = 0
    timestamps = []

    print(f"\n  [Rec] Recording '{label}' clip {clip_id:04d} "
          f"({CLIP_SECS}s / {n_frames} frames)…")

    # Countdown
    for i in range(3, 0, -1):
        ok, f = cap.read()
        if ok and f is not None:
            f = cv2.resize(f, (W, H))
            cv2.putText(f, f"Starting in {i}…", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
            cv2.putText(f, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Data Collector", f)
        cv2.waitKey(1000)

    t_start = time.monotonic()

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.resize(frame, (W, H))

        # Progress overlay
        progress = fi / n_frames
        bar_w    = int(progress * W)
        display  = frame.copy()
        cv2.rectangle(display, (0, H-8), (bar_w, H), (0,200,0), -1)
        cv2.putText(display,
                    f"RECORDING: {label}  [{fi+1}/{n_frames}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        cv2.imshow("Data Collector", display)

        # Save
        frame_path = frames_dir / f"frame_{fi:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        if writer:
            writer.write(frame)

        timestamps.append(time.monotonic() - t_start)
        saved += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if writer:
        writer.release()

    meta = {
        "label":       label,
        "clip_id":     clip_id,
        "clip_path":   str(clip_path) if not frames_only else None,
        "frames_dir":  str(frames_dir),
        "n_frames":    saved,
        "duration_s":  round(time.monotonic() - t_start, 2),
        "recorded_at": datetime.utcnow().isoformat(),
        "resolution":  [W, H],
        "fps":         FPS,
    }
    print(f"  [Rec] Saved {saved} frames → {frames_dir}")
    return meta


def interactive_session(args):
    ensure_dirs()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    metadata_path = DATA_ROOT / "metadata.json"
    all_meta: list = []
    if metadata_path.exists():
        with open(metadata_path) as f:
            all_meta = json.load(f)

    print("\n╔══════════════════════════════════════════╗")
    print("║       HOI Data Collector                 ║")
    print("╠══════════════════════════════════════════╣")
    for lbl in HOI_CLASSES:
        n = count_existing(lbl)
        print(f"║  {lbl:<28} {n:>4} clips ║")
    print("╚══════════════════════════════════════════╝")

    if args.label:
        labels_to_collect = [args.label] * args.clips
    else:
        labels_to_collect = None   # interactive prompt

    try:
        while True:
            # Live preview
            ok, frame = cap.read()
            if ok and frame is not None:
                frame = cv2.resize(frame, (W, H))
                cv2.putText(frame, "HOI Data Collector", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(frame, "Press SPACE to record | q to quit",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
                for i, lbl in enumerate(HOI_CLASSES):
                    n = count_existing(lbl)
                    cv2.putText(frame, f"  {lbl}: {n}", (10, 90 + i*20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
                cv2.imshow("Data Collector", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                if labels_to_collect:
                    lbl = labels_to_collect.pop(0)
                    if not labels_to_collect:
                        labels_to_collect = None
                else:
                    print("\nAvailable labels:")
                    for i, l in enumerate(HOI_CLASSES):
                        print(f"  [{i}] {l}")
                    choice = input("Select label (number or name): ").strip()
                    if choice.isdigit():
                        idx = int(choice)
                        lbl = HOI_CLASSES[idx] if 0 <= idx < len(HOI_CLASSES) else None
                    else:
                        lbl = choice if choice in HOI_CLASSES else None

                    if lbl is None:
                        print("  Invalid label, skipping.")
                        continue

                clip_id = count_existing(lbl) + 1
                meta    = record_clip(cap, lbl, clip_id, args.frames_only)
                all_meta.append(meta)

                with open(metadata_path, "w") as f:
                    json.dump(all_meta, f, indent=2)

                print(f"\n  [Saved] metadata.json updated ({len(all_meta)} total clips)")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[Collector] Session complete. Total clips: {len(all_meta)}")
        _print_summary(all_meta)


def _print_summary(meta: list):
    from collections import Counter
    counts = Counter(m["label"] for m in meta)
    total_frames = sum(m.get("n_frames", 0) for m in meta)
    print("\n── Collection Summary ─────────────────────")
    for lbl in HOI_CLASSES:
        n = counts.get(lbl, 0)
        bar = "█" * n
        print(f"  {lbl:<28} {n:>3} clips  {bar}")
    print(f"  Total frames across all clips: {total_frames}")
    print("───────────────────────────────────────────")


def parse_args():
    p = argparse.ArgumentParser(description="HOI Data Collector")
    p.add_argument("--camera",       type=int,  default=0)
    p.add_argument("--label",        type=str,  default=None,
                   choices=HOI_CLASSES, help="Auto-record this label")
    p.add_argument("--clips",        type=int,  default=5,
                   help="Number of clips to record when --label is set")
    p.add_argument("--frames-only",  action="store_true",
                   help="Save frames only, no .mp4")
    return p.parse_args()


if __name__ == "__main__":
    interactive_session(parse_args())