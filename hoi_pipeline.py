"""
hoi_pipeline.py
===============
Live webcam pipeline for the OnePassHOINet CNN+Transformer model.
Runs the full model forward pass on each frame from the webcam.

Usage
-----
  # Inference with a trained checkpoint:
  python hoi_pipeline.py --checkpoint checkpoints/hoi_epoch030.pt

  # Train first, then run:
  python hoi_pipeline.py --train --data data/ --epochs 30
  python hoi_pipeline.py --checkpoint checkpoints/hoi_epoch030.pt

  # Run untrained (random weights — for pipeline testing only):
  python hoi_pipeline.py --no-checkpoint

  # CPU mode (no CUDA):
  python hoi_pipeline.py --device cpu --checkpoint checkpoints/hoi_epoch030.pt
"""

import argparse
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hoi_model import (
    OnePassHOINet, HOIDataset, HOITrainer,
    HOI_CLASSES, model_summary,
    _preprocess, _soft_argmax,
    COLORS, SKELETON_EDGES,
)

# Colours (BGR)
COLORS = {
    "throwing":       (0,   0,   220),
    "catching":       (0,   180,  80),
    "holding":        (180, 160,   0),
    "no_interaction": (100, 100, 100),
    "ball":           (0,   165, 255),
    "skeleton":       (180, 180, 180),
    "wrist":          (255,   0, 220),
}

SKELETON_EDGES = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
    (0,5),(0,6),
]

W, H = 640, 480


# ──────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────
def draw_results(frame, pred, fps: float = 0.0) -> np.ndarray:
    """Annotate frame with HOI prediction, skeleton, and score bars."""
    out   = frame.copy()
    cls   = pred.hoi_class
    color = COLORS.get(cls, (180, 180, 180))
    H_fr, W_fr = out.shape[:2]

    # ── Ball bbox ──────────────────────────────────────────────
    if pred.ball_bbox is not None:
        x1,y1,x2,y2 = [int(v) for v in pred.ball_bbox]
        if x2 > x1 and y2 > y1:
            cv2.rectangle(out, (x1,y1), (x2,y2), COLORS["ball"], 2)
            cv2.putText(out, "ball", (x1, max(y1-6,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["ball"], 2, cv2.LINE_AA)

    # ── Person bbox + HOI class label ──────────────────────────
    if pred.person_bbox is not None:
        x1,y1,x2,y2 = [int(v) for v in pred.person_bbox]
        if x2 > x1 and y2 > y1:
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            bg_y = max(y1-22, 0)
            cv2.rectangle(out, (x1, bg_y), (x2, bg_y+22), color, -1)
            cv2.putText(out, f"person [{cls}]", (x1+4, bg_y+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)

    # ── Keypoints from heatmaps ────────────────────────────────
    if pred.heatmaps is not None:
        Hh, Wh = pred.heatmaps.shape[1], pred.heatmaps.shape[2]
        sx = W_fr / Wh;  sy = H_fr / Hh
        kp_coords = []
        for k in range(pred.heatmaps.shape[0]):
            hm = pred.heatmaps[k]
            fy, fx = np.unravel_index(hm.argmax(), hm.shape)
            conf = float(hm.max())
            kp_coords.append((int(fx * sx), int(fy * sy), conf))

        # Draw skeleton edges
        for i, j in SKELETON_EDGES:
            if i < len(kp_coords) and j < len(kp_coords):
                xi, yi, ci = kp_coords[i]
                xj, yj, cj = kp_coords[j]
                if ci > 0.1 and cj > 0.1:
                    cv2.line(out, (xi,yi), (xj,yj),
                             COLORS["skeleton"], 1, cv2.LINE_AA)
        # Draw joints
        for k, (kx, ky, kc) in enumerate(kp_coords):
            if kc > 0.1:
                kcolor = COLORS["wrist"] if k in (9,10) else (80,220,80)
                cv2.circle(out, (kx,ky), 4, kcolor, -1)

    # ── Top banner ─────────────────────────────────────────────
    ov = out.copy()
    cv2.rectangle(ov, (0,0), (W_fr, 58), (15,15,15), -1)
    cv2.addWeighted(ov, 0.72, out, 0.28, 0, out)
    cv2.putText(out, cls.replace("_"," ").upper(),
                (10,34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(out, f"conf {pred.confidence:.0%}  |  FPS {fps:.1f}",
                (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190,190,190), 1, cv2.LINE_AA)

    # ── Score bars ─────────────────────────────────────────────
    bx = W_fr - 170
    for i, (c, v) in enumerate(sorted(pred.scores.items(), key=lambda t: -t[1])):
        by0 = 8 + i*22
        cv2.rectangle(out, (bx,by0), (bx+int(v*148),by0+14),
                       COLORS.get(c,(120,120,120)), -1)
        cv2.putText(out, f"{c:<16} {v:.2f}",
                    (bx-2, by0+11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.33, (220,220,220), 1)

    return out


# ──────────────────────────────────────────────────────────────
# FPS counter
# ──────────────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self, n=30):
        self._t = deque(maxlen=n)
    def tick(self):
        self._t.append(time.monotonic())
    @property
    def fps(self):
        if len(self._t) < 2:
            return 0.0
        return (len(self._t)-1) / (self._t[-1]-self._t[0])


# ──────────────────────────────────────────────────────────────
# Training helper (run from this script with --train)
# ──────────────────────────────────────────────────────────────
def train_model(args):
    print("[Train] Building dataset from:", args.data)
    dataset = HOIDataset(data_root=args.data, pseudo_label=True)
    if len(dataset) == 0:
        print("[Train] No data found! Run data_collector.py first.")
        return

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(args.device == "cuda"),
        collate_fn=_collate,
    )

    model   = OnePassHOINet()
    model_summary(model)
    trainer = HOITrainer(model, lr=args.lr, device=args.device,
                         checkpoint_dir="checkpoints")

    if args.checkpoint:
        try:
            trainer.load(args.checkpoint)
            print("[Train] Resuming from checkpoint.")
        except FileNotFoundError:
            print("[Train] No checkpoint found, training from scratch.")

    trainer.train(loader, epochs=args.epochs)
    print("[Train] Done.")


def _collate(batch):
    """Custom collate to handle variable-size bbox tensors."""
    frames  = torch.stack([b[0] for b in batch])
    targets = {}
    for key in batch[0][1]:
        vals = [b[1][key] for b in batch]
        if vals[0].dim() == 0:
            targets[key] = torch.stack(vals)
        else:
            targets[key] = torch.cat(vals, dim=0)
    return frames, targets


# ──────────────────────────────────────────────────────────────
# Live inference loop
# ──────────────────────────────────────────────────────────────
def run_live(args):
    device = args.device

    model = OnePassHOINet().to(device)
    model_summary(model)

    if args.checkpoint:
        try:
            model.load_state_dict(torch.load(args.checkpoint,
                                             map_location=device))
            print(f"[Pipeline] Loaded checkpoint: {args.checkpoint}")
        except FileNotFoundError:
            print(f"[Pipeline] WARNING: checkpoint not found → random weights")
    model.eval()

    # Warm up
    with torch.no_grad():
        dummy = torch.zeros(1, 3, H, W, device=device)
        model(dummy)
    print("[Pipeline] Warm-up done. Starting camera.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 30, (W, H))

    fps      = FPSCounter()
    events   = []
    fidx     = 0

    print("[Pipeline] Running — q: quit | s: screenshot")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            fidx += 1

            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))

            # ── One forward pass ──────────────────────────────────────────
            with torch.no_grad():
                pred = model.predict(frame)
            # ─────────────────────────────────────────────────────────────

            fps.tick()
            annotated = draw_results(frame, pred, fps.fps)

            cv2.imshow("HOI — CNN+Transformer", annotated)
            if writer:
                writer.write(annotated)

            if pred.hoi_class != "no_interaction":
                events.append(pred.hoi_class)
                if fidx % 15 == 0:
                    print(f"[{fidx:06d}] {pred.hoi_class:<16} "
                          f"conf={pred.confidence:.2f}  "
                          f"scores={{{', '.join(f'{c}:{v:.2f}' for c,v in pred.scores.items())}}}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fn = f"hoi_{fidx:06d}.jpg"
                cv2.imwrite(fn, annotated)
                print(f"[Pipeline] Screenshot → {fn}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"\n[Pipeline] Done. frames={fidx}  events={len(events)}")
        if events:
            print("[Pipeline] Interaction breakdown:")
            for lbl, n in Counter(events).most_common():
                print(f"  {lbl:<20} {n:>5} frames")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="OnePassHOINet — CNN+Transformer HOI on webcam")
    p.add_argument("--train",       action="store_true",
                   help="Train the model before running inference")
    p.add_argument("--data",        default="data/",
                   help="Data directory from data_collector.py")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=4,  dest="batch_size")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--checkpoint",  default=None,
                   help="Path to .pt checkpoint file")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Run with random weights (testing only)")
    p.add_argument("--camera",      type=int,   default=0)
    p.add_argument("--device",      default="cuda", choices=["cuda","cpu"])
    p.add_argument("--save",        default=None,
                   help="Save annotated output to .mp4")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.train:
        train_model(args)
        # After training, set checkpoint to the last saved one
        import glob, os
        ckpts = sorted(glob.glob("checkpoints/hoi_epoch*.pt"))
        if ckpts and not args.checkpoint:
            args.checkpoint = ckpts[-1]

    if not args.no_checkpoint and args.checkpoint is None:
        print("[Pipeline] No --checkpoint given. "
              "Use --train first, or --no-checkpoint for random weights.")
        print("  Example: python hoi_pipeline.py --train --data data/ --epochs 30")
        print("  Then:    python hoi_pipeline.py --checkpoint checkpoints/hoi_epoch030.pt")
    else:
        run_live(args)