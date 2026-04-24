"""
hoi_model.py
============
Hybrid One-Pass HOI Architecture
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════
NUM_KP      = 17
HOI_CLASSES = ["throwing", "catching", "holding", "no_interaction"]
NUM_HOI     = len(HOI_CLASSES)
HOI_IDX     = {c: i for i, c in enumerate(HOI_CLASSES)}

PERSON_CLS    = 0
# All COCO classes that are commonly held in hand.
# remote=65 was missing — that's why the remote wasn't detected as an object.
OBJECT_CLASSES = [
    32,   # sports ball
    39,   # bottle
    41,   # cup
    45,   # bowl
    46,   # banana
    47,   # apple
    49,   # orange
    63,   # laptop
    64,   # mouse
    65,   # remote  ← THIS was missing (your remote control)
    66,   # keyboard
    67,   # cell phone
    73,   # book
    74,   # clock
    76,   # scissors
    77,   # teddy bear
    78,   # hair drier
    79,   # toothbrush
]
BALL_CLS      = 32   # kept for legacy references

# Two separate thresholds:
#   PERSON_CONF — keep high so random background people don't trigger
#   OBJECT_CONF — lower so partially-visible / small handheld objects fire
PERSON_CONF   = 0.40
OBJECT_CONF   = 0.15

CROP_SIZE   = 64
D_MODEL     = 128

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


# ══════════════════════════════════════════════════════════════
# HOI result container
# ══════════════════════════════════════════════════════════════
@dataclass
class HOIResult:
    hoi_class:    str   = "no_interaction"
    confidence:   float = 0.0
    scores:       dict  = field(default_factory=dict)
    person_bbox:  Optional[np.ndarray] = None
    ball_bbox:    Optional[np.ndarray] = None
    keypoints:    Optional[np.ndarray] = None
    frame_idx:    int   = 0

    def __str__(self):
        sc = " ".join(f"{c}:{v:.2f}" for c,v in self.scores.items())
        return f"[{self.frame_idx:05d}] {self.hoi_class:<16} conf={self.confidence:.2f}  {sc}"


# ══════════════════════════════════════════════════════════════
# Temporal smoother
# ══════════════════════════════════════════════════════════════
class TemporalSmoother:
    def __init__(self, alpha: float = 0.45):
        self.alpha    = alpha
        self.smoothed = {c: 1.0 / NUM_HOI for c in HOI_CLASSES}

    def update(self, scores: dict) -> dict:
        for c in HOI_CLASSES:
            self.smoothed[c] = (self.alpha * scores.get(c, 0.0)
                                + (1 - self.alpha) * self.smoothed[c])
        t = sum(self.smoothed.values()) + 1e-9
        return {c: v / t for c, v in self.smoothed.items()}

    def reset(self):
        self.smoothed = {c: 1.0 / NUM_HOI for c in HOI_CLASSES}



# ══════════════════════════════════════════════════════════════
# Wrist velocity tracker  (holding vs catching vs throwing)
# ══════════════════════════════════════════════════════════════
class WristVelocityTracker:
    """
    Tracks wrist position across the last N frames and computes
    velocity (speed + direction) as two additional spatial features.

    This is the critical signal that static pose can't provide:
      holding   : wrist velocity ≈ 0  (stationary)
      catching  : wrist moving TOWARD object (negative wrist-obj velocity)
      throwing  : wrist moving AWAY from object (positive wrist-obj velocity)
                  followed by rapid deceleration after release
    """
    def __init__(self, history: int = 6):
        self._hist = []          # [(wx, wy, timestamp), ...]
        self._maxlen = history

    def update(self, kps: np.ndarray, W: int, H: int) -> np.ndarray:
        """
        Update with new keypoints, return 2-dim velocity feature:
          [speed_norm, wrist_direction]
          speed_norm      : wrist speed normalised 0-1 (saturates at 200px/s)
          wrist_direction : +1 = moving up/forward (throw), -1 = moving down (catch)
        """
        import time
        best_wrist = None
        best_c     = -1.0
        for idx in (9, 10):
            wx, wy, wc = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
            if wc > best_c:
                best_c     = wc
                best_wrist = (wx / (W + 1e-6), wy / (H + 1e-6))

        if best_wrist is not None and best_c > 0.15:
            self._hist.append((best_wrist[0], best_wrist[1], time.monotonic()))
            if len(self._hist) > self._maxlen:
                self._hist.pop(0)

        return self._compute()

    def _compute(self) -> np.ndarray:
        if len(self._hist) < 2:
            return np.zeros(2, dtype=np.float32)
        x1, y1, t1 = self._hist[0]
        x2, y2, t2 = self._hist[-1]
        dt = max(t2 - t1, 1e-3)
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        speed = math.hypot(vx, vy)
        speed_norm = min(1.0, speed / 2.0)   # 2.0 = normalised saturate value

        # Direction: negative vy = moving up (throw/raise), positive = moving down
        # Encode as signed value in [-1, 1]
        direction = float(np.clip(-vy / (speed + 1e-6), -1.0, 1.0)) if speed > 0.05 else 0.0

        return np.array([speed_norm, direction], dtype=np.float32)

    def reset(self):
        self._hist = []


# ══════════════════════════════════════════════════════════════
# HOI Head
# ══════════════════════════════════════════════════════════════

class CropCNN(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.net(x)


class PoseMLP(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_KP * 3, 128), nn.ReLU(True),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.net(x)


class SpatialMLP(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(True),   # 20-dim: +wrist velocity speed + direction + object aspect ratio + wrist in object
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.net(x)


class HOIHead(nn.Module):
    """
    Pose + spatial geometry only — no image crop CNNs.

    WHY we removed CropCNN:
      The person/ball crops contain your face, shirt colour, and room.
      That's exactly what was memorised at 99% train accuracy.
      Pose keypoints and spatial geometry are person/camera/lighting invariant.

    Two tokens → Transformer → classifier:
      token_0  pose     (17 keypoints → PoseMLP → 128-d)
      token_1  spatial  (14-d geometry → SpatialMLP → 128-d)
    """
    def __init__(self, d_model: int = D_MODEL, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.pose_enc    = PoseMLP(out_dim=d_model)
        self.spatial_enc = nn.Sequential(
            SpatialMLP(out_dim=d_model // 2),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.2, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, NUM_HOI),
        )

    def forward(self, person_crop, ball_crop, pose_feat, spatial_feat):
        # person_crop and ball_crop accepted but unused —
        # kept so HOIDataset and predict() need no signature changes
        t0 = self.pose_enc(pose_feat)
        t1 = self.spatial_enc(spatial_feat)
        tokens = torch.stack([t0, t1], dim=1)   # (B, 2, 128)
        tokens = self.transformer(tokens)
        fused  = tokens.reshape(tokens.shape[0], -1)   # (B, 256)
        return self.classifier(fused)



# ══════════════════════════════════════════════════════════════
# Full HOI detector
# ══════════════════════════════════════════════════════════════

class HOIDetector:
    def __init__(
        self,
        hoi_head_path: Optional[str] = None,
        yolo_path:     str = "yolov8n-pose.pt",
        device:        str = "cuda" if torch.cuda.is_available() else "cpu",
        conf:          float = 0.40,
        smooth_alpha:  float = 0.45,
    ):
        from ultralytics import YOLO

        self.device = device
        self.conf   = conf

        # Pose model — detects person + keypoints
        print(f"[HOI] Loading pose model: {yolo_path}")
        self.yolo_pose = YOLO(yolo_path)

        # Detection model — detects objects (no class filter, catches everything)
        print(f"[HOI] Loading detection model: yolov8n.pt")
        self.yolo_det  = YOLO("yolov8n.pt")

        # Warm up both models
        _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.yolo_pose(_dummy, classes=[PERSON_CLS], conf=PERSON_CONF,
                       verbose=False, device=device)
        self.yolo_det(_dummy,  conf=OBJECT_CONF,
                      verbose=False, device=device)
        # Keep self.yolo pointing to pose model for any legacy references
        self.yolo = self.yolo_pose
        print("[HOI] Both models warmed up.")

        self.head = HOIHead().to(device)

        if hoi_head_path:
            self._load_head(hoi_head_path)
        else:
            print("[HOI] No HOI head weights given — head has random weights.")
            print("[HOI] Detection + pose will work correctly (YOLO is pretrained).")
            print("[HOI] HOI classification will be wrong until you train.")
            print("[HOI] To train:  python test_hoi.py --train --data data/\n")

        self.head.eval()
        self.smoother  = TemporalSmoother(smooth_alpha)
        self.wrist_vel = WristVelocityTracker(history=6)
        self._fidx     = 0
        print(f"[HOI] Ready on {device}.")

    def predict(self, frame_bgr: np.ndarray, debug: bool = False) -> HOIResult:
        self._fidx += 1
        H, W = frame_bgr.shape[:2]

        # ── Pass 1: pose model → person bboxes + keypoints ────────────────
        pose_results = self.yolo_pose(
            frame_bgr,
            classes=[PERSON_CLS],
            conf=PERSON_CONF,
            imgsz=480,       # slightly reduced → faster pose inference
            verbose=False,
            device=self.device,
        )

        # ── Pass 2: detection model → handheld objects only ──────────────
        # Filter to OBJECT_CLASSES so background furniture never fires.
        det_results = self.yolo_det(
            frame_bgr,
            classes=OBJECT_CLASSES,
            conf=OBJECT_CONF,
            imgsz=480,       # half resolution → 2x faster, still accurate for objects
            verbose=False,
            device=self.device,
        )

        if debug and self._fidx % 10 == 0:
            print(f"\n--- Frame {self._fidx} ---")
            r0 = det_results[0]
            if r0.boxes is not None and len(r0.boxes):
                for box in r0.boxes:
                    cid  = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = self.yolo_det.names.get(cid, str(cid))
                    print(f"  det cls={cid:3d} ({name:<20}) conf={conf:.2f}")
            else:
                print("  det: (no detections)")
            print(f"  OBJECT_CONF threshold: {OBJECT_CONF}")

        persons, ball_bbox = self._parse_two_results(
            pose_results[0], det_results[0], H, W)

        if ball_bbox is None or not persons:
            empty = {c: (1.0 if c=="no_interaction" else 0.0) for c in HOI_CLASSES}
            sm    = self.smoother.update(empty)
            pb    = persons[0]["bbox"] if persons else None
            kp    = persons[0]["kps"]  if persons else None
            return HOIResult(hoi_class="no_interaction",
                             confidence=sm["no_interaction"],
                             scores=sm,
                             person_bbox=pb, ball_bbox=ball_bbox,
                             keypoints=kp, frame_idx=self._fidx)

        bcx = (ball_bbox[0] + ball_bbox[2]) / 2
        bcy = (ball_bbox[1] + ball_bbox[3]) / 2
        person = min(persons, key=lambda p: _box_dist(p["bbox"], bcx, bcy))
        pbox   = person["bbox"]
        kps    = person["kps"]

        person_crop  = _crop_and_resize(frame_bgr, pbox, CROP_SIZE)
        ball_crop    = _crop_and_resize(frame_bgr, ball_bbox, CROP_SIZE)
        pose_feat    = _encode_pose(kps, W, H)
        vel_feat     = self.wrist_vel.update(kps, W, H)   # (2,) velocity features
        spatial_feat = _encode_spatial(pbox, ball_bbox, W, H, kps=kps,
                                       vel_feat=vel_feat)

        with torch.no_grad():
            # _to_tensor returns (3,H,W); add batch dim for single-frame inference
            pc = _to_tensor(person_crop, self.device).unsqueeze(0)   # (1, 3, 64, 64)
            bc = _to_tensor(ball_crop,   self.device).unsqueeze(0)   # (1, 3, 64, 64)
            pf = torch.tensor(pose_feat,    dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 51)
            sf = torch.tensor(spatial_feat, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 10)

            logits = self.head(pc, bc, pf, sf)
            probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

        frame_scores = {c: float(probs[i]) for i, c in enumerate(HOI_CLASSES)}
        smoothed     = self.smoother.update(frame_scores)
        best         = max(smoothed, key=smoothed.get)

        return HOIResult(
            hoi_class=best,
            confidence=round(smoothed[best], 3),
            scores=smoothed,
            person_bbox=pbox,
            ball_bbox=ball_bbox,
            keypoints=kps,
            frame_idx=self._fidx,
        )

    def train_mode(self):
        self.head.train()

    def eval_mode(self):
        self.head.eval()

    def save_head(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.head.state_dict(), path)
        print(f"[HOI] HOI head saved → {path}")

    def _load_head(self, path: str):
        try:
            self.head.load_state_dict(torch.load(path, map_location=self.device))
            print(f"[HOI] HOI head loaded ← {path}")
        except Exception as e:
            print(f"[HOI] Could not load HOI head: {e}")

    def _parse_two_results(self, pose_r, det_r, H, W):
        """
        Parse results from two separate YOLO calls:
          pose_r  — from yolo_pose (persons + keypoints)
          det_r   — from yolo_det  (all object detections, no class filter)
        """
        # ── Extract persons from pose result ────────────────────────────────
        persons  = []
        kp_data  = pose_r.keypoints.data if pose_r.keypoints is not None else None
        kp_cursor = 0
        if pose_r.boxes is not None:
            for box in pose_r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = np.array(box.xyxy[0].tolist(), dtype=np.float32)
                if cls == PERSON_CLS and conf >= PERSON_CONF:
                    kps = (kp_data[kp_cursor].cpu().numpy()
                           if kp_data is not None and kp_cursor < len(kp_data)
                           else np.zeros((17, 3), dtype=np.float32))
                    persons.append({"bbox": bbox, "kps": kps})
                kp_cursor += 1

        # ── Extract objects from detection result (everything non-person) ───
        objects = []
        if det_r.boxes is not None:
            for box in det_r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = np.array(box.xyxy[0].tolist(), dtype=np.float32)
                if cls != PERSON_CLS:   # exclude person detections
                    objects.append({"bbox": bbox, "conf": conf, "cls": cls})

        # ── Pick best object: closest wrist, within proximity limit ────────
        # MAX_WRIST_DIST: object must be within 35% of frame width from a wrist.
        # This prevents background chairs/furniture being picked up.
        MAX_WRIST_DIST = W * 0.55

        best_obj = None
        if objects and persons:
            kps    = persons[0]["kps"]
            wrists = [(float(kps[i][0]), float(kps[i][1]))
                      for i in (9, 10) if kps[i][2] > 0.10]

            if wrists:
                def wdist(obj):
                    ocx = (obj["bbox"][0] + obj["bbox"][2]) / 2
                    ocy = (obj["bbox"][1] + obj["bbox"][3]) / 2
                    return min(math.hypot(ocx-wx, ocy-wy) for wx, wy in wrists)

                # Objects within wrist distance limit
                nearby = [o for o in objects if wdist(o) < MAX_WRIST_DIST]

                if nearby:
                    best_obj = min(nearby, key=wdist)["bbox"]
                else:
                    # Nothing near wrists — try nearest object to person bbox center
                    px1,py1,px2,py2 = persons[0]["bbox"]
                    pcx = (px1+px2)/2; pcy = (py1+py2)/2
                    def pdist(obj):
                        ocx = (obj["bbox"][0]+obj["bbox"][2])/2
                        ocy = (obj["bbox"][1]+obj["bbox"][3])/2
                        return math.hypot(ocx-pcx, ocy-pcy)
                    # Only grab if within the person bbox bounds
                    inside = [o for o in objects
                              if px1 < (o["bbox"][0]+o["bbox"][2])/2 < px2
                              and py1 < (o["bbox"][1]+o["bbox"][3])/2 < py2]
                    if inside:
                        best_obj = min(inside, key=pdist)["bbox"]
            else:
                # No wrist keypoints — take highest-conf known object inside person bbox
                px1,py1,px2,py2 = persons[0]["bbox"]
                inside = [o for o in objects
                          if o["cls"] in OBJECT_CLASSES
                          and px1 < (o["bbox"][0]+o["bbox"][2])/2 < px2
                          and py1 < (o["bbox"][1]+o["bbox"][3])/2 < py2]
                if inside:
                    best_obj = max(inside, key=lambda o: o["conf"])["bbox"]
                elif objects:
                    best_obj = max(
                        [o for o in objects if o["cls"] in OBJECT_CLASSES] or objects,
                        key=lambda o: o["conf"]
                    )["bbox"]

        elif objects:
            # No person — just take highest-conf known object
            known = [o for o in objects if o["cls"] in OBJECT_CLASSES]
            if known:
                best_obj = max(known, key=lambda o: o["conf"])["bbox"]

        return persons, best_obj

    def _parse_yolo(self, r, H, W):
        """
        Parse YOLO result into persons and best object bbox.

        Two-threshold strategy:
          - Persons  kept at PERSON_CONF (0.40)
          - Objects  kept at OBJECT_CONF (0.25) — catches small/partial objects
            in hand that YOLO is less confident about

        Object selection: pick the object closest to either wrist.
        No wrist fallback — if nothing detected return None so caller
        outputs no_interaction cleanly instead of guessing.
        """
        persons   = []
        objects   = []
        kp_data   = r.keypoints.data if r.keypoints is not None else None
        kp_cursor = 0

        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = np.array(box.xyxy[0].tolist(), dtype=np.float32)

            if cls == PERSON_CLS:
                # still need to advance kp_cursor even if below threshold
                kps = (kp_data[kp_cursor].cpu().numpy()
                       if kp_data is not None and kp_cursor < len(kp_data)
                       else np.zeros((17, 3), dtype=np.float32))
                kp_cursor += 1
                if conf >= PERSON_CONF:
                    persons.append({"bbox": bbox, "kps": kps})

            elif cls in OBJECT_CLASSES:
                # lower threshold so partially-visible handheld objects fire
                if conf >= OBJECT_CONF:
                    objects.append({"bbox": bbox, "conf": conf, "cls": cls})

        best_obj_bbox = None

        if objects and persons:
            kps    = persons[0]["kps"]
            wrists = [(float(kps[i][0]), float(kps[i][1]))
                      for i in (9, 10) if kps[i][2] > 0.2]

            if wrists:
                def wrist_dist(obj):
                    ocx = (obj["bbox"][0] + obj["bbox"][2]) / 2
                    ocy = (obj["bbox"][1] + obj["bbox"][3]) / 2
                    return min(math.hypot(ocx - wx, ocy - wy)
                               for wx, wy in wrists)
                best_obj_bbox = min(objects, key=wrist_dist)["bbox"]
            else:
                best_obj_bbox = max(objects, key=lambda o: o["conf"])["bbox"]

        elif objects:
            best_obj_bbox = max(objects, key=lambda o: o["conf"])["bbox"]

        # No fallback — return None if nothing detected so predict()
        # returns no_interaction instead of running on garbage data
        return persons, best_obj_bbox


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

class HOILoss(nn.Module):
    """
    Weighted cross-entropy with label smoothing.
    Label smoothing (0.15) prevents the model from becoming overconfident —
    which is exactly the cause of the 100% train accuracy / 100% no_interaction
    at inference problem. It forces the model to keep small probability mass
    on non-predicted classes, so it generalises instead of memorising.
    """
    def __init__(self):
        super().__init__()
        # Order matches HOI_CLASSES: throwing, catching, holding, no_interaction
        # throwing=2.0  catching=0.7  holding=2.5  no_interaction=2.0
        weights = torch.tensor([2.0, 0.7, 2.5, 2.0], dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    def forward(self, logits, labels):
        return self.ce(logits, labels)


class HOIDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str = "data",
                 yolo_path: str = "yolov8n-pose.pt",
                 device: str = "cpu",
                 conf: float = 0.35):
        from ultralytics import YOLO
        from pathlib import Path

        self.device = device
        self.conf   = conf
        self.yolo   = YOLO(yolo_path)

        self.samples = []
        for lbl_idx, label in enumerate(HOI_CLASSES):
            lbl_dir = Path(data_root) / label
            if not lbl_dir.exists():
                continue
            for fpath in sorted(lbl_dir.rglob("*.jpg")):
                self.samples.append((str(fpath), lbl_idx))

        print(f"[Dataset] {len(self.samples)} frames  "
              f"({', '.join(f'{HOI_CLASSES[i]}:{sum(1 for _,l in self.samples if l==i)}' for i in range(NUM_HOI))})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frame = cv2.imread(path)
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        H, W = frame.shape[:2]

        # Use all object classes, not just BALL_CLS
        results = self.yolo(frame, classes=[PERSON_CLS] + OBJECT_CLASSES,
                             conf=self.conf, verbose=False, device=self.device)
        r       = results[0]
        persons, ball_bbox = _parse_yolo_static(r, H, W)

        if not persons:
            persons = [{"bbox": np.array([0,0,W,H], dtype=np.float32),
                        "kps":  np.zeros((17,3), dtype=np.float32)}]
        if ball_bbox is None:
            ball_bbox = np.array([W//2-20, H//2-20, W//2+20, H//2+20], dtype=np.float32)

        bcx = (ball_bbox[0] + ball_bbox[2]) / 2
        bcy = (ball_bbox[1] + ball_bbox[3]) / 2
        person = min(persons, key=lambda p: _box_dist(p["bbox"], bcx, bcy))
        pbox   = person["bbox"]
        kps    = person["kps"]

        # Augment crops: random brightness/contrast + horizontal flip
        person_crop = _crop_and_resize(frame, pbox, CROP_SIZE)
        ball_crop   = _crop_and_resize(frame, ball_bbox, CROP_SIZE)
        person_crop = _augment_crop(person_crop)
        ball_crop   = _augment_crop(ball_crop)

        # Augment pose: add small Gaussian noise to keypoint positions
        kps_aug = kps.copy()
        noise   = np.random.normal(0, 0.01, kps_aug[:, :2].shape).astype(np.float32)
        kps_aug[:, :2] += noise * (kps_aug[:, 2:3] > 0.25)  # only on confident kps

        pc = _to_tensor(person_crop)
        bc = _to_tensor(ball_crop)
        pf = torch.tensor(_encode_pose(kps_aug, W, H),                dtype=torch.float32)
        sf = torch.tensor(_encode_spatial(pbox, ball_bbox, W, H, kps=kps_aug,
                                           vel_feat=None), dtype=torch.float32)

        return pc, bc, pf, sf, torch.tensor(label, dtype=torch.long)


def train_hoi_head(
    data_root:  str   = "data",
    save_path:  str   = "checkpoints/hoi_head.pt",
    epochs:     int   = 30,
    batch_size: int   = 8,
    lr:         float = 1e-3,
    device:     str   = "cpu",
    yolo_path:  str   = "yolov8n-pose.pt",
):
    import os

    dataset = HOIDataset(data_root, yolo_path=yolo_path, device=device)
    if len(dataset) == 0:
        print("[Train] No data found. Run data_collector.py first.")
        return

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )

    head      = HOIHead().to(device)
    criterion = HOILoss().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n[Train] Training HOI head for {epochs} epochs on {device}")
    print(f"[Train] {sum(p.numel() for p in head.parameters()):,} parameters\n")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for pc, bc, pf, sf, labels in loader:
            pc, bc  = pc.to(device), bc.to(device)
            pf, sf  = pf.to(device), sf.to(device)
            labels  = labels.to(device)

            optimizer.zero_grad()
            logits = head(pc, bc, pf, sf)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

        scheduler.step()
        avg_loss = total_loss / max(total, 1)
        acc      = correct / max(total, 1)
        print(f"Epoch {epoch+1:03d}/{epochs}  "
              f"loss={avg_loss:.4f}  acc={acc:.2%}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save(head.state_dict(), save_path)
            print(f"  → saved {save_path}")

    print(f"\n[Train] Done. Final checkpoint: {save_path}")
    return head


# ══════════════════════════════════════════════════════════════
# Feature helpers
# ══════════════════════════════════════════════════════════════

def _crop_and_resize(frame: np.ndarray, bbox: np.ndarray,
                     size: int = CROP_SIZE) -> np.ndarray:
    H, W = frame.shape[:2]
    x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
    x2 = min(W, int(bbox[2])); y2 = min(H, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        crop = np.zeros((size, size, 3), dtype=np.uint8)
    else:
        crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (size, size))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _augment_crop(img_rgb: np.ndarray) -> np.ndarray:
    """
    Random augmentation on a (H,W,3) uint8 RGB crop.
    Applied during training only — makes the model invariant to
    brightness changes, minor color shifts, and left/right flips.
    """
    img = img_rgb.copy().astype(np.float32)
    # Random brightness  ±30%
    img *= np.random.uniform(0.7, 1.3)
    # Random contrast    ±20%
    mean = img.mean()
    img  = (img - mean) * np.random.uniform(0.8, 1.2) + mean
    # Horizontal flip (50% chance) — person/ball appearance is flip-invariant
    if np.random.rand() < 0.5:
        img = img[:, ::-1, :].copy()
    return np.clip(img, 0, 255).astype(np.uint8)


def _to_tensor(img_rgb: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Returns (3, H, W) — no batch dim. DataLoader adds batch; predict() adds manually."""
    t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).to(device)


def _encode_pose(kps: np.ndarray, W: int, H: int) -> np.ndarray:
    flat = kps.copy().astype(np.float32)
    flat[:, 0] /= (W + 1e-6)
    flat[:, 1] /= (H + 1e-6)
    return flat.flatten()


def _encode_spatial(pbox: np.ndarray, bbox: np.ndarray,
                    W: int, H: int,
                    kps: np.ndarray = None,
                    vel_feat: np.ndarray = None) -> np.ndarray:
    """
    16-dim geometry + motion feature vector.

    Dims 0-3:   person bbox (cx,cy,w,h normalised)
    Dims 4-7:   object bbox (cx,cy,w,h normalised)
    Dims 8-9:   person_centre - object_centre (relative position)
    Dims 10-11: wrist_centre  - object_centre (most discriminative)
    Dim  12:    best elbow extension 0-1  (throw=high, hold=low)
    Dim  13:    wrist height above shoulder 0-1 (throw=high, catch=low)
    Dim  14:    wrist speed 0-1  (holding=0, throwing/catching=high)
    Dim  15:    wrist direction -1→+1  (throw=+1 up/forward, catch=-1 down)
    """
    def _norm(b):
        cx = (b[0]+b[2]) / 2 / W
        cy = (b[1]+b[3]) / 2 / H
        w  = (b[2]-b[0])     / W
        h  = (b[3]-b[1])     / H
        return np.array([cx, cy, w, h], dtype=np.float32)

    p = _norm(pbox); b = _norm(bbox)
    person_to_obj = p[:2] - b[:2]

    wrist_to_obj    = np.zeros(2,  dtype=np.float32)
    elbow_ext       = np.zeros(1,  dtype=np.float32)
    wrist_above_sh  = np.zeros(1,  dtype=np.float32)

    if kps is not None:
        # Wrist-to-object (dims 10-11)
        best_c = -1.0
        for idx in (9, 10):
            wx, wy, wc = kps[idx]
            if wc > best_c:
                best_c = wc
                wrist_to_obj = np.array([
                    wx / (W + 1e-6) - b[0],
                    wy / (H + 1e-6) - b[1],
                ], dtype=np.float32)

        # Elbow extension (dim 12) — 0=bent, 1=straight
        for sh_i, el_i, wr_i in [(5,7,9), (6,8,10)]:
            sx,sy,sc = kps[sh_i]; ex,ey,ec = kps[el_i]; wx2,wy2,wc2 = kps[wr_i]
            if sc > 0.25 and ec > 0.25 and wc2 > 0.25:
                ba = np.array([sx-ex, sy-ey])
                bc_ = np.array([wx2-ex, wy2-ey])
                cos = np.dot(ba,bc_) / (np.linalg.norm(ba)*np.linalg.norm(bc_)+1e-8)
                ang = math.degrees(math.acos(float(np.clip(cos,-1,1))))
                elbow_ext[0] = max(elbow_ext[0], max(0.0, (ang-90)/90))

        # Wrist above shoulder (dim 13) — throw=1, catch/hold=0
        for wk, sk in [(9,5),(10,6)]:
            wx3,wy3,wc3 = kps[wk]; ssx,ssy,ssc = kps[sk]
            if wc3 > 0.25 and ssc > 0.25:
                wrist_above_sh[0] = max(wrist_above_sh[0],
                                        max(0.0, min(1.0, (ssy-wy3)/80)))

    # Velocity features (dims 14-15)
    if vel_feat is not None and len(vel_feat) == 2:
        velocity = vel_feat.astype(np.float32)
    else:
        velocity = np.zeros(2, dtype=np.float32)
    obj_w = (bbox[2] - bbox[0]) / (W + 1e-6)
    obj_h = (bbox[3] - bbox[1]) / (H + 1e-6)
    obj_aspect = np.array([
        obj_w / (obj_h + 1e-6),          # aspect ratio
        min(obj_w, obj_h),                # shorter side (size proxy)
    ], dtype=np.float32)

    # NEW: is the wrist inside the object bbox? (strong holding signal)
    wrist_in_obj = np.zeros(2, dtype=np.float32)
    if kps is not None:
        for fi, idx in enumerate((9, 10)):
            wx, wy, wc = kps[idx]
            if wc > 0.1:
                in_x = bbox[0] < wx < bbox[2]
                in_y = bbox[1] < wy < bbox[3]
                wrist_in_obj[fi] = 1.0 if (in_x and in_y) else 0.0
    return np.concatenate([p, b, person_to_obj,wrist_to_obj, elbow_ext, wrist_above_sh, velocity, obj_aspect, wrist_in_obj])

def _box_dist(bbox: np.ndarray, cx: float, cy: float) -> float:
    return math.hypot((bbox[0]+bbox[2])/2 - cx, (bbox[1]+bbox[3])/2 - cy)


def _parse_yolo_static(r, H=None, W=None):
    """Dataset version of _parse_yolo — matches two-threshold logic, no self."""
    persons   = []
    objects   = []
    kp_data   = r.keypoints.data if r.keypoints is not None else None
    kp_cursor = 0

    for box in r.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = np.array(box.xyxy[0].tolist(), dtype=np.float32)

        if cls == PERSON_CLS:
            kps = (kp_data[kp_cursor].cpu().numpy()
                   if kp_data is not None and kp_cursor < len(kp_data)
                   else np.zeros((17, 3), dtype=np.float32))
            kp_cursor += 1
            if conf >= PERSON_CONF:
                persons.append({"bbox": bbox, "kps": kps})
        elif cls in OBJECT_CLASSES:
            if conf >= OBJECT_CONF:
                objects.append({"bbox": bbox, "conf": conf, "cls": cls})

    best_obj = None

    if objects and persons:
        kps    = persons[0]["kps"]
        wrists = [(float(kps[i][0]), float(kps[i][1]))
                  for i in (9, 10) if kps[i][2] > 0.2]
        if wrists:
            def wrist_dist(obj):
                ocx = (obj["bbox"][0] + obj["bbox"][2]) / 2
                ocy = (obj["bbox"][1] + obj["bbox"][3]) / 2
                return min(math.hypot(ocx - wx, ocy - wy) for wx, wy in wrists)
            best_obj = min(objects, key=wrist_dist)["bbox"]
        else:
            best_obj = max(objects, key=lambda o: o["conf"])["bbox"]
    elif objects:
        best_obj = max(objects, key=lambda o: o["conf"])["bbox"]

    return persons, best_obj


# ══════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════

def draw_results(frame: np.ndarray, result: HOIResult,
                 fps: float = 0.0) -> np.ndarray:
    out   = frame.copy()
    cls   = result.hoi_class
    color = COLORS.get(cls, (180, 180, 180))
    H, W  = out.shape[:2]

    if result.ball_bbox is not None:
        x1,y1,x2,y2 = result.ball_bbox.astype(int)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(out, (x1,y1), (x2,y2), COLORS["ball"], 2)
            cv2.putText(out, "object", (x1, max(y1-6,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["ball"], 2, cv2.LINE_AA)

    if result.person_bbox is not None:
        x1,y1,x2,y2 = result.person_bbox.astype(int)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            bg_y = max(y1-24, 0)
            cv2.rectangle(out, (x1,bg_y), (x2,bg_y+22), color, -1)
            cv2.putText(out, f"person  [{cls}]", (x1+4,bg_y+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)

    if result.keypoints is not None:
        kps = result.keypoints
        for i, j in SKELETON_EDGES:
            xi,yi,ci = kps[i]; xj,yj,cj = kps[j]
            if ci > 0.25 and cj > 0.25:
                cv2.line(out, (int(xi),int(yi)), (int(xj),int(yj)),
                         COLORS["skeleton"], 1, cv2.LINE_AA)
        for k in range(17):
            x,y,c = kps[k]
            if c > 0.25:
                kc = COLORS["wrist"] if k in (9,10) else (80,220,80)
                cv2.circle(out, (int(x),int(y)), 4, kc, -1)

    ov = out.copy()
    cv2.rectangle(ov, (0,0), (W,58), (15,15,15), -1)
    cv2.addWeighted(ov, 0.72, out, 0.28, 0, out)
    cv2.putText(out, cls.replace("_"," ").upper(),
                (10,34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(out, f"conf {result.confidence:.0%}  |  FPS {fps:.1f}",
                (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190,190,190), 1, cv2.LINE_AA)

    bx = W - 170
    for i, (c,v) in enumerate(sorted(result.scores.items(), key=lambda t: -t[1])):
        by0 = 8 + i*22
        cv2.rectangle(out, (bx,by0), (bx+int(v*148),by0+14),
                      COLORS.get(c,(120,120,120)), -1)
        cv2.putText(out, f"{c:<16} {v:.2f}", (bx-2,by0+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220,220,220), 1)
    return out