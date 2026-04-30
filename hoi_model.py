"""
hoi_model.py
============
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
    65,   # remote
    66,   # keyboard
    67,   # cell phone
    73,   # book
    74,   # clock
    76,   # scissors
    77,   # teddy bear
    78,   # hair drier
    79,   # toothbrush
]
BALL_CLS      = 32

PERSON_CONF   = 0.40
OBJECT_CONF   = 0.15          


MIN_OBJ_PERSON_AREA_RATIO = 0.008   
MAX_OBJ_PERSON_AREA_RATIO = 0.30    
WRIST_SEARCH_RATIO = 0.70           
MIN_OBJECT_CONF_FINAL = 0.20       

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


class BBoxSmoother:
    
    RESET_FRAMES = 5

    def __init__(self, alpha: float = 0.50):
        self.alpha       = alpha
        self._smoothed   = None
        self._none_count = 0

    def update(self, bbox: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if bbox is None:
            self._none_count += 1
            if self._none_count >= self.RESET_FRAMES:
                self._smoothed = None
            return self._smoothed

        self._none_count = 0
        if self._smoothed is None:
            self._smoothed = bbox.copy()
        else:
            # NEW: reset if detection jumps more than 20% of frame width
            prev_cx = (self._smoothed[0] + self._smoothed[2]) / 2
            new_cx  = (bbox[0] + bbox[2]) / 2
            if abs(new_cx - prev_cx) > 0.20 * (bbox[2] - bbox[0]) * 3:
                self._smoothed = bbox.copy()  # hard reset on large jump
            else:
                self._smoothed = (self.alpha * bbox + (1.0 - self.alpha) * self._smoothed)
        return self._smoothed.copy()

    def reset(self):
        self._smoothed   = None
        self._none_count = 0


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
# Wrist velocity tracker
# ══════════════════════════════════════════════════════════════
class WristVelocityTracker:
    def __init__(self, history: int = 6):
        self._hist   = []
        self._maxlen = history

    def update(self, kps: np.ndarray, W: int, H: int) -> np.ndarray:
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
        dt  = max(t2 - t1, 1e-3)
        vx  = (x2 - x1) / dt
        vy  = (y2 - y1) / dt
        speed      = math.hypot(vx, vy)
        speed_norm = min(1.0, speed / 2.0)
        direction  = float(np.clip(-vy / (speed + 1e-6), -1.0, 1.0)) if speed > 0.05 else 0.0
        return np.array([speed_norm, direction], dtype=np.float32)

    def reset(self):
        self._hist = []


# ══════════════════════════════════════════════════════════════
# Ballistic object tracker
# ══════════════════════════════════════════════════════════════
class BallisticTracker:
    MAX_COASTING = 30

    def __init__(self):
        self._bbox:       Optional[np.ndarray] = None
        self._vel:        np.ndarray = np.zeros(4, dtype=np.float32)
        self._prev_bbox:  Optional[np.ndarray] = None
        self._prev_t:     float = 0.0
        self._coast_frames: int = 0
        self._last_t:     float = 0.0

    def update(self, bbox: Optional[np.ndarray]) -> Optional[np.ndarray]:
        now = time.monotonic()
        dt  = max(now - self._last_t, 1e-3) if self._last_t else 0.033
        self._last_t = now

        if bbox is not None:
            if self._prev_bbox is not None and self._prev_t:
                prev_dt    = max(now - self._prev_t, 1e-3)
                self._vel  = (bbox - self._prev_bbox) / prev_dt
            self._prev_bbox    = self._bbox
            self._prev_t       = now
            self._bbox         = bbox.copy()
            self._coast_frames = 0
            return bbox

        if self._bbox is None:
            return None

        self._coast_frames += 1
        if self._coast_frames > self.MAX_COASTING:
            self._bbox = None
            self._vel  = np.zeros(4, dtype=np.float32)
            return None

        gravity     = np.array([0.0, 180.0, 0.0, 180.0], dtype=np.float32)
        self._bbox  = self._bbox + self._vel * dt + 0.5 * gravity * dt * dt
        self._vel[1] += gravity[1] * dt
        self._vel[3] += gravity[3] * dt
        return self._bbox.copy()

    @property
    def coasting(self) -> bool:
        return self._coast_frames > 0

    @property
    def coast_confidence(self) -> float:
        return max(0.0, 1.0 - self._coast_frames / self.MAX_COASTING)

    def reset(self):
        self.__init__()


# ══════════════════════════════════════════════════════════════
# Throw state machine
# ══════════════════════════════════════════════════════════════
from enum import Enum

class ThrowState(Enum):
    IDLE    = "idle"
    WINDUP  = "windup"
    RELEASE = "release"
    FOLLOW  = "follow"

class ThrowStateMachine:
    SPEED_WINDUP      = 0.25
    SPEED_DROP        = 0.10
    RELEASE_DIST_JUMP = 0.12
    FOLLOW_FRAMES     = 22
    WINDUP_TIMEOUT    = 45

    def __init__(self):
        self.state         = ThrowState.IDLE
        self._follow_count = 0
        self._windup_count = 0
        self._prev_wrist_obj_dist: Optional[float] = None

    def update(self, vel_speed, vel_direction, wrist_obj_dist, hoi_head_scores):
        prev_dist                 = self._prev_wrist_obj_dist
        self._prev_wrist_obj_dist = wrist_obj_dist

        if self.state == ThrowState.IDLE:
            if vel_speed > self.SPEED_WINDUP and vel_direction > 0.1:
                self.state         = ThrowState.WINDUP
                self._windup_count = 0

        elif self.state == ThrowState.WINDUP:
            self._windup_count += 1
            dist_jumped = (prev_dist is not None
                           and wrist_obj_dist - prev_dist > self.RELEASE_DIST_JUMP)
            if dist_jumped:
                self.state         = ThrowState.RELEASE
                self._follow_count = 0
            elif vel_speed < self.SPEED_DROP or self._windup_count > self.WINDUP_TIMEOUT:
                self.state = ThrowState.IDLE

        elif self.state == ThrowState.RELEASE:
            self.state         = ThrowState.FOLLOW
            self._follow_count = 0

        elif self.state == ThrowState.FOLLOW:
            self._follow_count += 1
            if self._follow_count >= self.FOLLOW_FRAMES:
                self.state = ThrowState.IDLE

        scores = dict(hoi_head_scores)
        if self.state in (ThrowState.RELEASE, ThrowState.FOLLOW):
            t        = self._follow_count / max(self.FOLLOW_FRAMES, 1)
            throw_sc = 1.0 - 0.4 * t
            scores["throwing"]       = throw_sc
            scores["no_interaction"] = 0.0
            scores["holding"]        = max(0.0, scores.get("holding", 0) * 0.3)
            scores["catching"]       = max(0.0, scores.get("catching", 0) * 0.5)
            total  = sum(scores.values()) + 1e-9
            scores = {c: v / total for c, v in scores.items()}
        elif self.state == ThrowState.WINDUP:
            scores["throwing"] = min(1.0, scores.get("throwing", 0) + 0.25)
            total  = sum(scores.values()) + 1e-9
            scores = {c: v / total for c, v in scores.items()}
        return scores

    def reset(self):
        self.__init__()


class AsymmetricSmoother:
    def __init__(self, alpha_rise=0.70, alpha_fall=0.30, alpha_std=0.40):
        self.alpha_rise = alpha_rise
        self.alpha_fall = alpha_fall
        self.alpha_std  = alpha_std
        self.smoothed   = {c: 1.0 / NUM_HOI for c in HOI_CLASSES}

    def update(self, scores: dict) -> dict:
        for c in HOI_CLASSES:
            new_s = scores.get(c, 0.0)
            old_s = self.smoothed[c]
            if c in ("throwing", "catching"):
                alpha = self.alpha_rise if new_s > old_s else self.alpha_fall
            else:
                alpha = self.alpha_std
            self.smoothed[c] = alpha * new_s + (1.0 - alpha) * old_s
        total = sum(self.smoothed.values()) + 1e-9
        return {c: v / total for c, v in self.smoothed.items()}

    def reset(self):
        self.smoothed = {c: 1.0 / NUM_HOI for c in HOI_CLASSES}


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
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, out_dim), nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.net(x)


class PoseMLP(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_KP * 3, 128), nn.ReLU(True),
            nn.Linear(128, out_dim), nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.net(x)


class SpatialMLP(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(True),
            nn.Linear(64, out_dim), nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.net(x)


class HOIHead(nn.Module):
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
        self.classifier  = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(d_model, NUM_HOI),
        )

    def forward(self, person_crop, ball_crop, pose_feat, spatial_feat):
        t0     = self.pose_enc(pose_feat)
        t1     = self.spatial_enc(spatial_feat)
        tokens = torch.stack([t0, t1], dim=1)
        tokens = self.transformer(tokens)
        fused  = tokens.reshape(tokens.shape[0], -1)
        return self.classifier(fused)


# ══════════════════════════════════════════════════════════════
# Helper: IoU between two bboxes [x1,y1,x2,y2]
# ══════════════════════════════════════════════════════════════
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua    = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-6)


def _clamp_bbox(bbox: np.ndarray, W: int, H: int) -> np.ndarray:
    """Clamp bbox to frame boundaries."""
    return np.array([
        max(0.0, bbox[0]), max(0.0, bbox[1]),
        min(W,   bbox[2]), min(H,   bbox[3]),
    ], dtype=np.float32)


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

        print(f"[HOI] Loading pose model: {yolo_path}")
        self.yolo_pose = YOLO(yolo_path)

        print(f"[HOI] Loading detection model: yolov8n.pt")
        self.yolo_det  = YOLO("yolov8n.pt")

        _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.yolo_pose(_dummy, classes=[PERSON_CLS], conf=PERSON_CONF,
                       verbose=False, device=device)
        self.yolo_det(_dummy,  conf=OBJECT_CONF,
                      verbose=False, device=device)
        self.yolo = self.yolo_pose
        print("[HOI] Both models warmed up.")

        self.head = HOIHead().to(device)
        if hoi_head_path:
            self._load_head(hoi_head_path)
        else:
            print("[HOI] No HOI head weights given — using random weights.")

        self.head.eval()
        self.smoother      = AsymmetricSmoother()
        self.wrist_vel     = WristVelocityTracker(history=6)
        self._ballistic    = BallisticTracker()
        self._bbox_smooth  = BBoxSmoother(alpha=0.50)   # FIX 5
        self._throw_sm     = ThrowStateMachine()
        self._prev_wrist_to_obj_dist = None
        self._fidx = 0
        print(f"[HOI] Ready on {device}.")

    def predict(self, frame_bgr: np.ndarray, debug: bool = False) -> HOIResult:
        self._fidx += 1
        H, W = frame_bgr.shape[:2]

        pose_results = self.yolo_pose(
            frame_bgr, classes=[PERSON_CLS], conf=PERSON_CONF,
            imgsz=480, verbose=False, device=self.device,
        )
        det_results = self.yolo_det(
            frame_bgr, classes=OBJECT_CLASSES, conf=OBJECT_CONF,
            imgsz=480, verbose=False, device=self.device,
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

        persons, raw_ball_bbox = self._parse_two_results(
            pose_results[0], det_results[0], H, W
        )

        # Clamp raw detection to frame (FIX 6)
        if raw_ball_bbox is not None:
            raw_ball_bbox = _clamp_bbox(raw_ball_bbox, W, H)

        # BBox smoother before ballistic (FIX 5)
        smoothed_obj = self._bbox_smooth.update(raw_ball_bbox)

        # Ballistic prediction if detector misses
        ball_bbox = self._ballistic.update(smoothed_obj)

        if ball_bbox is not None:
            ball_bbox = _clamp_bbox(ball_bbox, W, H)

        if ball_bbox is None or not persons:
            self._throw_sm.reset()
            empty = {c: (1.0 if c == "no_interaction" else 0.0) for c in HOI_CLASSES}
            sm    = self.smoother.update(empty)
            pb    = persons[0]["bbox"] if persons else None
            kp    = persons[0]["kps"]  if persons else None
            return HOIResult(
                hoi_class="no_interaction", confidence=sm["no_interaction"],
                scores=sm, person_bbox=pb, ball_bbox=ball_bbox,
                keypoints=kp, frame_idx=self._fidx,
            )

        bcx    = (ball_bbox[0] + ball_bbox[2]) / 2
        bcy    = (ball_bbox[1] + ball_bbox[3]) / 2
        person = min(persons, key=lambda p: _box_dist(p["bbox"], bcx, bcy))
        pbox   = person["bbox"]
        kps    = person["kps"]

        person_crop  = _crop_and_resize(frame_bgr, pbox, CROP_SIZE)
        ball_crop    = _crop_and_resize(frame_bgr, ball_bbox, CROP_SIZE)
        pose_feat    = _encode_pose(kps, W, H)
        vel_feat     = self.wrist_vel.update(kps, W, H)
        spatial_feat = _encode_spatial(pbox, ball_bbox, W, H, kps=kps, vel_feat=vel_feat)

        with torch.no_grad():
            pc = _to_tensor(person_crop, self.device).unsqueeze(0)
            bc = _to_tensor(ball_crop,   self.device).unsqueeze(0)
            pf = torch.tensor(pose_feat,    dtype=torch.float32).unsqueeze(0).to(self.device)
            sf = torch.tensor(spatial_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.head(pc, bc, pf, sf)
            probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

        frame_scores      = {c: float(probs[i]) for i, c in enumerate(HOI_CLASSES)}
        wrist_to_obj_dist = float(np.linalg.norm(spatial_feat[10:12]))
        vel_speed         = float(vel_feat[0])
        vel_direction     = float(vel_feat[1])

        frame_scores = self._throw_sm.update(
            vel_speed, vel_direction, wrist_to_obj_dist, frame_scores,
        )
        self._prev_wrist_to_obj_dist = wrist_to_obj_dist

        smoothed = self.smoother.update(frame_scores)
        best     = max(smoothed, key=smoothed.get)

        return HOIResult(
            hoi_class=best, confidence=round(smoothed[best], 3),
            scores=smoothed, person_bbox=pbox, ball_bbox=ball_bbox,
            keypoints=kps, frame_idx=self._fidx,
        )

    # ── internal helpers ────────────────────────────────────────────────────

    def train_mode(self): self.head.train()
    def eval_mode(self):  self.head.eval()

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
        Parse results from two separate YOLO calls.

        Key fixes vs original:
        - Object bbox size is validated against person bbox (not raw frame size)
        - Wrist search radius is relative to person width
        - Improved fallback chain: wrist → person-interior → highest-conf
        - IoU-based dedup removes overlapping low-conf detections
        - All returned bboxes are clamped to frame bounds
        """
        # ── Persons from pose result ─────────────────────────────────────
        persons   = []
        kp_data   = pose_r.keypoints.data if pose_r.keypoints is not None else None
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
                    persons.append({"bbox": _clamp_bbox(bbox, W, H), "kps": kps})
                kp_cursor += 1

        # ── Raw objects from detection result ────────────────────────────
        raw_objects = []
        if det_r.boxes is not None:
            for box in det_r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = np.array(box.xyxy[0].tolist(), dtype=np.float32)
                if cls != PERSON_CLS:
                    raw_objects.append({"bbox": _clamp_bbox(bbox, W, H),
                                        "conf": conf, "cls": cls})

        if not raw_objects or not persons:
            # No objects or no persons — no interaction
            return persons, None

        # ── Area filter relative to person bbox (FIX 1, 2) ──────────────
        px1, py1, px2, py2 = persons[0]["bbox"]
        person_area = max((px2 - px1) * (py2 - py1), 1.0)

        def obj_area(o):
            b = o["bbox"]
            return max((b[2] - b[0]) * (b[3] - b[1]), 1.0)

        objects = []
        for o in raw_objects:
            a     = obj_area(o)
            ratio = a / person_area
            if (MIN_OBJ_PERSON_AREA_RATIO <= ratio <= MAX_OBJ_PERSON_AREA_RATIO
                    and o["conf"] >= MIN_OBJECT_CONF_FINAL):
                objects.append(o)

        # If strict filter removes everything, fall back to looser area check
        if not objects:
            objects = [o for o in raw_objects
                       if obj_area(o) / person_area >= MIN_OBJ_PERSON_AREA_RATIO * 0.4
                       and o["conf"] >= OBJECT_CONF]

        if not objects:
            return persons, None
        
        

        # ── IoU-based dedup — remove heavily overlapping boxes (FIX 1) ──
        objects.sort(key=lambda o: o["conf"], reverse=True)
        kept = []
        for o in objects:
            if not any(_iou(o["bbox"], k["bbox"]) > 0.55 for k in kept):
                kept.append(o)
        objects = kept

        # ── Wrist-proximity search (FIX 3) ───────────────────────────────
        kps    = persons[0]["kps"]
        person_w = px2 - px1
        max_wrist_dist = person_w * WRIST_SEARCH_RATIO   # relative to person

        wrists = [(float(kps[i][0]), float(kps[i][1]))
                  for i in (9, 10) if float(kps[i][2]) > 0.10]  # lowered from 0.25

        best_obj = None

        if wrists:
            def wdist(obj):
                ocx = (obj["bbox"][0] + obj["bbox"][2]) / 2
                ocy = (obj["bbox"][1] + obj["bbox"][3]) / 2
                return min(math.hypot(ocx - wx, ocy - wy) for wx, wy in wrists)

            nearby = [o for o in objects if wdist(o) < max_wrist_dist]
            if nearby:
                best_obj = min(nearby, key=wdist)["bbox"]

        # FIX 4: Fallback — wrist-based search failed, try person-interior
        if best_obj is None:
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            interior = [o for o in objects
                        if px1 < (o["bbox"][0] + o["bbox"][2]) / 2 < px2
                        and py1 < (o["bbox"][1] + o["bbox"][3]) / 2 < py2]
            if interior:
                best_obj = min(
                    interior,
                    key=lambda o: math.hypot(
                        (o["bbox"][0]+o["bbox"][2])/2 - pcx,
                        (o["bbox"][1]+o["bbox"][3])/2 - pcy,
                    )
                )["bbox"]

        # Final fallback — highest-confidence object
        if best_obj is None:
            best_obj = max(objects, key=lambda o: o["conf"])["bbox"]

        return persons, best_obj

    # kept for dataset compatibility
    def _parse_yolo(self, r, H, W):
        persons  = []
        objects  = []
        kp_data  = r.keypoints.data if r.keypoints is not None else None
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
            elif cls in OBJECT_CLASSES and conf >= OBJECT_CONF:
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
                    return min(math.hypot(ocx-wx, ocy-wy) for wx, wy in wrists)
                best_obj_bbox = min(objects, key=wrist_dist)["bbox"]
            else:
                best_obj_bbox = max(objects, key=lambda o: o["conf"])["bbox"]
        elif objects:
            best_obj_bbox = max(objects, key=lambda o: o["conf"])["bbox"]
        return persons, best_obj_bbox


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

class HOILoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torch.tensor([2.0, 0.7, 2.5, 2.0], dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    def forward(self, logits, labels):
        return self.ce(logits, labels)


class HOIDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str = "data",
                 yolo_path: str = "yolov8n-pose.pt",
                 device: str = "cpu", conf: float = 0.35):
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
        results = self.yolo(frame, classes=[PERSON_CLS] + OBJECT_CLASSES,
                            conf=self.conf, verbose=False, device=self.device)
        r       = results[0]
        persons, ball_bbox = _parse_yolo_static(r, H, W)

        if not persons:
            persons = [{"bbox": np.array([0, 0, W, H], dtype=np.float32),
                        "kps":  np.zeros((17, 3), dtype=np.float32)}]
        if ball_bbox is None:
            ball_bbox = np.array([W//2-20, H//2-20, W//2+20, H//2+20], dtype=np.float32)

        bcx    = (ball_bbox[0] + ball_bbox[2]) / 2
        bcy    = (ball_bbox[1] + ball_bbox[3]) / 2
        person = min(persons, key=lambda p: _box_dist(p["bbox"], bcx, bcy))
        pbox   = person["bbox"]
        kps    = person["kps"]

        person_crop = _augment_crop(_crop_and_resize(frame, pbox, CROP_SIZE))
        ball_crop   = _augment_crop(_crop_and_resize(frame, ball_bbox, CROP_SIZE))

        kps_aug = kps.copy()
        noise   = np.random.normal(0, 0.01, kps_aug[:, :2].shape).astype(np.float32)
        kps_aug[:, :2] += noise * (kps_aug[:, 2:3] > 0.25)

        pc = _to_tensor(person_crop)
        bc = _to_tensor(ball_crop)
        pf = torch.tensor(_encode_pose(kps_aug, W, H), dtype=torch.float32)

        if label == HOI_IDX["throwing"]:
            syn_vel = np.array([np.random.uniform(0.6, 1.0),
                                np.random.uniform(0.2, 1.0)], dtype=np.float32)
        elif label == HOI_IDX["catching"]:
            syn_vel = np.array([np.random.uniform(0.4, 0.9),
                                np.random.uniform(-1.0, -0.3)], dtype=np.float32)
        elif label == HOI_IDX["holding"]:
            syn_vel = np.array([np.random.uniform(0.0, 0.12),
                                np.random.uniform(-0.15, 0.15)], dtype=np.float32)
        else:
            syn_vel = np.array([np.random.uniform(0.0, 0.25),
                                np.random.uniform(-0.25, 0.25)], dtype=np.float32)

        sf = torch.tensor(_encode_spatial(pbox, ball_bbox, W, H,
                                          kps=kps_aug, vel_feat=syn_vel),
                          dtype=torch.float32)
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

    loader    = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    head      = HOIHead().to(device)
    criterion = HOILoss().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    print(f"\n[Train] {epochs} epochs  device={device}  "
          f"params={sum(p.numel() for p in head.parameters()):,}\n")

    for epoch in range(epochs):
        head.train()
        total_loss = correct = total = 0
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
        acc      = correct   / max(total, 1)
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

def _crop_and_resize(frame, bbox, size=CROP_SIZE):
    H, W = frame.shape[:2]
    x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
    x2 = min(W, int(bbox[2])); y2 = min(H, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        crop = np.zeros((size, size, 3), dtype=np.uint8)
    else:
        crop = frame[y1:y2, x1:x2]
    return cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB)


def _augment_crop(img_rgb):
    img  = img_rgb.copy().astype(np.float32)
    img *= np.random.uniform(0.7, 1.3)
    mean = img.mean()
    img  = (img - mean) * np.random.uniform(0.8, 1.2) + mean
    if np.random.rand() < 0.5:
        img = img[:, ::-1, :].copy()
    return np.clip(img, 0, 255).astype(np.uint8)


def _to_tensor(img_rgb, device="cpu"):
    t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).to(device)


def _encode_pose(kps, W, H):
    flat = kps.copy().astype(np.float32)
    flat[:, 0] /= (W + 1e-6)
    flat[:, 1] /= (H + 1e-6)
    return flat.flatten()


def _encode_spatial(pbox, bbox, W, H, kps=None, vel_feat=None):
    def _norm(b):
        cx = (b[0]+b[2]) / 2 / W
        cy = (b[1]+b[3]) / 2 / H
        w  = (b[2]-b[0])     / W
        h  = (b[3]-b[1])     / H
        return np.array([cx, cy, w, h], dtype=np.float32)

    p = _norm(pbox); b = _norm(bbox)
    person_to_obj   = p[:2] - b[:2]
    wrist_to_obj    = np.zeros(2, dtype=np.float32)
    elbow_ext       = np.zeros(1, dtype=np.float32)
    wrist_above_sh  = np.zeros(1, dtype=np.float32)

    if kps is not None:
        best_c = -1.0
        for idx in (9, 10):
            wx, wy, wc = kps[idx]
            if wc > best_c:
                best_c       = wc
                wrist_to_obj = np.array([
                    wx / (W + 1e-6) - b[0],
                    wy / (H + 1e-6) - b[1],
                ], dtype=np.float32)
        for sh_i, el_i, wr_i in [(5,7,9), (6,8,10)]:
            sx,sy,sc = kps[sh_i]; ex,ey,ec = kps[el_i]; wx2,wy2,wc2 = kps[wr_i]
            if sc > 0.25 and ec > 0.25 and wc2 > 0.25:
                ba  = np.array([sx-ex, sy-ey])
                bc_ = np.array([wx2-ex, wy2-ey])
                cos = np.dot(ba, bc_) / (np.linalg.norm(ba)*np.linalg.norm(bc_)+1e-8)
                ang = math.degrees(math.acos(float(np.clip(cos, -1, 1))))
                elbow_ext[0] = max(elbow_ext[0], max(0.0, (ang-90)/90))
        for wk, sk in [(9,5),(10,6)]:
            wx3,wy3,wc3 = kps[wk]; ssx,ssy,ssc = kps[sk]
            if wc3 > 0.25 and ssc > 0.25:
                wrist_above_sh[0] = max(wrist_above_sh[0],
                                        max(0.0, min(1.0, (ssy-wy3)/80)))

    velocity = vel_feat.astype(np.float32) if vel_feat is not None and len(vel_feat) == 2 \
               else np.zeros(2, dtype=np.float32)

    obj_w      = (bbox[2] - bbox[0]) / (W + 1e-6)
    obj_h      = (bbox[3] - bbox[1]) / (H + 1e-6)
    obj_aspect = np.array([obj_w / (obj_h + 1e-6), min(obj_w, obj_h)],
                          dtype=np.float32)

    wrist_in_obj = np.zeros(2, dtype=np.float32)
    if kps is not None:
        for fi, idx in enumerate((9, 10)):
            wx, wy, wc = kps[idx]
            if wc > 0.1:
                wrist_in_obj[fi] = 1.0 if (bbox[0] < wx < bbox[2]
                                            and bbox[1] < wy < bbox[3]) else 0.0

    return np.concatenate([p, b, person_to_obj, wrist_to_obj,
                           elbow_ext, wrist_above_sh, velocity,
                           obj_aspect, wrist_in_obj])


def _box_dist(bbox, cx, cy):
    return math.hypot((bbox[0]+bbox[2])/2 - cx, (bbox[1]+bbox[3])/2 - cy)


def _parse_yolo_static(r, H=None, W=None):
    persons  = []
    objects  = []
    kp_data  = r.keypoints.data if r.keypoints is not None else None
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
        elif cls in OBJECT_CLASSES and conf >= OBJECT_CONF:
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
                return min(math.hypot(ocx-wx, ocy-wy) for wx, wy in wrists)
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
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(out, (x1,y1), (x2,y2), COLORS["ball"], 2)
            cv2.putText(out, "object", (x1, max(y1-6,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["ball"], 2, cv2.LINE_AA)

    if result.person_bbox is not None:
        x1,y1,x2,y2 = result.person_bbox.astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
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