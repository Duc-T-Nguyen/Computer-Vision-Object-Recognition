import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
from torchvision import transforms

from two_pass_hoi import TwoPassHOIDetector
from two_pass_hoi import THROWABLE_COCO_LABELS
from two_pass_hoi import ObjectDetector
from two_pass_hoi import extract_human_object_pairs
from two_pass_hoi import build_spatial_map

# -----------------------------------
#  Function to Get the HOI
# -----------------------------------

def run_inference(model, frame_path, device, score_thresh=0.5, throw_thresh=0.2):
    """
    Full two-pass inference on a single image with no GT boxes.
    frame_path: Path or str to image file.
    Returns all (human, object) pairs with their throw probability.
    """
    model.eval()

    # Load with cv2, convert BGR -> RGB, then wrap in PIL for transforms
    bgr   = cv2.imread(str(frame_path))
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)

    W, H = img_pil.size   # PIL gives (width, height)

    # Transform for feature extractor
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Faster RCNN needs just ToTensor, no normalize
    detector_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    img_tensor      = transform(img_pil).unsqueeze(0).to(device)
    img_for_detect  = detector_transform(img_pil).unsqueeze(0).to(device)

    # ----------------------------------------
    #  Pass 1: detect all persons and objects
    # ----------------------------------------

    detector = ObjectDetector(score_thresh=score_thresh).to(device)
    detector.eval()

    with torch.no_grad():
        detections = detector(img_for_detect)

    pairs = extract_human_object_pairs(detections[0], allowed_object_labels = THROWABLE_COCO_LABELS)

    # Temporarily add this inside run_inference after detections
    print(f"  Detected {len(detections[0]['boxes'])} objects")
    print(f"  Labels: {detections[0]['labels'].tolist()}")
    print(f"  Scores: {[f'{s:.2f}' for s in detections[0]['scores'].tolist()]}")
    print(f"  Pairs found: {len(pairs)}")

    if len(pairs) == 0:
        print("  No human-object pairs detected.")
        return []

    # Scale boxes from original image space to 640x640 space
    scale_x = 640 / W
    scale_y = 640 / H

    def scale_box(box):
        x1, y1, x2, y2 = box.cpu().numpy()
        return torch.tensor([
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
        ], dtype=torch.float32, device=device)

    # --------------------------------------
    #  Pass 2: classify each pair
    # --------------------------------------

    results = []
    for pair in pairs:
        h_box = scale_box(pair['human_box']).unsqueeze(0)   # [1, 4]
        o_box = scale_box(pair['object_box']).unsqueeze(0)  # [1, 4]

        spatial_map = build_spatial_map(h_box, o_box).to(device)  # [1, 2, 64, 64]

        with torch.no_grad():
            logit = model(img_tensor, h_box, o_box, spatial_map)
            prob  = torch.sigmoid(logit).item()

        results.append({
            'human_box':    pair['human_box'].cpu().numpy(),
            'object_box':   pair['object_box'].cpu().numpy(),
            'object_label': pair['object_label'].item(),
            'throw_prob':   prob,
            'is_throw':     prob >= throw_thresh,
        })

    print("All pair probabilities:")
    for r in results:
        print(f"throw_prob={r['throw_prob']:.4f}  "
              f"object_label={r['object_label']}")

    results.sort(key=lambda x: x['throw_prob'], reverse=True)
    return results


def visualize_results(frame_path, results, output_path):
    """
    Draw bounding boxes and throw probabilities on the image and save.
    frame_path : Path or str to the source image
    output_path: Path or str where the annotated image will be saved
    """
    bgr = cv2.imread(str(frame_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(rgb)

    for r in results:
        h    = r['human_box']
        o    = r['object_box']
        prob = r['throw_prob']

        color = 'lime' if r['is_throw'] else 'red'
        alpha = 0.8   if r['is_throw'] else 0.3

        # Human box (solid)
        ax.add_patch(patches.Rectangle(
            (h[0], h[1]), h[2]-h[0], h[3]-h[1],
            linewidth=2, edgecolor=color, facecolor='none', alpha=alpha
        ))

        # Object box (dashed)
        ax.add_patch(patches.Rectangle(
            (o[0], o[1]), o[2]-o[0], o[3]-o[1],
            linewidth=2, edgecolor=color, facecolor='none',
            linestyle='--', alpha=alpha
        ))

        # Label
        ax.text(
            h[0], h[1] - 5,
            f"throw: {prob:.2f}",
            color=color, fontsize=10,
            bbox=dict(facecolor='black', alpha=0.5, pad=2)
        )

    ax.set_title(
        f"{Path(frame_path).name} — "
        f"{sum(r['is_throw'] for r in results)} throw(s) detected",
        fontsize=12
    )
    ax.axis('off')
    plt.tight_layout()

    try:
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"  Saved -> {output_path}")
    except Exception as e:
        print(f"  Failed to save annotated image: {e}")
    finally:
        plt.close(fig)   # always free memory even if save failed


# -----------------------------------
#  Go Retrieve Frames from Directory
# -----------------------------------

video_dirs = Path(__file__).parent / "testing_frames"

video_frames = {}
for vid_dir in video_dirs.iterdir():
    if vid_dir.is_dir():
        frames = [f for f in vid_dir.iterdir() if f.is_file()]
        video_frames[vid_dir.name] = frames

# -----------------------------------
#  Load model once outside the loop
# -----------------------------------

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model      = TwoPassHOIDetector(num_classes=1).to(device)
checkpoint = torch.load(
    Path(__file__).parent / 'two_pass_checkpoints' / 'epoch_02.pth',
    map_location=device
)
missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
print("Missing:   ", missing)
print("Unexpected:", unexpected)
model.eval()

# -----------------------------------
#  Perform HOI on Frames
# -----------------------------------

OUTPUT_ROOT = Path(__file__).parent / 'two_pass_classified_and_detected_frames'

for vid_dir, frames in video_frames.items():
    out_dir = OUTPUT_ROOT / vid_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame_path in frames:
        print(f"\n[{vid_dir}] {frame_path.name}")

        results = run_inference(model, frame_path, device)

        print(f"  Found {len(results)} human-object pairs:")
        for i, r in enumerate(results):
            print(f"    Pair {i+1}: throw_prob={r['throw_prob']:.3f}  "
                  f"is_throw={r['is_throw']}")

        out_path = out_dir / frame_path.name
        visualize_results(frame_path, results, output_path=out_path)
