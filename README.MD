# Computer Vision Object Interaction Detection

A simple project for human-object interaction (HOI) detection using YOLO pose/object detection and a custom HOI head.

## Project Structure

- `hoi_model.py` - Core HOI detection code, model definitions, training dataset, and inference pipeline.
- `hoi_pipeline.py` - Live webcam pipeline for the full OnePassHOINet model with real-time annotation.
- `test_hoi.py` - Script for testing HOI detection on live webcam or saved frame folders, and for training the HOI head.
- `data_collector.py` - Tool for collecting labeled webcam clips and frames for training.
- `frames_capture.py` - Simple script to capture raw webcam frames into `testing_frames/` folders.
- `checkpoints/` - Saved model checkpoints, including `hoi_head.pt` used by `test_hoi.py`.
- `data/` - Collected training data organized by HOI class.
- `testing_frames/` - Captured frame folders for offline evaluation.
- `yolov8n-pose.pt`, `yolov8n.pt` - YOLO model weights used for detection.

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- PyTorch

Install required packages with pip:

```bash
pip install opencv-python numpy torch
```

## Usage

### 1. Collect training data

Run `data_collector.py` to capture labeled clips or frame-only data.

```bash
python data_collector.py --label throwing --clips 10
python data_collector.py --label catching --clips 10
python data_collector.py --label holding --clips 10
python data_collector.py --label no_interaction --clips 10
```

If you want interactive label selection, run:

```bash
python data_collector.py
```

### 2. Capture raw testing frames

Use `frames_capture.py` to save webcam frames into `testing_frames/` for offline evaluation.

```bash
python frames_capture.py
```

### 3. Train the HOI head

Train the classifier head on collected data with:

```bash
python test_hoi.py --train --data data/ --epochs 30 --batch-size 8
```

This saves the checkpoint to `checkpoints/hoi_head.pt` by default.

### 4. Run live HOI inference

Use `test_hoi.py` with a YOLO-based detector and optional HOI head:

```bash
python test_hoi.py --live --checkpoint checkpoints/hoi_head.pt
```

If you only want to run the detector without a trained HOI head, omit `--checkpoint`.

### 5. Run on saved frame folders

Evaluate on a captured frame folder:

```bash
python test_hoi.py --folder
```

Or specify a folder explicitly:

```bash
python test_hoi.py --folder testing_frames/frame_folder_vid_1
```

## Alternate live pipeline

`hoi_pipeline.py` also provides a separate live webcam demo for the `OnePassHOINet` model:

```bash
python hoi_pipeline.py --checkpoint checkpoints/hoi_epoch030.pt
```

Use `--no-checkpoint` to run with random weights for pipeline testing.

## Notes

- `test_hoi.py` uses `HOIDetector` from `hoi_model.py`.
- `data_collector.py` writes metadata to `data/metadata.json`.
- `frames_capture.py` stores captured frames under `testing_frames/frame_folder_vid_<N>/`.

