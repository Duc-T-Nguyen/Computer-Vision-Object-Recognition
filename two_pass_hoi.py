# imports

# -------------------
#  General
# -------------------
import os
import json
from pathlib import Path
import sys
import scipy.io
import numpy as np

# --------------------
#  Data/Image Imports
# --------------------
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# for v-coco
# sys.path.insert(0, '{Path(__file__).parent}/v-coco')
# from vsrl_utils import CacheLoader
# import vsrl_tuils as vu

from torch.utils.data import ConcatDataset

# ----------------------------------
#  Additional Essential DNN Imports
# ----------------------------------
from torch import nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_align

# -----------------------------------------
#  Dataset Classes for HICO-DET and V-COCO
# -----------------------------------------

# HICO-DET dataset
# ----------------------------------------------
#  HICO-DET
#  Structure on disk:
#    hico_20160224_det/
#      images/train2015/   images/test2015/
#      anno_bbox.mat     <-main annotation file
# ----------------------------------------------

class HICODetDataset(Dataset):
    # each item from dataset is (human_box, object_box, hoi_id) triplet
    THROW_HOI_IDS = {230, 341, 501} # baseball bat throw, frisbee throw, sports_ball throw
    IMG_SIZE = 640 # must match images to size used in build_spatial_map

    def __init__(self, root, split = 'train', throw_only = False):
        self.root = root
        self.split = split
        self.throw_only = throw_only
        self.img_size = self.IMG_SIZE
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.samples = self.load_annotations()

    # parses the bbox struct into an array
    @staticmethod
    def parse_bbox_struct(bbox_field):
        # Numeric array — shape [4] or [N, 4]
        if isinstance(bbox_field, np.ndarray) and bbox_field.dtype.kind in ('f', 'i', 'u'):
            boxes = np.atleast_2d(bbox_field).astype(np.float32)
            return boxes - 1.0

        # Object array wrapping mat_structs — unwrap first
        if isinstance(bbox_field, np.ndarray) and bbox_field.dtype.kind == 'O':
            bbox_field = bbox_field.flat[0]  # grab the actual mat_struct

        x1 = np.atleast_1d(np.array(bbox_field.x1, dtype=np.float32))
        y1 = np.atleast_1d(np.array(bbox_field.y1, dtype=np.float32))
        x2 = np.atleast_1d(np.array(bbox_field.x2, dtype=np.float32))
        y2 = np.atleast_1d(np.array(bbox_field.y2, dtype=np.float32))
        boxes = np.stack([x1, y1, x2, y2], axis=1) - 1.0  # 0-index
        return boxes

    def load_annotations(self):
        # load the mat file for annotations of the bounding boxes
        mat = scipy.io.loadmat(
            os.path.join(self.root, 'anno_bbox.mat'),
            struct_as_record = False,
            squeeze_me = True
        )

        field = 'bbox_train' if self.split == 'train' else 'bbox_test'
        records = mat[field]

        samples = []
        for rec in records:
            filename = rec.filename
            img_path = os.path.join(
                self.root,
                'images',
                f'{"train" if self.split == "train" else "test"}2015',
                filename
            )

            # hoi is an array of structs, with id, bbox for human, and bbox for object
            hois = rec.hoi if hasattr(rec.hoi, '__iter__') else [rec.hoi]
            for hoi in hois:
                hoi_id = int(hoi.id)
                is_throw = hoi_id in self.THROW_HOI_IDS

                # if we are only training only throwing thing skip if the current hoi in record is not throwing
                if self.throw_only and not is_throw:
                    continue

                # Skip invisible HOIs
                if int(hoi.invis) == 1:
                    continue

                # bbox: [x1, y1, x2, y2] (1-indexed in .mat)
                human_boxes = self.parse_bbox_struct(hoi.bboxhuman)
                object_boxes = self.parse_bbox_struct(hoi.bboxobject)

                connections = np.atleast_2d(np.array(hoi.connection, dtype=np.int32))
                if connections.ndim == 1:
                    connections = connections.reshape(1, -1)
                connections = connections - 1

                for conn in connections:
                    h_idx, o_idx = conn[0], conn[1]

                    if h_idx < 0 or h_idx >= len(human_boxes):
                        continue
                    if o_idx < 0 or o_idx >= len(object_boxes):
                        continue

                    samples.append({
                        'img_path': img_path,
                        'human_box': human_boxes[h_idx],
                        'object_box': object_boxes[o_idx],
                        'hoi_id': hoi_id,
                        'label': float(is_throw)
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['img_path']).convert('RGB')
        W, H = img.size

        # scale bounding boxes from original image to resized image
        scale_x = self.img_size / W
        scale_y = self.img_size / H

        def scale_box(box):
            x1, y1, x2, y2 = box
            return np.array([
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ], dtype=np.float32)

        human_box = scale_box(s['human_box'])
        object_box = scale_box(s['object_box'])

        return {
            'image': self.transform(img),
            'human_box': torch.tensor(human_box),
            'object_box': torch.tensor(object_box),
            'hoi_id': s['hoi_id'],
            'label': torch.tensor(s['label']),
            'img_wh': torch.tensor([W, H], dtype = torch.float32)
        }

# V-COCO Dataset
# --------------------------------------------------------------------------------
#  V-COCO
#  Structure on disk:
#    v-coco/
#      data/
#        vcoco_train.json   vcoco_val.json   vcoco_test.json
#        instances_vcoco_all_2014.json    <-COCO-format instance file
#      coco/images/train2014/  val2014/
# --------------------------------------------─-----------------------------------

class VCOCODataset(Dataset):
    # V-COCO: 29 actions. Action index 26 = 'throw', each item is a (human_box, object_box) pair annotated with

    THROW_ACTION = 'throw'

    def __init__(self, vcoco_root, coco_root, split='train'):
        self.coco_root = coco_root
        self.split     = split
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.samples = self._load_vcoco(vcoco_root, split)

    def _load_vcoco(self, vcoco_root, split):
        # Load the V-COCO annotation struct
        vcoco = vu.load_vcoco(f'vcoco_{split}', vcoco_root)

        # Load COCO instance annotations to get per-image bboxes
        coco_ann_file = os.path.join(
            vcoco_root, 'data',
            f'instances_vcoco_all_2014.json'
        )
        with open(coco_ann_file) as f:
            coco_data = json.load(f)

        # Build ann_id → bbox lookup from COCO instances
        ann_to_bbox = {
            a['id']: a['bbox']   # [x, y, w, h] COCO format
            for a in coco_data['annotations']
        }
        img_to_file = {
            i['id']: i['file_name']
            for i in coco_data['images']
        }

        samples = []
        # Find the throw action struct
        throw_action = next(
            a for a in vcoco if a['action_name'] == self.THROW_ACTION
        )

        n = len(throw_action['image_id'])
        for i in range(n):
            img_id     = int(throw_action['image_id'][i])
            human_id   = int(throw_action['ann_id'][i])
            label      = int(throw_action['label'][i])  # 1=throw, -1=no throw

            # role_object_id: the thrown object annotation id (0 = no object)
            obj_ann_id = int(throw_action['role_object_id'][i, 0])  # column 0 = obj role

            if human_id not in ann_to_bbox:
                continue

            h_xywh = ann_to_bbox[human_id]
            h_box  = self._xywh_to_xyxy(h_xywh)

            # If no object annotated, create a dummy box (model learns no-object case)
            if obj_ann_id == 0 or obj_ann_id not in ann_to_bbox:
                o_box = np.array([0., 0., 1., 1.], dtype=np.float32)
            else:
                o_box = self._xywh_to_xyxy(ann_to_bbox[obj_ann_id])

            img_file = img_to_file[img_id]
            subfolder = 'train2014' if split == 'train' else 'val2014'
            img_path  = os.path.join(self.coco_root, 'images', subfolder, img_file)

            samples.append({
                'img_path':  img_path,
                'human_box': h_box,
                'object_box':o_box,
                'label':     float(label == 1),
            })

        return samples

    @staticmethod
    def _xywh_to_xyxy(bbox):
        x, y, w, h = bbox
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s['img_path']).convert('RGB')
        W, H = img.size

        return {
            'image':      self.transform(img),
            'human_box':  torch.tensor(s['human_box']),
            'object_box': torch.tensor(s['object_box']),
            'label':      torch.tensor(s['label']),
            'img_wh':     torch.tensor([W, H], dtype=torch.float32),
        }

# Pass 1 functionality:

# --------------------------
#  Object Detector
# --------------------------

class ObjectDetector(nn.Module):
    def __init__(self, score_thresh=0.5):
        super().__init__()
        self.detector = fasterrcnn_resnet50_fpn(
            weights =  FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.score_thresh = score_thresh

    def forward(self, images):
        self.detector.eval()
        with torch.no_grad():
            predictions = self.detector(images)

        filtered = []
        for pred in predictions:
            mask = pred['scores'] >= self.score_thresh
            filtered.append({
                'boxes':  pred['boxes'][mask],
                'labels': pred['labels'][mask],
                'scores': pred['scores'][mask],
            })
        return filtered

# ----------------------------
#  Extract Human Object Pairs
# ----------------------------
# COCO label IDs for objects that appear in throwing interactions
# These correspond to what Faster RCNN actually outputs
THROWABLE_COCO_LABELS = {
    29,  # frisbee
    32,  # sports ball
    33,  # kite
    34,  # baseball bat
    35,  # baseball glove
    36,  # skateboard
    37,  # surfboard
    38,  # tennis racket
    46,  # banana
    47,  # apple
    48,  # sandwich
    49,  # orange
    50,  # broccoli
    51,  # carrot
    52,  # hot dog
    53,  # pizza
    54,  # donut
    55,  # cake
    56,  # chair  — can be thrown in some contexts
    77,  # cell phone
    78,  # microwave
    84,  # book
    85,  # clock
    86,  # vase
    87,  # scissors
    88,  # teddy bear
    89,  # hair drier
    90,  # toothbrush
}

def extract_human_object_pairs(detections, person_label=1, allowed_object_labels = None):
    # Extract all (human, object) candidate pairs from detections.

    pairs = []
    boxes  = detections['boxes']
    labels = detections['labels']
    scores = detections['scores']

    person_idx = (labels == person_label).nonzero(as_tuple=True)[0]

    if allowed_object_labels is not None:
        object_mask = torch.zeros(len(labels), dtype=torch.bool)

        for lbl in allowed_object_labels:
            object_mask |= (labels == lbl)

        object_idx = object_mask.nonzero(as_tuple = True)[0]
    else:
        object_idx = (labels != person_label).nonzero(as_tuple=True)[0]

    for p in person_idx:
        for o in object_idx:
            pairs.append({
                'human_box':    boxes[p],
                'object_box':   boxes[o],
                'human_score':  scores[p],
                'object_score': scores[o],
                'object_label': labels[o],
            })
    return pairs

# Pass 2 functionality:

# --------------------------
#  Feature Extractor
# --------------------------

class FeatureExtractor(nn.Module):
    def __init__(self, output_size=(7, 7)):
        super().__init__()
        backbone = torchvision.models.resnet50(pretrained=True)
        # Use up to layer4
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten  = nn.Flatten()

    def forward(self, images, boxes):
        """
        images : [B, 3, H, W]
        boxes  : [B, 4]  xyxy, one box per image in the batch
        Returns: [B, 2048]
        """
        print(f"Image shape: {images.shape}, Boxes shape: {boxes.shape}")

        feat_map = self.features(images)   # [1, 2048, H', W']
        H, W     = images.shape[2], images.shape[3]
        spatial_scale = feat_map.shape[2] / H

        batch_idx = torch.arange(len(boxes), device = boxes.device).unsqueeze(1)
        rois = torch.cat([
            batch_idx.float(),  # batch index
            boxes
        ], dim=1)


        pooled = roi_align(feat_map, rois, output_size=(7, 7), spatial_scale=spatial_scale)

        return self.flatten(self.pool(pooled))  # [N, 2048]

# --------------------------
#  Interaction Classifier
# --------------------------

class InteractionHead(nn.Module):
    # Two-stream head: appearance (RoI features) + spatial (box geometry).

    def __init__(self, feature_dim=2048, spatial_dim=64, num_classes=1):
        super().__init__()

        # Appearance stream
        self.appearance_fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        # Spatial stream — encodes relative geometry
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * spatial_dim * spatial_dim, 512),
            nn.ReLU(),
        )

        # Fusion + classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),  # 1 = "throwing" binary, or N HOI classes
        )

    def forward(self, human_feat, object_feat, spatial_map):
        appearance = self.appearance_fc(
            torch.cat([human_feat, object_feat], dim=-1)
        )
        spatial    = self.spatial_conv(spatial_map)
        fused      = torch.cat([appearance, spatial], dim=-1)
        return self.classifier(fused)

# ------------------------------------------------------------
#  Build the Map for Spatial Relationship of Human and Object
# ------------------------------------------------------------

def build_spatial_map(human_boxes, object_boxes, map_size=64, img_size=640):
    """
    Two-channel binary mask: channel 0 = human, channel 1 = object.
    Captures relative position/scale for throwing pose reasoning.
    """
    B = human_boxes.shape[0]
    spatial = torch.zeros(B, 2, map_size, map_size, device = human_boxes.device)

    def fill_channel(ch, boxes):
        scaled = (boxes / img_size * map_size).long().clamp(0, map_size - 1)
        for i in range(B):
            x1, y1, x2, y2 = scaled[i]
            spatial[i, ch, y1:y2+1, x1:x2+1] = 1.0

    fill_channel(0, human_boxes)
    fill_channel(1, object_boxes)
    return spatial

# ---------------------------------
#  Model/Two Pass Class
# ---------------------------------

class TwoPassHOIDetector(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.extractor   = FeatureExtractor()
        self.interaction = InteractionHead(num_classes=num_classes)
        self.criterion   = nn.BCEWithLogitsLoss()

    def forward(self, images, human_boxes, object_boxes, spatial_maps):
        """
        Training / validation forward pass using GT boxes.

        images       : [B, 3, H, W]
        human_boxes  : [B, 4]
        object_boxes : [B, 4]
        spatial_maps : [B, 2, 64, 64]

        Returns logits [B, 1]  (squeeze before loss)
        """
        h_feat  = self.extractor(images, human_boxes)
        o_feat  = self.extractor(images, object_boxes)
        logits  = self.interaction(h_feat, o_feat, spatial_maps)
        return logits   # [B, 1]

    def compute_loss(self, logits, labels):
        """labels: [B] float tensor of 0/1"""
        return self.criterion(logits.squeeze(1), labels)

    @torch.no_grad()
    def predict(self, images, human_boxes, object_boxes, spatial_maps,
                threshold=0.5):
        self.eval()
        logits = self.forward(images, human_boxes, object_boxes, spatial_maps)
        probs  = torch.sigmoid(logits.squeeze(1))
        return probs, (probs >= threshold).long()

# ─────────────────────────────
#  Training Function per Epoch
# ─────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        imgs       = batch['image'].to(device)        # [B, 3, H, W]
        h_boxes    = batch['human_box'].to(device)    # [B, 4]
        o_boxes    = batch['object_box'].to(device)   # [B, 4]
        labels     = batch['label'].to(device)        # [B]

        # Build spatial maps for the batch
        spatial_maps = build_spatial_map(h_boxes, o_boxes).to(device) # [B, 2, 64, 64]

        logits = model(imgs, h_boxes, o_boxes, spatial_maps)
        loss   = model.compute_loss(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds   = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total   += len(labels)

    return total_loss / len(loader), correct / total

# ---------------------
#  Evaluation Function
# ---------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        images       = batch['image'].to(device)
        human_boxes  = batch['human_box'].to(device)
        object_boxes = batch['object_box'].to(device)
        labels       = batch['label'].float().to(device)

        spatial_maps = build_spatial_map(human_boxes, object_boxes).to(device)

        logits = model(images, human_boxes, object_boxes, spatial_maps)
        loss   = model.compute_loss(logits, labels)

        total_loss += loss.item()
        preds   = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total   += len(labels)

    return total_loss / len(loader), correct / total

# ---------------
#  DataLoader
# ---------------

def build_loaders(hico_root, vcoco_root = None, coco_root = None, batch_size=16):
    hico_train = HICODetDataset(hico_root, split='train')
    hico_val   = HICODetDataset(hico_root, split='test')

    # vcoco_train = VCOCODataset(vcoco_root, coco_root, split='train')
    # vcoco_val   = VCOCODataset(vcoco_root, coco_root, split='val')

    # Optionally concatenate both datasets for joint training
    # joint_train = ConcatDataset([hico_train, vcoco_train])

    train_loader = DataLoader(
        # joint_train,
        hico_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        hico_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, val_loader

# --------------------------
#  Main training script
# --------------------------

def train(train_loader, val_loader, epochs=5, lr=1e-4, checkpoint_dir = 'two_pass_checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TwoPassHOIDetector(num_classes=1).to(device)

    # Freeze the backbone for the first few epochs — only train the heads
    for param in model.extractor.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW([
        {'params': model.extractor.parameters(),   'lr': lr * 0.1},
        {'params': model.interaction.parameters(), 'lr': lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1:02d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        # Save checkpoint every epoch to Drive
        torch.save({
            'epoch':      epoch,
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
        }, f'{checkpoint_dir}/epoch_{epoch+1:02d}.pth')
    return model

if __name__ == '__main__':
    train_loader, val_loader = build_loaders(Path(__file__).parent / 'hico_20160224_det')
    trained_model = train(train_loader, val_loader)
