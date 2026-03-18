import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou, mask_iou
from .load_config_file import ConfigLoader


class PerceptionDifficultyEvaluator:
    """Evaluates perception difficulty of test images using YOLO models."""

    def __init__(self, config_file: str, use_case: str, cycle: int = None):
        self.config_file = config_file
        self.use_case = use_case
        self.cycle = cycle

    def evaluate_detection(self, image_path: str, label_path: str) -> float:
        """
        Compute perception difficulty score for an image based on (1 - mean CIoU).
        """
        config = ConfigLoader(self.config_file).load()

        MODEL_PATH = config['best_model'][f'{self.use_case}']
        print(f"Loading model from: {MODEL_PATH}")
        TEST_IMAGES = config['test_images'][f'{self.use_case}']
        TEST_LABELS = config['test_labels'][f'{self.use_case}']
        img_path = os.path.join(TEST_IMAGES, image_path)
        label_file = os.path.join(TEST_LABELS, label_path)

        model = YOLO(MODEL_PATH)

        results = model(img_path)
        pred_boxes = results[0].obb.xyxy.cpu()  # x1, y1, x2, y2
        image_shape = results[0].orig_shape  # (height, width)
        gt_boxes = self.load_gt_boxes(label_file, image_shape)

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            return 0.0

        if len(gt_boxes) == 0 and len(pred_boxes) > 0:
            return 1.0

        if len(gt_boxes) > 0 and len(pred_boxes) == 0:
            return 1.0

        ciou_matrix = self.compute_ciou_matrix(pred_boxes, gt_boxes)
        mean_ciou = ciou_matrix.mean().item()
        difficulty_score = 1 - mean_ciou
        return difficulty_score

    def evaluate_segmentation(self, image_path: str, label_path: str) -> float:
        """
        Calcule le score de difficulté basé sur (1 - Mask IoU moyen des meilleurs matchs).
        """
        config = ConfigLoader(self.config_file).load()

        TEST_IMAGES = config['merging'][f'{self.use_case}']['merged_data']
        TEST_LABELS = config['merging'][f'{self.use_case}']['merged_data']
        print(f"*********** TEST_IMAGES: {TEST_IMAGES}")
        print(f"*********** TEST_LABELS: {TEST_LABELS}")
        img_full_path = os.path.join(f"{TEST_IMAGES}_cycle{self.cycle}", image_path)
        label_full_path = os.path.join(f"{TEST_LABELS}_cycle{self.cycle}", label_path)

        if int(self.cycle) == 0:
            print("Cycle 0 Before Fine-tuning...")
            MODEL_PATH = config['best_model'][f'{self.use_case}']
        else:
            print(f"Cycle {self.cycle} Fine-tuning...")
            MODEL_PATH = os.path.join("fine-tuning", self.use_case, f"cycle{self.cycle}_{self.use_case}", "weights", "best.pt")

        print(f"*********** Loading model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)

        results = model(img_full_path, retina_masks=True, verbose=False)

        if results[0].masks is not None:
            pred_masks = results[0].masks.data
        else:
            pred_masks = torch.empty((0, results[0].orig_shape[0], results[0].orig_shape[1]))

        image_shape = results[0].orig_shape  # (height, width)

        device = pred_masks.device
        gt_masks = self.load_gt_masks(label_full_path, image_shape, device=device)

        num_gt = gt_masks.shape[0]
        num_pred = pred_masks.shape[0]

        if num_gt == 0 and num_pred == 0:
            return 0.0

        if num_gt == 0 and num_pred > 0:
            return 1.0
        if num_gt > 0 and num_pred == 0:
            return 1.0

        H, W = image_shape[0], image_shape[1]

        def to_flat_masks(masks, name):
            if masks is None:
                return torch.empty((0, H * W), device='cpu')
            if not isinstance(masks, torch.Tensor):
                masks = torch.tensor(masks)
            if masks.ndim == 3:
                n, h, w = masks.shape
                if h * w != H * W:
                    raise ValueError(f"{name} has unexpected spatial shape {masks.shape} vs image shape {image_shape}")
                return masks.reshape(n, -1).float()
            elif masks.ndim == 2:
                n, px = masks.shape
                if px == H * W:
                    return masks.float()
                if n == H * W:
                    return masks.T.float()
                raise ValueError(f"{name} has unexpected 2D shape {masks.shape} for image shape {image_shape}")
            elif masks.numel() == 0:
                return masks.reshape(0, H * W).to(dtype=torch.float)
            else:
                raise ValueError(f"{name} has unsupported ndim={masks.ndim}")

        try:
            gt_flat = to_flat_masks(gt_masks, 'gt_masks').to(device=pred_masks.device)
            pred_flat = to_flat_masks(pred_masks, 'pred_masks')
        except Exception as e:
            print(f"Error preparing masks for IoU computation: {e}")
            raise

        if gt_flat.size(1) != pred_flat.size(1):
            raise RuntimeError(f"Mask pixel dimension mismatch: gt {gt_flat.size(1)} vs pred {pred_flat.size(1)} for image {image_path} with orig_shape {image_shape}")

        iou_matrix = mask_iou(gt_flat, pred_flat)

        max_ious_per_gt, _ = torch.max(iou_matrix, dim=1)
        mean_iou = max_ious_per_gt.mean().item()

        difficulty_score = 1.0 - mean_iou
        return difficulty_score

    @staticmethod
    def load_gt_boxes(label_path, img_shape):
        h, w = img_shape
        boxes = []

        if not os.path.exists(label_path):
            return torch.empty((0, 4))

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                coords = list(map(float, parts[1:]))

                xs = coords[0::2]
                ys = coords[1::2]

                xs = [x * w for x in xs]
                ys = [y * h for y in ys]

                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                boxes.append([x1, y1, x2, y2])

        return torch.tensor(boxes)

    @staticmethod
    def compute_ciou_matrix(pred_boxes, gt_boxes):
        """
        Retourne la matrice CIoU (n_pred, n_gt) entre tous les couples de boxes.
        """
        n_pred = pred_boxes.shape[0]
        n_gt = gt_boxes.shape[0]
        ciou_matrix = torch.zeros((n_pred, n_gt))
        for i in range(n_pred):
            for j in range(n_gt):
                ciou_matrix[i, j] = bbox_iou(pred_boxes[i].unsqueeze(0), gt_boxes[j].unsqueeze(0), CIoU=True)
        return ciou_matrix

    @staticmethod
    def load_gt_masks(label_path, img_shape, device='cpu'):
        """
        Charge les polygones du fichier txt et les convertit en masques binaires.
        Retourne un tenseur de forme (N_objects, Height, Width).
        """
        h, w = img_shape
        masks_list = []

        if not os.path.exists(label_path):
            return torch.empty((0, h, w), device=device)

        with open(label_path, "r") as f:
            lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                coords = list(map(float, parts[1:]))
                mask = np.zeros((h, w), dtype=np.uint8)
                poly = np.array(coords).reshape(-1, 2)
                poly[:, 0] *= w
                poly[:, 1] *= h
                poly = poly.astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
                masks_list.append(mask)

        if not masks_list:
            return torch.empty((0, h, w), device=device)

        return torch.tensor(np.array(masks_list), device=device)
