
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import argparse
import csv
import random
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 4
LR = 1e-3
N_EPOCHS = 50
BACKBONE_SIZE = "base"  # Changed from "small" to "base" for better features
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ============================================================
# Mask Mapping (No Background)
# ============================================================

value_map = {
    100: 0,     # Trees
    200: 1,     # Lush Bushes
    300: 2,     # Dry Grass
    500: 3,     # Dry Bushes
    550: 4,     # Ground Clutter
    600: 5,     # Flowers
    700: 6,     # Logs
    800: 7,     # Rocks
    7100: 8,    # Landscape
    10000: 9    # Sky
}

N_CLASSES = len(value_map)  # 10 classes (no background)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr


# ============================================================
# Dice Loss for handling class imbalance
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


class CombinedLoss(nn.Module):
    """Combined CrossEntropy + Dice Loss."""
    def __init__(self, num_classes, class_weights=None, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        return self.ce_weight * self.ce(pred, target) + self.dice_weight * self.dice(pred, target)


# ============================================================
# Dataset with improved augmentation
# ============================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, height, width, transform=None, is_train=False):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.mask_dir = os.path.join(data_dir, "Segmentation")
        self.transform = transform
        self.ids = os.listdir(self.image_dir)
        self.is_train = is_train

        # Save resize dimensions
        self.H = height
        self.W = width
        
        # Normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.ids)
    
    def _apply_augmentations(self, image, mask):
        """Apply synchronized augmentations to both image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip (useful for aerial/satellite style imagery)
        if random.random() > 0.7:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation (-15 to +15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        
        # Random scale and crop
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            new_h = int(self.H * scale)
            new_w = int(self.W * scale)
            image = TF.resize(image, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)
            
            # Center crop back to original size
            image = TF.center_crop(image, [self.H, self.W])
            mask = TF.center_crop(mask, [self.H, self.W])
        
        return image, mask

    def __getitem__(self, idx):
        img_name = self.ids[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = convert_mask(mask)

        # Resize BOTH image and mask
        image = image.resize((self.W, self.H), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((self.W, self.H), Image.NEAREST)
        
        # Apply synchronized augmentations for training
        if self.is_train:
            image, mask = self._apply_augmentations(image, mask)
            # Color jitter (only on image)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        
        # Convert to tensor
        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

# ============================================================
# Segmentation Head
# ============================================================

class SegmentationHeadConvNeXt(nn.Module):
    """Improved segmentation head with multi-scale features and residual connections."""
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        hidden_dim = 256

        # Stem with batch norm
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        # Multi-scale feature extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )

        self.classifier = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        
        # Multi-scale features with residual
        f1 = self.block1(x) + x
        f2 = self.block2(x) + x
        f3 = self.block3(x) + x
        
        # Concatenate multi-scale features
        x = torch.cat([f1, f2, f3], dim=1)
        x = self.fusion(x)
        
        return self.classifier(x)


# ============================================================
# Metrics
# ============================================================

def compute_iou(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    target = target.view(-1)

    ious = []

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            continue
        ious.append((intersection / union).item())

    return np.mean(ious)


# ============================================================
# Main
# ============================================================

def compute_class_weights(data_dir, num_classes):
    """Compute class weights based on pixel frequency in training data."""
    print("Computing class weights from training data...")
    image_dir = os.path.join(data_dir, "Segmentation")
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    for img_name in tqdm(os.listdir(image_dir), desc="Analyzing masks"):
        mask_path = os.path.join(image_dir, img_name)
        mask = Image.open(mask_path)
        mask_arr = convert_mask(mask)
        for cls in range(num_classes):
            class_counts[cls] += np.sum(mask_arr == cls)
    
    # Compute inverse frequency weights
    total_pixels = class_counts.sum()
    class_freq = class_counts / total_pixels
    
    # Use inverse frequency with smoothing
    weights = 1.0 / (class_freq + 1e-6)
    weights = weights / weights.sum() * num_classes  # Normalize
    
    # Clip extreme weights
    weights = np.clip(weights, 0.5, 10.0)
    
    print(f"Class weights: {weights}")
    return torch.FloatTensor(weights)


def main():
    torch.backends.cudnn.benchmark = True
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, "..")

    parser = argparse.ArgumentParser(description="Train segmentation head with optional auto-resume")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(root_dir, "best_model.pth"),
        help="Path to model checkpoint for auto-resume and best-model saving"
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=os.path.join(root_dir, "training_results.csv"),
        help="CSV file to append epoch metrics (train_loss, val_loss, val_iou)"
    )
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        help="Start training from scratch, ignoring existing model"
    )
    parser.add_argument(
        "--last_model_path",
        type=str,
        default=os.path.join(root_dir, "last_model.pth"),
        help="Path to save the last trained model checkpoint"
    )
    args = parser.parse_args()

    train_dir = os.path.join(script_dir, "..", "Offroad_Segmentation_Training_Dataset", "train")
    val_dir = os.path.join(script_dir, "..", "Offroad_Segmentation_Training_Dataset", "val")

    # Image size (multiple of 14 for DINOv2) - using larger resolution
    w = int(((960 * 0.75) // 14) * 14)  # 714 -> 700
    h = int(((540 * 0.75) // 14) * 14)  # 405 -> 392

    # Create datasets with improved augmentation
    trainset = MaskDataset(train_dir, h, w, is_train=True)
    valset = MaskDataset(val_dir, h, w, is_train=False)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(trainset)}")
    print(f"Val samples: {len(valset)}")

    # ========================================================
    # Load DINOv2 backbone
    # ========================================================

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }

    backbone_name = f"dinov2_{backbone_archs[BACKBONE_SIZE]}"
    backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
    backbone.eval()
    backbone.to(DEVICE)

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # Get embedding size
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(DEVICE)

    with torch.no_grad():
        output = backbone.forward_features(imgs)["x_norm_patchtokens"]

    embed_dim = output.shape[2]

    classifier = SegmentationHeadConvNeXt(
        in_channels=embed_dim,
        out_channels=N_CLASSES,
        tokenW=w // 14,
        tokenH=h // 14
    ).to(DEVICE)

    # Auto-resume from existing best model (if present), else train from scratch
    if os.path.exists(args.model_path) and not args.fresh_start:
        try:
            classifier.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
            print(f"Loaded existing model for retraining: {args.model_path}")
        except Exception as e:
            print(f"Could not load existing model (architecture changed?): {e}")
            print("Training from scratch with new architecture.")
    else:
        print(f"Training from scratch with DINOv2 features.")

    # Compute class weights for handling imbalance
    class_weights = compute_class_weights(train_dir, N_CLASSES).to(DEVICE)
    
    # Combined loss: CrossEntropy + Dice
    criterion = CombinedLoss(N_CLASSES, class_weights=class_weights, ce_weight=0.4, dice_weight=0.6)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(classifier.parameters(), lr=LR, weight_decay=1e-4)
    
    # Learning rate scheduler - Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_iou = 0
    patience = 15  # Early stopping patience
    patience_counter = 0

    # Ensure CSV exists with header
    results_csv_dir = os.path.dirname(args.results_csv)
    if results_csv_dir:
        os.makedirs(results_csv_dir, exist_ok=True)
    if not os.path.exists(args.results_csv) or args.fresh_start:
        with open(args.results_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "learning_rate"])
        print(f"Created training results CSV: {args.results_csv}")

    # ========================================================
    # Training Loop
    # ========================================================

    for epoch in range(N_EPOCHS):

        classifier.train()
        train_loss = 0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Train]"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            with torch.no_grad():
                features = backbone.forward_features(imgs)["x_norm_patchtokens"]

            logits = classifier(features)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
        
        # Step the scheduler
        scheduler.step()

        # ================= Validation =================

        classifier.eval()
        val_loss = 0
        val_iou = []

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Val]"):
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                features = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(features)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

                iou = compute_iou(outputs, masks, N_CLASSES)
                val_iou.append(iou)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = np.mean(val_iou)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val IoU: {avg_val_iou:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        # Append epoch metrics to CSV
        with open(args.results_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{avg_train_loss:.6f}",
                f"{avg_val_loss:.6f}",
                f"{avg_val_iou:.6f}",
                f"{current_lr:.8f}",
            ])

        # Save last model (every epoch)
        torch.save(classifier.state_dict(), args.last_model_path)
        print(f"Saved last model to {args.last_model_path}")

        # Save best model
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            patience_counter = 0
            torch.save(classifier.state_dict(), args.model_path)
            print(f"Saved new best model to {args.model_path}!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                break

    print("\nTraining complete.")
    print(f"Last model saved to: {args.last_model_path}")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Training metrics CSV: {args.results_csv}")


if __name__ == "__main__":
    main()