
 Semantic Scene Segmentation

A semantic segmentation model for off-road/rover imagery using **DINOv2 Vision Transformer** backbone with a custom ConvNeXt-style decoder head.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)


## Architecture

```
Image → DINOv2 ViT-Base (frozen) → Patch Tokens → Segmentation Head → Prediction
```

- **Backbone**: DINOv2 `vitb14_reg` (768-dim embeddings, frozen)
- **Decoder**: Multi-scale ConvNeXt head (3×3, 5×5, 7×7 depthwise convs + residual)
- **Loss**: Combined CrossEntropy (40%) + Dice Loss (60%) with class weighting

## Classes 

| ID | Raw Value | Class |
|----|-----------|-------|
| 0 | 100 | Trees |
| 1 | 200 | Lush Bushes |
| 2 | 300 | Dry Grass |
| 3 | 500 | Dry Bushes |
| 4 | 550 | Ground Clutter |
| 5 | 600 | Flowers |
| 6 | 700 | Logs |
| 7 | 800 | Rocks |
| 8 | 7100 | Landscape |
| 9 | 10000 | Sky |


Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Semantic-Scene-Segmentation.git
cd Semantic-Scene-Segmentation

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies (with CUDA support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Prepare Dataset

```
Dataset/
├── train/
│   ├── Color_Images/    # RGB images (.png)
│   └── Segmentation/    # Mask images (.png)
├── val/
│   ├── Color_Images/
│   └── Segmentation/
└── test/
    ├── Color_Images/
    └── Segmentation/
```

### 3. Train Model

```bash
# Train from scratch
python model_scripts/train_segmentation.py --fresh_start

# Continue training from checkpoint
python model_scripts/train_segmentation.py
```

**Training Options:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Best model save path | `best_model.pth` |
| `--last_model_path` | Last epoch model path | `last_model.pth` |
| `--results_csv` | Training metrics CSV | `training_results.csv` |
| `--fresh_start` | Ignore checkpoint, train fresh | `False` |

### 4. Test Model

```bash
python model_scripts/test_segmentation.py
```

**Testing Options:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Model weights path | `best_model.pth` |
| `--data_dir` | Test dataset path | `Offroad_Segmentation_testImages` |
| `--output_dir` | Predictions output | `predictions` |
| `--batch_size` | Inference batch size | `2` |

## Training Features

- **DINOv2 Backbone** - Pre-trained ViT-Base with frozen weights
- **Class-weighted Loss** - Handles imbalanced classes  
- **Dice + CrossEntropy** - Combined loss for better segmentation
- **Cosine Annealing** - LR scheduler with warm restarts
- **Data Augmentation** - Flip, rotation, scale, color jitter
- **Early Stopping** - Patience of 15 epochs
- **Gradient Clipping** - Max norm of 1.0
 Configuration

Edit `model_scripts/train_segmentation.py`:

```python
BATCH_SIZE = 4
LR = 1e-3
N_EPOCHS = 50
BACKBONE_SIZE = "base"  # small, base, large, giant
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Acknowledgments

- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI
- Hackster.io Hackathon

=======
# Semantic-Scene-Segmentation
This project implements a semantic segmentation pipeline that performs pixel-level classification of images. The model is trained using PyTorch and evaluates performance using IoU and Dice metrics.  The system processes input images and generates color-coded segmentation masks where each pixel is assigned a semantic class .
>>>>>>> c3133231bd8508e8a0cae05281876e8d9542b873
