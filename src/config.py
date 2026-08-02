"""Configuration with environment variable support"""
from typing import Optional
import os
from pathlib import Path
from torch.nn import PairwiseDistance

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Training configuration
batch_size = int(os.getenv("BATCH_SIZE", "64"))
num_train_triplets = int(os.getenv("NUM_TRAIN_TRIPLETS", "40000"))
num_valid_triplets = int(os.getenv("NUM_VALID_TRIPLETS", "20000"))
margin = float(os.getenv("MARGIN", "0.0001"))
epochs = int(os.getenv("EPOCHS", "100"))
learning_rate = float(os.getenv("LEARNING_RATE", "1e-4"))
weight_decay = float(os.getenv("WEIGHT_DECAY", "0.0"))

# Model configuration
embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "512"))
num_classes = int(os.getenv("NUM_CLASSES", "30"))

# GPU
device_order = os.getenv("DEVICE_ORDER", "PCI_BUS_ID")
cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# Data paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
train_root_dir = Path(os.getenv("TRAIN_ROOT_DIR", str(PROJECT_ROOT / "dataset" / "train")))
valid_root_dir = Path(os.getenv("VALID_ROOT_DIR", str(PROJECT_ROOT / "dataset" / "valid")))
train_csv_name = Path(os.getenv("TRAIN_CSV_NAME", str(PROJECT_ROOT / "train_list.csv")))
valid_csv_name = Path(os.getenv("VALID_CSV_NAME", str(PROJECT_ROOT / "val_list.csv")))
all_csv_name = Path(os.getenv("ALL_CSV_NAME", str(PROJECT_ROOT / "all_list.csv")))

# Checkpoint paths
weight_dir = Path(os.getenv("WEIGHT_DIR", str(PROJECT_ROOT / "weight")))
output_dir = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output")))

# Metrics
map_k = int(os.getenv("MAP_K", "5"))

l2_distance = PairwiseDistance(p=2)


def ensure_directories_exist() -> None:
    """Create required directories if they don't exist."""
    weight_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


__all__: list = [
    "batch_size", "num_train_triplets", "num_valid_triplets", "margin",
    "epochs", "learning_rate", "weight_decay", "embedding_dimension",
    "num_classes", "device_order", "cuda_visible_devices", "train_root_dir",
    "valid_root_dir", "train_csv_name", "valid_csv_name", "all_csv_name",
    "weight_dir", "output_dir", "map_k", "l2_distance", "ensure_directories_exist"
]
