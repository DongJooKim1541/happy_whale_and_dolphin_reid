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

# Weight of the species cross-entropy term relative to the triplet term:
#   total = ce_weight * CE + triplet
ce_loss_weight = float(os.getenv("CE_LOSS_WEIGHT", "0.01"))

# Number of hard negatives retrieved per anchor during in-batch mining.
hard_negatives_per_anchor = int(os.getenv("HARD_NEGATIVES_PER_ANCHOR", "1"))

# Distance threshold above which a query is judged a previously unseen individual.
new_id_threshold = float(os.getenv("NEW_ID_THRESHOLD", "0.1"))

# DataLoader worker processes.
num_workers = int(os.getenv("NUM_WORKERS", "0" if os.name == "nt" else "4"))

# Model configuration
model_name = os.getenv("MODEL_NAME", "resnet18")
pretrained = os.getenv("PRETRAINED", "1") not in ("0", "false", "False")
embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "512"))
num_classes = int(os.getenv("NUM_CLASSES", "30"))

# GPU
device_order = os.getenv("DEVICE_ORDER", "PCI_BUS_ID")
cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# Data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    "weight_dir", "output_dir", "map_k", "l2_distance", "ensure_directories_exist",
    "ce_loss_weight", "hard_negatives_per_anchor", "new_id_threshold",
    "num_workers", "model_name", "pretrained",
]
