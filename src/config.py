from torch.nn import PairwiseDistance

# Training configuration
batch_size = 64
num_train_triplets = 40000
num_valid_triplets = 20000
margin = 0.0001
epochs = 100
learning_rate = 1e-4
weight_decay = 0.0

# Model configuration
embedding_dimension = 512
num_classes = 30  # whale/dolphin species

# GPU
device_order = "PCI_BUS_ID"
cuda_visible_devices = "0"  # Default to GPU 0

# Data paths (can be overridden by environment variables)
train_root_dir = "./dataset/train/"
valid_root_dir = "./dataset/valid/"
train_csv_name = "./train_list.csv"
valid_csv_name = "./val_list.csv"
all_csv_name = "./all_list.csv"

# Checkpoint paths
weight_dir = "./weight/"
output_dir = "./output/"

# Metrics
map_k = 5  # MAP@K metric

l2_distance = PairwiseDistance(p=2)
