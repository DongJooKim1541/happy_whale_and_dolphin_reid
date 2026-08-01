import os
import torch
import numpy as np
from tqdm import tqdm

from config import (
    batch_size, num_valid_triplets, embedding_dimension, num_classes,
    cuda_visible_devices, train_root_dir, valid_root_dir,
    train_csv_name, valid_csv_name, map_k, weight_dir
)
from models import ResNetTriplet
from data.whale_dataset import get_dataloaders
from utils import knn, calculate_map


@torch.no_grad()
def make_gallery(model, gallery_dataloader, device):
    """Create gallery embeddings from training set.

    Args:
        model: Model instance
        gallery_dataloader: Gallery dataloader
        device: Device to compute on

    Returns:
        gallery_embeddings: Gallery embeddings tensor
        gallery_ids: Individual IDs for gallery samples
    """
    model.eval()

    gallery_embeddings_list = []
    gallery_ids_list = []

    for batch_sample in tqdm(gallery_dataloader, desc="Creating gallery"):
        imgs = batch_sample['anchor_img'].to(device)
        embeddings, _ = model(imgs)

        gallery_embeddings_list.append(embeddings.cpu())
        gallery_ids_list.extend(batch_sample['individual_id'])

    gallery_embeddings = torch.cat(gallery_embeddings_list)
    return gallery_embeddings, np.array(gallery_ids_list)


@torch.no_grad()
def evaluate(model, gallery_embeddings, gallery_ids, valid_dataloader, device, margin=0.1):
    """Evaluate model on validation set using MAP@K metric.

    Args:
        model: Model instance
        gallery_embeddings: Gallery embeddings
        gallery_ids: Individual IDs for gallery
        valid_dataloader: Validation dataloader
        device: Device to compute on
        margin: Distance threshold for 'new_individual' classification

    Returns:
        map_score: Mean Average Precision @ K
        num_new: Number of new individuals detected
        num_matched: Number of matched individuals
    """
    model.eval()

    gallery_embeddings = gallery_embeddings.to(device)
    total_map = 0.0
    total_new = 0
    total_matched = 0
    num_queries = 0

    for batch_sample in tqdm(valid_dataloader, desc="Evaluating"):
        query_imgs = batch_sample['anchor_img'].to(device)
        query_ids = batch_sample['individual_id']

        # Get embeddings
        query_embeddings, _ = model(query_imgs)

        # KNN search
        distances, indices = knn(gallery_embeddings, query_embeddings, k=map_k)

        # Get top-1 prediction for MAP calculation
        pred_distances = distances[0].cpu().numpy()
        pred_indices = indices[0].cpu().numpy()

        # Calculate MAP
        map_score, num_new, num_matched = calculate_map(
            gallery_ids, pred_indices, pred_distances,
            query_ids, margin=margin, k=map_k
        )

        total_map += map_score
        total_new += num_new
        total_matched += num_matched
        num_queries += len(query_ids)

    avg_map = total_map / len(valid_dataloader)
    return avg_map, total_new, total_matched


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ResNetTriplet(
        model_name="resnet18",
        embedding_dimension=embedding_dimension,
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(weight_dir, 'model_epoch_100.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    # Data
    dataloaders = get_dataloaders(
        train_root_dir=train_root_dir,
        valid_root_dir=valid_root_dir,
        train_csv_name=train_csv_name,
        valid_csv_name=valid_csv_name,
        num_train_triplets=num_valid_triplets,
        num_valid_triplets=num_valid_triplets,
        batch_size=batch_size,
        num_workers=4
    )

    # Create gallery
    print("\nCreating gallery...")
    gallery_embeddings, gallery_ids = make_gallery(model, dataloaders['gallery'], device)
    print(f"Gallery size: {len(gallery_embeddings)} samples")

    # Evaluate
    print("\nEvaluating on validation set...")
    map_score, num_new, num_matched = evaluate(
        model, gallery_embeddings, gallery_ids,
        dataloaders['valid'], device, margin=0.1
    )

    print(f"\n=== Results ===")
    print(f"MAP@{map_k}: {map_score:.4f}")
    print(f"Matched individuals: {num_matched}")
    print(f"New individuals: {num_new}")
