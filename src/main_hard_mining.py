"""Hard negative mining for triplet loss training.

Hard negative mining selects the hardest negative samples (closest to anchor)
to encourage the model to learn more discriminative features.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import (
    batch_size, num_train_triplets, margin, learning_rate,
    weight_decay, embedding_dimension, num_classes,
    cuda_visible_devices, train_root_dir, valid_root_dir,
    train_csv_name, valid_csv_name, weight_dir
)
from models import ResNetTriplet
from data.whale_dataset import get_dataloaders
from utils import TripletLoss, knn, calculate_map, ensure_output_dirs


def forward_pass(imgs, model):
    """Forward pass through model."""
    embeddings, preds = model(imgs)
    return embeddings, preds


def select_hard_negatives(gallery_embeddings, gallery_ids, anchor_embeddings, anchor_ids, k=1):
    """Select hard negatives using KNN.

    Hard negative: closest sample with different individual_id
    """
    batch_size = len(anchor_embeddings)
    hard_negatives = torch.zeros_like(anchor_embeddings)

    dist = torch.norm(
        gallery_embeddings.unsqueeze(1) - anchor_embeddings.unsqueeze(0),
        dim=2, p=2
    )
    _, indices = dist.topk(k * batch_size, largest=False, dim=0)
    indices = indices.cpu()

    gallery_ids = np.array(gallery_ids)
    anchor_ids = np.array(anchor_ids)

    for b in range(batch_size):
        for idx in indices[:, b]:
            if gallery_ids[idx] != anchor_ids[b]:
                hard_negatives[b] = gallery_embeddings[idx]
                break

    return hard_negatives


def train_epoch_hard_mining(model, optimizer, dataloader, triplet_loss, ce_loss, device):
    """Train one epoch with hard negative mining.

    Args:
        model: Model instance
        optimizer: Optimizer
        dataloader: Training dataloader
        triplet_loss: Triplet loss function
        ce_loss: Cross-entropy loss function
        device: Device to train on

    Returns:
        avg_triplet_loss, avg_ce_loss, avg_acc
    """
    model.train()

    total_triplet_loss = 0.0
    total_ce_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch_sample in tqdm(dataloader, desc="Training (Hard Mining)"):
        anchor_imgs = batch_sample['anchor_img'].float().to(device)
        positive_imgs = batch_sample['positive_img'].float().to(device)
        anchor_species = batch_sample['anchor_species'].long().to(device)
        positive_species = batch_sample['positive_species'].long().to(device)

        all_imgs = torch.cat((anchor_imgs, positive_imgs))
        batch_size = anchor_imgs.shape[0]

        # Forward pass
        embeddings, preds = forward_pass(all_imgs, model)

        # Split
        anchor_embeddings = embeddings[:batch_size]
        positive_embeddings = embeddings[batch_size:]
        anchor_preds = preds[:batch_size]
        positive_preds = preds[batch_size:]

        # Hard negative mining
        gallery = torch.cat((anchor_embeddings, positive_embeddings))
        gallery_ids = np.concatenate([
            batch_sample['individual_id'],
            batch_sample['individual_id']
        ])

        hard_negatives = select_hard_negatives(
            gallery, gallery_ids,
            anchor_embeddings, batch_sample['individual_id'],
            k=batch_size * 2
        ).to(device)

        # Loss
        t_loss = triplet_loss(anchor_embeddings, positive_embeddings, hard_negatives)
        ce_loss_val = ce_loss(anchor_preds, anchor_species) + ce_loss(positive_preds, positive_species)
        total_loss = 0.01 * ce_loss_val + t_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Metrics
        total_triplet_loss += t_loss.item()
        total_ce_loss += ce_loss_val.item()
        acc = ((anchor_preds.max(dim=1)[1] == anchor_species).float().mean())
        total_acc += acc.item()
        num_batches += 1

    return (
        total_triplet_loss / num_batches,
        total_ce_loss / num_batches,
        total_acc / num_batches
    )


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_output_dirs(weight_dir)

    # Model
    model = ResNetTriplet(
        model_name="resnet18",
        embedding_dimension=embedding_dimension,
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    triplet_loss_fn = TripletLoss(margin=margin)
    ce_loss_fn = nn.CrossEntropyLoss()

    # Data
    dataloaders = get_dataloaders(
        train_root_dir=train_root_dir,
        valid_root_dir=valid_root_dir,
        train_csv_name=train_csv_name,
        valid_csv_name=valid_csv_name,
        num_train_triplets=num_train_triplets,
        num_valid_triplets=20000,
        batch_size=batch_size,
        num_workers=4
    )

    # Training with hard mining
    num_epochs = 100
    for epoch in range(num_epochs):
        triplet_loss, ce_loss, acc = train_epoch_hard_mining(
            model, optimizer, dataloaders['train'],
            triplet_loss_fn, ce_loss_fn, device
        )

        print(f"\nEpoch {epoch+1}/{num_epochs} (Hard Mining)")
        print(f"  Triplet Loss: {triplet_loss:.6f}")
        print(f"  CE Loss: {ce_loss:.6f}")
        print(f"  Species Accuracy: {acc*100:.2f}%")

        # Save checkpoint
        checkpoint_path = os.path.join(weight_dir, f'model_hard_mining_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    print("\nTraining with hard mining complete!")
