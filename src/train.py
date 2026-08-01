import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import (
    batch_size, num_train_triplets, num_valid_triplets, margin,
    learning_rate, weight_decay, embedding_dimension, num_classes,
    cuda_visible_devices, train_root_dir, valid_root_dir,
    train_csv_name, valid_csv_name, weight_dir
)
from models import ResNetTriplet
from data.whale_dataset import get_dataloaders
from utils import TripletLoss, ensure_output_dirs


def forward_pass(imgs, model):
    """Forward pass through model.

    Args:
        imgs: Concatenated [anchor, positive] images
        model: Model instance
        batch_size: Size of anchor batch (determines split point)

    Returns:
        anchor_embeddings, anchor_pred: Embeddings and predictions for anchors
        positive_embeddings, positive_pred: Embeddings and predictions for positives
    """
    embeddings, preds = model(imgs)
    return embeddings, preds


def knn_hard_negatives(gallery_embeddings, gallery_ids, anchor_embeddings, anchor_ids, k=1):
    """Select hard negatives using KNN.

    Args:
        gallery_embeddings: All embeddings
        gallery_ids: Individual IDs
        anchor_embeddings: Current batch anchor embeddings
        anchor_ids: Current batch anchor IDs
        k: Number of hard negatives per anchor

    Returns:
        hard_negative_embeddings: Selected hard negative embeddings
    """
    batch_size = len(anchor_embeddings)
    hard_negative_embeddings = torch.zeros_like(anchor_embeddings)

    # KNN search
    dist = torch.norm(
        gallery_embeddings.unsqueeze(1) - anchor_embeddings.unsqueeze(0),
        dim=2, p=2
    )
    _, indices = dist.topk(k * batch_size, largest=False, dim=0)
    indices = indices.cpu()

    gallery_ids = np.array(gallery_ids)
    anchor_ids = np.array(anchor_ids)

    for b in range(batch_size):
        # Select first hard negative that differs from anchor ID
        for idx in indices[:, b]:
            if gallery_ids[idx] != anchor_ids[b]:
                hard_negative_embeddings[b] = gallery_embeddings[idx]
                break

    return hard_negative_embeddings


def train_epoch(model, optimizer, dataloader, triplet_loss, ce_loss, device):
    """Train for one epoch.

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

    for batch_sample in tqdm(dataloader, desc="Training"):
        anchor_imgs = batch_sample['anchor_img'].float().to(device)
        positive_imgs = batch_sample['positive_img'].float().to(device)
        anchor_species = batch_sample['anchor_species'].long().to(device)
        positive_species = batch_sample['positive_species'].long().to(device)

        # Concatenate anchor and positive for single forward pass
        all_imgs = torch.cat((anchor_imgs, positive_imgs))
        batch_size = anchor_imgs.shape[0]

        # Forward pass
        embeddings, preds = forward_pass(all_imgs, model)

        # Split outputs
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

        negative_embeddings = knn_hard_negatives(
            gallery, gallery_ids,
            anchor_embeddings, batch_sample['individual_id'],
            k=batch_size * 2
        ).to(device)

        # Calculate losses
        t_loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        ce_loss_val = ce_loss(anchor_preds, anchor_species) + ce_loss(positive_preds, positive_species)

        # Combined loss
        total_loss = 0.01 * ce_loss_val + t_loss

        # Backward pass
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
    # Setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_output_dirs(weight_dir)

    # Model, optimizer, loss
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
        num_valid_triplets=num_valid_triplets,
        batch_size=batch_size,
        num_workers=4
    )

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        triplet_loss, ce_loss, acc = train_epoch(
            model, optimizer, dataloaders['train'],
            triplet_loss_fn, ce_loss_fn, device
        )

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Triplet Loss: {triplet_loss:.6f}")
        print(f"  CE Loss: {ce_loss:.6f}")
        print(f"  Species Accuracy: {acc*100:.2f}%")

        # Save checkpoint
        checkpoint_path = os.path.join(weight_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    print("\nTraining complete!")
