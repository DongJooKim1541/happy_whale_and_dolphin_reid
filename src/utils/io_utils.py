import os
import torch
import numpy as np


def ensure_output_dirs(*dirs):
    """Create output directories if they don't exist.

    Args:
        *dirs: Variable number of directory paths
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, save_path):
    """Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """Load model checkpoint.

    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        device: Device to load on

    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def save_gallery(gallery_embeddings, gallery_ids, save_path):
    """Save gallery embeddings and IDs.

    Args:
        gallery_embeddings: Gallery embeddings tensor
        gallery_ids: Individual IDs for gallery samples
        save_path: Path to save gallery
    """
    np.savez(
        save_path,
        embeddings=gallery_embeddings.cpu().numpy(),
        ids=gallery_ids
    )


def load_gallery(save_path, device='cuda'):
    """Load gallery embeddings and IDs.

    Args:
        save_path: Path to gallery file
        device: Device to load on

    Returns:
        embeddings: Gallery embeddings tensor
        ids: Individual IDs
    """
    data = np.load(save_path)
    embeddings = torch.from_numpy(data['embeddings']).float().to(device)
    ids = data['ids']
    return embeddings, ids
