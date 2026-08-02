import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, Any


def ensure_output_dirs(*dirs: Union[str, Path]) -> None:
    """Create output directories if they don't exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, save_path: Union[str, Path]) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Union[str, Path],
                    device: str = 'cuda') -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def save_gallery(gallery_embeddings: torch.Tensor, gallery_ids: np.ndarray,
                 save_path: Union[str, Path]) -> None:
    """Save gallery embeddings and IDs."""
    np.savez(
        save_path,
        embeddings=gallery_embeddings.cpu().numpy(),
        ids=gallery_ids
    )


def load_gallery(save_path: Union[str, Path],
                 device: str = 'cuda') -> Tuple[torch.Tensor, np.ndarray]:
    """Load gallery embeddings and IDs."""
    data = np.load(save_path)
    embeddings = torch.from_numpy(data['embeddings']).float().to(device)
    ids = data['ids']
    return embeddings, ids
