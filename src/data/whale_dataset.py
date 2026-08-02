import os
from typing import Optional, Dict, List, Tuple, Callable, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import pil_loader


class TripletWhaleDataset(Dataset):
    """Whale/dolphin re-identification dataset with triplet sampling."""

    def __init__(self, root_dir: str, csv_name: str, num_triplets: int,
                 transform: Optional[Callable] = None, train: bool = True) -> None:
        """
        Args:
            root_dir (str): Root directory containing images
            csv_name (str): Path to CSV with columns: individual_id, image, species
            num_triplets (int): Number of triplets to generate
            transform (callable): Image transformations
            train (bool): Training mode (for triplet generation)
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_name)
        self.num_triplets = num_triplets
        self.transform = transform
        self.train = train
        self.training_triplets = self.generate_triplets()

    def generate_triplets(self) -> List[List[Any]]:
        """Generate triplet samples (anchor, positive, negative)."""

        def make_whale_class_dict(df: pd.DataFrame) -> Dict[str, List[Tuple[str, str, str]]]:
            """Create dict mapping individual_id -> [(id, image_path, species), ...]"""
            whale_classes: Dict[str, List[Tuple[str, str, str]]] = {}
            for idx, row in df.iterrows():
                individual_id = row['individual_id']
                if individual_id not in whale_classes:
                    whale_classes[individual_id] = []
                whale_classes[individual_id].append(
                    (individual_id, row['image'], row['species'])
                )
            return whale_classes

        triplets = []
        whale_classes = make_whale_class_dict(self.df)
        all_ids = self.df['individual_id'].unique()

        for individual_id in all_ids:
            samples = whale_classes[individual_id]
            num_samples = len(samples)

            if num_samples < 2:
                continue  # Skip individuals with only 1 sample

            for i in range(num_samples):
                anchor_idx = i
                # Sample positive from same individual (different from anchor)
                positive_idx = np.random.randint(0, num_samples)
                while anchor_idx == positive_idx:
                    positive_idx = np.random.randint(0, num_samples)

                anchor_id, anchor_image, anchor_species = samples[anchor_idx]
                _, positive_image, positive_species = samples[positive_idx]

                triplets.append([individual_id, anchor_image, positive_image, anchor_species, positive_species])

        return triplets

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return triplet sample."""
        individual_id, anchor_image, positive_image, anchor_species, positive_species = self.training_triplets[idx]

        # Load images
        anchor_path = os.path.join(self.root_dir, str(anchor_image))
        positive_path = os.path.join(self.root_dir, str(positive_image))

        anchor_img = pil_loader(anchor_path)
        positive_img = pil_loader(positive_path)

        sample = {
            'anchor_img': anchor_img,
            'positive_img': positive_img,
            'individual_id': individual_id,
            'anchor_species': anchor_species,
            'positive_species': positive_species,
        }

        if self.transform:
            sample['anchor_img'] = self.transform(sample['anchor_img'])
            sample['positive_img'] = self.transform(sample['positive_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)


def get_dataloaders(train_root_dir, valid_root_dir, train_csv_name, valid_csv_name,
                    num_train_triplets, num_valid_triplets, batch_size, num_workers):
    """Create train, validation, and gallery dataloaders.

    Returns:
        dict: Dictionary with 'train', 'valid', 'gallery' dataloaders
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'gallery': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }

    datasets = {
        'train': TripletWhaleDataset(
            root_dir=train_root_dir,
            csv_name=train_csv_name,
            num_triplets=num_train_triplets,
            transform=data_transforms['train'],
            train=True
        ),
        'valid': TripletWhaleDataset(
            root_dir=valid_root_dir,
            csv_name=valid_csv_name,
            num_triplets=num_valid_triplets,
            transform=data_transforms['valid'],
            train=False
        ),
        'gallery': TripletWhaleDataset(
            root_dir=train_root_dir,
            csv_name=train_csv_name,
            num_triplets=num_train_triplets,
            transform=data_transforms['gallery'],
            train=False
        )
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        'valid': torch.utils.data.DataLoader(
            datasets['valid'], batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        'gallery': torch.utils.data.DataLoader(
            datasets['gallery'], batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    }

    return dataloaders
