import torch
import numpy as np
from typing import Tuple, List, Union


def knn(gallery: torch.Tensor, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """K-nearest neighbors search using L2 distance."""
    dist = torch.norm(gallery.unsqueeze(1) - query.unsqueeze(0), dim=2, p=2)
    knn_result = dist.topk(k, largest=False, dim=0)
    return knn_result.values, knn_result.indices


def calculate_map(gallery_ids: Union[np.ndarray, List], pred_ids: Union[np.ndarray, List],
                  pred_distances: Union[torch.Tensor, np.ndarray],
                  query_ids: Union[np.ndarray, List], margin: float = 0.1,
                  k: int = 5) -> Tuple[float, int, int]:
    """Calculate Mean Average Precision @ K."""
    gallery_ids = np.array(gallery_ids)
    query_ids = np.array(query_ids)
    pred_ids = np.array(pred_ids)

    predicted_ids = gallery_ids[pred_ids].copy()
    predicted_ids[pred_distances > margin] = 'new_id'

    num_new = 0
    num_matched = 0
    for i in range(len(query_ids)):
        if query_ids[i] not in gallery_ids:
            query_ids[i] = 'new_id'
            num_new += 1
        else:
            num_matched += 1

    scores = np.zeros(len(query_ids))
    for i in range(k):
        predicted_id = predicted_ids[k - i - 1]
        correct = query_ids == predicted_id
        scores[correct] = 1 / (k - i)

    map_score = scores.mean()
    return map_score, num_new, num_matched


def hard_negative_mining(embeddings: torch.Tensor, individual_ids: Union[torch.Tensor, np.ndarray],
                         k: int = 1) -> List[torch.Tensor]:
    """Hard negative mining: select hard negatives for each anchor."""
    num_samples = len(embeddings)
    hard_negative_indices = []

    for i in range(num_samples):
        anchor_embedding = embeddings[i:i+1]
        anchor_id = individual_ids[i]

        positive_mask = (individual_ids == anchor_id)
        negative_mask = (individual_ids != anchor_id)

        if not negative_mask.any():
            continue

        negative_embeddings = embeddings[negative_mask]
        distances = torch.norm(
            anchor_embedding.unsqueeze(1) - negative_embeddings.unsqueeze(0),
            dim=2, p=2
        ).squeeze(0)

        _, hard_indices = torch.topk(distances, min(k, len(distances)), largest=False)
        hard_negative_indices.append(hard_indices)

    return hard_negative_indices
