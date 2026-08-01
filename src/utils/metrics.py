import torch
import numpy as np


def knn(gallery, query, k=5):
    """K-nearest neighbors search using L2 distance.

    Args:
        gallery: Gallery embeddings (num_gallery, embedding_dim)
        query: Query embeddings (num_query, embedding_dim)
        k: Number of nearest neighbors

    Returns:
        distances: L2 distances to k nearest neighbors (k, num_query)
        indices: Indices of k nearest neighbors (k, num_query)
    """
    dist = torch.norm(gallery.unsqueeze(1) - query.unsqueeze(0), dim=2, p=2)
    knn_result = dist.topk(k, largest=False, dim=0)

    return knn_result.values, knn_result.indices


def calculate_map(gallery_ids, pred_ids, pred_distances, query_ids, margin=0.1, k=5):
    """Calculate Mean Average Precision @ K.

    Args:
        gallery_ids: Individual IDs in gallery set (num_gallery,)
        pred_ids: Predicted individual IDs for queries (num_query,)
        pred_distances: Distances to predictions (num_query,)
        query_ids: Ground truth individual IDs for queries (num_query,)
        margin: Distance threshold for 'new_individual' classification
        k: Number of top-k predictions to consider

    Returns:
        map_score: Mean Average Precision
        num_new: Count of new individuals detected
        num_matched: Count of matched individuals
    """
    gallery_ids = np.array(gallery_ids)
    query_ids = np.array(query_ids)
    pred_ids = np.array(pred_ids)

    # Mark predictions beyond margin as 'new_individual'
    predicted_ids = gallery_ids[pred_ids].copy()
    predicted_ids[pred_distances > margin] = 'new_id'

    # Mark query IDs not in gallery as 'new_individual'
    num_new = 0
    num_matched = 0
    for i in range(len(query_ids)):
        if query_ids[i] not in gallery_ids:
            query_ids[i] = 'new_id'
            num_new += 1
        else:
            num_matched += 1

    # Calculate AP@K
    scores = np.zeros(len(query_ids))
    for i in range(k):
        predicted_id = predicted_ids[k - i - 1]
        correct = query_ids == predicted_id
        scores[correct] = 1 / (k - i)

    map_score = scores.mean()

    return map_score, num_new, num_matched


def hard_negative_mining(embeddings, individual_ids, k=1):
    """Hard negative mining: select hard negatives for each anchor.

    Returns indices of hard negatives where:
    ||f(anchor) - f(positive)|| > ||f(anchor) - f(negative)||

    Args:
        embeddings: All embeddings (num_samples, embedding_dim)
        individual_ids: Individual ID for each embedding (num_samples,)
        k: Number of hard negatives per anchor

    Returns:
        hard_negative_indices: Hard negative indices for each sample
    """
    num_samples = len(embeddings)
    hard_negative_indices = []

    for i in range(num_samples):
        anchor_embedding = embeddings[i:i+1]
        anchor_id = individual_ids[i]

        # Find positives and negatives
        positive_mask = (individual_ids == anchor_id)
        negative_mask = (individual_ids != anchor_id)

        if not negative_mask.any():
            continue  # No negatives available

        # Calculate distances to all negatives
        negative_embeddings = embeddings[negative_mask]
        distances = torch.norm(
            anchor_embedding.unsqueeze(1) - negative_embeddings.unsqueeze(0),
            dim=2, p=2
        ).squeeze(0)

        # Select top-k hardest (closest) negatives
        _, hard_indices = torch.topk(distances, min(k, len(distances)), largest=False)

        hard_negative_indices.append(hard_indices)

    return hard_negative_indices
