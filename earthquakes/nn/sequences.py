import numpy as np
import torch

from numpy.typing import NDArray
from typing import List, Union


def process_arrival_times(
    arrival_times: Union[NDArray, torch.Tensor],
    max_lookback_time: float,
    max_neighbors: int,
    dtype: torch.dtype = torch.float32,
):
    """
    Build sparse causal neighborhoods.

    For event i, valid neighbors j satisfy:
        j < i
        arrival_times[i] - arrival_times[j] <= max_lookback_time

    Returns:
        neighbor_idx:  [N, K]
        neighbor_dt:   [N, K]
        neighbor_mask: [N, K]
    """

    if max_lookback_time < 0:
        raise ValueError("max_lookback_time must be non-negative.")

    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive.")

    if isinstance(arrival_times, np.ndarray):
        arrival_times = torch.as_tensor(arrival_times, dtype=dtype)
    else:
        arrival_times = arrival_times.to(dtype=dtype)

    if arrival_times.ndim != 1:
        raise ValueError("arrival_times must have shape [N].")

    N = arrival_times.shape[0]

    if N == 0:
        raise ValueError("arrival_times must contain at least one event.")

    if N > 1 and not torch.all(arrival_times[1:] >= arrival_times[:-1]):
        raise ValueError("arrival_times must be sorted.")

    device = arrival_times.device

    # Earliest allowed neighbor according to physical lookback.
    cutoff_times = arrival_times - max_lookback_time
    left_idx = torch.searchsorted(arrival_times, cutoff_times, right=False)  # [N]

    # Candidate neighbors: i-1, ..., i-K.
    i = torch.arange(N, device=device)[:, None]                              # [N, 1]
    offsets = torch.arange(1, max_neighbors + 1, device=device)[None, :]    # [1, K]
    neighbor_idx = i - offsets                                               # [N, K]

    neighbor_mask = (neighbor_idx >= 0) & (neighbor_idx >= left_idx[:, None])

    # Safe indexing for invalid padded neighbors.
    safe_neighbor_idx = neighbor_idx.clamp_min(0)

    # Edge feature: physical elapsed time.
    neighbor_dt = arrival_times[:, None] - arrival_times[safe_neighbor_idx]
    neighbor_dt = torch.where(neighbor_mask, neighbor_dt, torch.zeros_like(neighbor_dt))

    return {
        "neighbor_idx": safe_neighbor_idx.long(),
        "neighbor_dt": neighbor_dt,
        "neighbor_mask": neighbor_mask,
    }


def pack_sequences(
    features: List[torch.Tensor],
    neighbor_idxs: List[torch.Tensor],
    neighbor_dts: List[torch.Tensor],
    neighbor_masks: List[torch.Tensor],
):
    """
    Pack variable-length event sequences.

    Per sequence:
        features:      [L, F]  (or [L])
        neighbor_idx:  [L, K]
        neighbor_dt:   [L, K]
        neighbor_mask: [L, K]

    Returns:
        features:      [B, Lmax, F]
        neighbor_idx:  [B, Lmax, Kmax]
        neighbor_dt:   [B, Lmax, Kmax]
        neighbor_mask: [B, Lmax, Kmax]
        event_mask:    [B, Lmax]
    """

    B = len(features)

    if B == 0:
        raise ValueError("At least one sequence is required.")

    if not (len(neighbor_idxs) == len(neighbor_dts) == len(neighbor_masks) == B):
        raise ValueError("All input lists must have the same length.")

    # Scalar feature -> [L, 1].
    features = [x[:, None] if x.ndim == 1 else x for x in features]

    if any(x.ndim != 2 for x in features):
        raise ValueError("Each feature tensor must have shape [L, F] or [L].")

    feature_dim = features[0].shape[1]

    if any(x.shape[1] != feature_dim for x in features):
        raise ValueError("All sequences must have the same feature dimension.")

    device = features[0].device

    tensors = features + neighbor_idxs + neighbor_dts + neighbor_masks
    if any(x.device != device for x in tensors):
        raise ValueError("All tensors must be on the same device.")

    max_event_length = max(x.shape[0] for x in features)
    max_neighbors = max(x.shape[1] for x in neighbor_idxs)

    features_packed = torch.zeros(
        (B, max_event_length, feature_dim),
        dtype=features[0].dtype,
        device=device,
    )

    neighbor_idxs_packed = torch.zeros(
        (B, max_event_length, max_neighbors),
        dtype=torch.long,
        device=device,
    )

    neighbor_dts_packed = torch.zeros(
        (B, max_event_length, max_neighbors),
        dtype=neighbor_dts[0].dtype,
        device=device,
    )

    neighbor_masks_packed = torch.zeros(
        (B, max_event_length, max_neighbors),
        dtype=torch.bool,
        device=device,
    )

    event_mask = torch.zeros(
        (B, max_event_length),
        dtype=torch.bool,
        device=device,
    )

    for b, (feat, idx, dt, mask) in enumerate(
        zip(features, neighbor_idxs, neighbor_dts, neighbor_masks)
    ):
        L, K = idx.shape

        if feat.shape[0] != L:
            raise ValueError(
                f"Sequence {b}: features has {feat.shape[0]} events "
                f"but sparse graph has {L}."
            )

        if idx.shape != dt.shape or idx.shape != mask.shape:
            raise ValueError(
                f"Sequence {b}: idx, dt, and mask must have identical shapes."
            )

        features_packed[b, :L] = feat
        neighbor_idxs_packed[b, :L, :K] = idx
        neighbor_dts_packed[b, :L, :K] = dt
        neighbor_masks_packed[b, :L, :K] = mask
        event_mask[b, :L] = True

    return {
        "features": features_packed,
        "neighbor_idx": neighbor_idxs_packed,
        "neighbor_dt": neighbor_dts_packed,
        "neighbor_mask": neighbor_masks_packed,
        "event_mask": event_mask,
    }


def gather_neighbors(h: torch.Tensor, neighbor_idx: torch.Tensor):
    """
    h:            [B, N, D]
    neighbor_idx: [B, N, K]

    returns:
        [B, N, K, D]
    """

    B, N, _ = h.shape

    if neighbor_idx.shape[:2] != (B, N):
        raise ValueError("neighbor_idx must match h in B and N dimensions.")

    batch_idx = torch.arange(B, device=h.device)[:, None, None]

    return h[batch_idx, neighbor_idx]