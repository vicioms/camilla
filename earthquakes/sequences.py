import numpy as np
import pandas as pd
import torch
from typing import List, Union

def extract_subsequences(arrival_times: np.ndarray, features: np.ndarray, max_num_sequences : int, duration_scale : float, random_seed : Union[int, None] = None):
    if( random_seed is not None ):
        np.random.seed(random_seed)
    t_min = arrival_times.min()
    t_max = arrival_times.max()
    t_start = np.random.uniform(t_min, t_max, size=max_num_sequences)
    durations = np.random.exponential(scale=duration_scale, size=max_num_sequences)
    t_end = np.minimum(t_start + durations, t_max)
    masks = arrival_times[None, :] >= t_start[:, None]
    masks = masks & (arrival_times[None, :] <= t_end[:, None])
    valid_masks = masks.sum(axis=1) > 0
    masks = masks[valid_masks, :]
    t_start = t_start[valid_masks]
    t_end = t_end[valid_masks]
    subsequence_arrival_times = [ arrival_times[masks[index, :]] for index in range(masks.shape[0]) ]
    subsequence_features = [ features[masks[index, :], :] for index in range(masks.shape[0]) ]
    return subsequence_arrival_times, subsequence_features, t_start, t_end
def pack_sequences(feature_list : List[np.ndarray], batch_first : bool, valid_mask_is_true : bool):
    num_of_events = np.array([len(features) for features in feature_list])
    max_num_events = num_of_events.max()
    num_sequences = len(feature_list)
    # we first construct batch first version
    packed_features = np.zeros((num_sequences, max_num_events, feature_list[0].shape[1]), dtype=np.float32)
    masks = np.full((num_sequences, max_num_events), fill_value=not valid_mask_is_true, dtype=bool)
    for index in range(num_sequences):
        n_events = num_of_events[index]
        packed_features[index, 0:n_events, :] = feature_list[index]
        masks[index, 0:n_events] = valid_mask_is_true
    if(not batch_first):
        packed_features = np.transpose(packed_features, (1,0,2))
    return packed_features, masks

def arrival_to_inter_times(arrival_times : np.ndarray, t_start : float, t_end : float):
    inter_times = np.diff(arrival_times, prepend=t_start, append=t_end)
    return inter_times