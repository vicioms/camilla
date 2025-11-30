import numpy as np
import pandas as pd
import torch
from typing import List, Union

class Sequence:
    def __init__(self, arrival_times: np.ndarray, features: np.ndarray):
        self.arrival_times = arrival_times
        self.features = features

    def sample_sequences(self, max_num_sequences : int, 
                        duration_scale : float,
                        return_inter_times : bool,
                        random_seed : Union[int, None] = None):
        t_min = self.arrival_times.min()
        t_max = self.arrival_times.max()
        t_start = np.random.uniform(t_min, t_max, size=max_num_sequences)
        durations = np.random.exponential(scale=duration_scale, size=max_num_sequences)
        t_end = np.minimum(t_start + durations, t_max)
        masks = self.arrival_times[None, :] >= t_start[:, None]
        masks = masks & (self.arrival_times[None, :] <= t_end[:, None])
        valid_masks = masks.sum(axis=1) > 0
        masks = masks[valid_masks, :]
        
        t_start = t_start[valid_masks]
        t_end = t_end[valid_masks]
        subsequence_arrival_times = [ self.arrival_times[masks[index, :]] for index in range(masks.shape[0]) ]
        subsequence_features = [ self.features[masks[index, :], :] for index in range(masks.shape[0]) ]
        subsequence_all_inter_times = [ np.diff(subsequence_arrival_times[index], prepend=t_start[index], append=t_end[index])  for index in range(len(subsequence_arrival_times))]
        if(return_inter_times):
            subsequence_censored_inter_times = np.array([ subsequence_all_inter_times[index][-1]   for index in range(len(subsequence_all_inter_times))])
            subsequence_inter_times = [ subsequence_all_inter_times[index][:-1]  for index in range(len(subsequence_all_inter_times))]
            return subsequence_inter_times, subsequence_censored_inter_times, subsequence_features, t_start, t_end
        else:
            return subsequence_arrival_times, subsequence_features, t_start, t_end
    @staticmethod
    def pack_sequences(sequences: List[np.ndarray]):
        lengths = [ seq.shape[0] for seq in sequences ]
        max_length = max(lengths)
        batch_size = len(sequences)
        feature_dim = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1
        packed_array = np.zeros((batch_size, max_length, feature_dim))
        mask = np.zeros((batch_size, max_length), dtype=bool)
        for i, seq in enumerate(sequences):
            length = seq.shape[0]
            packed_array[i, :length, ...] = seq
        return packed_array, np.array(lengths)