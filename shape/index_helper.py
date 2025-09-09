import numpy as np

def labels_get_indices(labels : np.ndarray, exclude_background : bool, add_unravelled_indices : bool) -> dict:
    unique_labels, unique_inverse, unique_counts = np.unique(labels.flatten(), return_inverse=True, return_counts=True)
    all_indices = np.argsort(unique_inverse)
    if(exclude_background):
        if(unique_labels[0] == 0):
            num_zero_labels = unique_counts[0]
            unique_labels = unique_labels[1:]
            unique_counts = unique_counts[1:]
            all_indices = all_indices[num_zero_labels:]
    indices_dict = {}
    ref = 0
    for label, count in zip(unique_labels, unique_counts):
        indices_flat = all_indices[ref:ref+count]
        indices_dict[label] = {'indices_flattened': indices_flat, 'count' : count}
        if(add_unravelled_indices):
            unravelled_indices = np.unravel_index(indices_flat, labels.shape)
            indices_dict[label]['indices'] = unravelled_indices
        ref += count
    return indices_dict