import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd

class SteadDataset(Dataset):
    def __init__(self, chunk_files, channel_first):
        self.files = []
        self.event_lists = []
        self.stopping_indices = None
        for chunk in chunk_files:
            file = h5py.File(chunk, 'r')
            metadata = pd.read_csv(chunk.replace('hdf5', 'csv'))
            ev_list = metadata['trace_name'].astype('str').to_list()
            self.files.append(file)
            self.event_lists.append(ev_list)
            if self.stopping_indices is not None:
                self.stopping_indices.append(self.stopping_indices[-1] + len(ev_list))
            else:
                self.stopping_indices = [len(ev_list)]
        self.stopping_indices = np.array(self.stopping_indices)
        self.channel_first = channel_first
    def __len__(self):
        return sum([len(ev_list) for ev_list in self.event_lists])
    

    def __getitem__(self, idx):
        # find which chunk
        chunk_idx = 0
        while idx >= self.stopping_indices[chunk_idx]:
            chunk_idx += 1
        relative_idx = idx - self.stopping_indices[chunk_idx - 1] if chunk_idx > 0 else idx
        event_name = self.event_lists[chunk_idx][relative_idx]
        file = self.files[chunk_idx].get('data/' + event_name)
        trace = np.array(file)
        p_arrival = file.attrs['p_arrival_sample']
        s_arrival = file.attrs['s_arrival_sample']
        coda_end = file.attrs['coda_end_sample']
        if(p_arrival == ''):
            p_arrival = np.nan
        if(s_arrival == ''):
            s_arrival = np.nan
        if(coda_end == ''):
            coda_end = np.nan
        if self.channel_first:
            trace = trace.transpose(1, 0)
        return trace, p_arrival.item(), s_arrival.item(), coda_end.item(), event_name