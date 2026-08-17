import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from pathlib import Path
from typing import List, Tuple, Optional, Union
from datetime import datetime

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

def get_isc_catalog(start_time, end_time, bounding_box : Tuple[float, float, float, float], min_magnitude : Union[float, None] = None, max_magnitude : Union[float, None] = None, include_all_origins=False, include_all_magnitudes=False, include_arrivals=False):
    minlatitude, maxlatitude, minlongitude, maxlongitude  = bounding_box

    isc_client = Client("ISC")
    start_time = UTCDateTime(start_time)
    end_time = UTCDateTime(end_time)
    args = {}
    if min_magnitude is not None:
        args['minmagnitude'] = min_magnitude
    if max_magnitude is not None:
        args['maxmagnitude'] = max_magnitude
    cat = isc_client.get_events(
        starttime=start_time,
        endtime=end_time,
        minlatitude=minlatitude,
        maxlatitude=maxlatitude,
        minlongitude=minlongitude,
        maxlongitude=maxlongitude,
        includeallorigins=include_all_origins,
        includeallmagnitudes=include_all_magnitudes,
        includearrivals=include_arrivals,
        **args
    )
    return cat


isc_catalog_family =  {
    "mb":     "mb",
    "mB":     "mB",
    "mbtmp":  "mb",

    "ML":     "ML",
    "Ml":     "ML",
    "ml":     "ML",
    "MLh":    "ML",

    "MS":     "Ms",
    "Ms":     "Ms",
    "Ms7":    "Ms",
    "Ms_20":  "Ms",

    "Mw":     "Mw",
    "MW":     "Mw",
    "Mwb":    "Mw",
    "Mww":    "Mw",

    "Mwp":    "Mwp",
    "MwMwp":  "Mw_proxy",
    "Mw(mB)": "Mw_proxy",

    "M":      "unspecified",
}

def catalog_to_dataframe(cat, include_all_magnitudes=False):
    df = []
    if include_all_magnitudes:
        magnitudes_df = []
    for event in cat:
        origin = (event.preferred_origin()or (event.origins[0] if event.origins else None))
        event_id = str(event.resource_id) if event.resource_id else None
        if not event_id:
            continue
        row = {"event_id" : event_id,
                "time" : origin.time.datetime if origin else None, 
                "latitude" : origin.latitude if origin else np.nan, 
                "longitude" : origin.longitude if origin else np.nan, 
                "depth" : origin.depth / 1000 if origin and origin.depth is not None else np.nan,
                "magnitude" : event.preferred_magnitude().mag if event.preferred_magnitude() else np.nan,
                "magnitude_type" : event.preferred_magnitude().magnitude_type if event.preferred_magnitude() else None}
        row["magnitude_family"] = isc_catalog_family.get(row["magnitude_type"], "unspecified")
        df.append(row)
        if include_all_magnitudes and event.magnitudes:
            for mag in event.magnitudes:
                magnitudes_df.append({"event_id" : event_id, "magnitude" : mag.mag, "magnitude_type" : mag.magnitude_type, "magnitude_family" : isc_catalog_family.get(mag.magnitude_type, "unspecified")})


    df = pd.DataFrame(df)
    if include_all_magnitudes:
        if len(magnitudes_df) == 0:
            magnitudes_df = pd.DataFrame(columns=["event_id", "magnitude", "magnitude_type", "magnitude_family"])
        else:
            magnitudes_df = pd.DataFrame(magnitudes_df)
        return df, magnitudes_df
    else:
        return df


