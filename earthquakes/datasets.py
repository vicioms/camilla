import torch
from torch.utils.data import Dataset, IterableDataset
import h5py
import numpy as np
import pandas as pd

from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

from pathlib import Path
from datetime import datetime
from typing import Union, List, Optional, Tuple, Any


class OnlineWaveformDataset(Dataset):
    def __init__(
        self,
        client: Any,
        channels: Union[str, List[str]] = ("HH?", "BH?"),
    ):
        if isinstance(client, str):
            self.client = Client(client)
        else:
            self.client = client

        self.channels = self._format_channels(channels)

    @staticmethod
    def _format_channels(
        channels: Union[str, List[str]],
    ) -> str:
        if isinstance(channels, str):
            return channels

        return ",".join(channels)

    def get_client(self):
        return self.client

    def get_stations(
        self,
        min_longitude: float,
        max_longitude: float,
        min_latitude: float,
        max_latitude: float,
        start_time: UTCDateTime,
        end_time: UTCDateTime,
        channels: Optional[Union[str, List[str]]] = None,
    ):
        channels = (
            self.channels
            if channels is None
            else self._format_channels(channels)
        )

        inventory = self.client.get_stations(
            minlongitude=min_longitude,
            maxlongitude=max_longitude,
            minlatitude=min_latitude,
            maxlatitude=max_latitude,
            channel=channels,
            starttime=start_time,
            endtime=end_time,
            level="channel",
        )

        metadata = {}

        for network in inventory:
            for station in network:
                key = (network.code, station.code)

                metadata[key] = {
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "elevation": station.elevation,
                    "site_name": getattr(station.site, "name", None),
                    "channels": [
                        {
                            "code": channel.code,
                            "location": channel.location_code,
                            "sample_rate": channel.sample_rate,
                        }
                        for channel in station.channels
                    ],
                }

        return metadata

    @staticmethod
    def select_channels(
        station_metadata,
        priorities=("HH", "BH"),
    ):
        """
        Select a single complete three-component channel set.

        Preference
        ----------
        HHZ, HHN, HHE
        HHZ, HH1, HH2
        BHZ, BHN, BHE
        BHZ, BH1, BH2

        All components must have:
            - the same location code
            - the same sample rate

        Returns
        -------
        dict or None
        """

        channels = station_metadata["channels"]

        for prefix in priorities:
            locations = sorted({
                channel["location"]
                for channel in channels
                if channel["code"].startswith(prefix)
            })

            for location in locations:
                available = {
                    channel["code"]: channel
                    for channel in channels
                    if (
                        channel["location"] == location
                        and channel["code"].startswith(prefix)
                    )
                }

                component_sets = (
                    ("Z", "N", "E"),
                    ("Z", "1", "2"),
                )

                for components in component_sets:
                    desired = [
                        f"{prefix}{component}"
                        for component in components
                    ]

                    if not all(
                        code in available
                        for code in desired
                    ):
                        continue

                    sample_rates = [
                        available[code]["sample_rate"]
                        for code in desired
                    ]

                    if not np.allclose(
                        sample_rates,
                        sample_rates[0],
                    ):
                        continue

                    return {
                        "location": location,
                        "channels": desired,
                        "components": components,
                        "sample_rate": sample_rates[0],
                    }

        return None

    def get_waveforms_in_box(
        self,
        box: Tuple[float, float, float, float],
        start_time: UTCDateTime,
        end_time: UTCDateTime,
        channels: Optional[Union[str, List[str]]] = None,
    ):
        """
        Retrieve preferred three-component waveforms from all
        suitable stations in a geographic box.

        Parameters
        ----------
        box
            (
                min_longitude,
                max_longitude,
                min_latitude,
                max_latitude,
            )

        start_time
            Start of waveform window.

        end_time
            End of waveform window.

        channels
            Optional channel query overriding self.channels.

        Returns
        -------
        stream : obspy.Stream
            Combined waveform stream.

        metadata : dict
            Metadata for stations for which a valid three-component
            channel set was identified.
        """

        (
            min_longitude,
            max_longitude,
            min_latitude,
            max_latitude,
        ) = box

        metadata = self.get_stations(
            min_longitude=min_longitude,
            max_longitude=max_longitude,
            min_latitude=min_latitude,
            max_latitude=max_latitude,
            start_time=start_time,
            end_time=end_time,
            channels=channels,
        )

        bulk = []
        selected_metadata = {}

        for (network, station), station_metadata in metadata.items():
            selected = self.select_channels(
                station_metadata
            )

            if selected is None:
                continue

            location = selected["location"]

            selected_metadata[(network, station)] = {
                **station_metadata,
                "selected": selected,
            }

            for channel in selected["channels"]:
                bulk.append((
                    network,
                    station,
                    location,
                    channel,
                    start_time,
                    end_time,
                ))

        if len(bulk) == 0:
            return Stream(), selected_metadata

        try:
            stream = self.client.get_waveforms_bulk(
                bulk
            )

        except FDSNNoDataException:
            stream = Stream()

        return stream, selected_metadata

    #def set_catalog(self, longitudes, latitudes, times, magnitudes):
        

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


