import numpy as np
import ngl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import haversine_distances
from datetime import datetime, timedelta
import pathlib
import json

earthquake_list = pd.read_csv('scardec_M7_full.txt', sep=' ', header=None, names=['mw','year', 'month', 'day', 'hour', 'minute', 'second', 'lat', 'lon' ] + ['dummy%i' % i for i in range(7)])
earthquake_list['datetime'] = pd.to_datetime(earthquake_list[['year', 'month', 'day', 'hour', 'minute', 'second']])
station_list = ngl.ngl_process_list(ngl.ngl_5min_24h)
stations_to_quakes_distances = haversine_distances(np.radians(station_list[['lat', 'lon']]), np.radians(earthquake_list[['lat', 'lon']]))
cutoff_radius_fraction =  400/6371

list_of_quake_dicts = []

for quake_idx, quake in earthquake_list.iterrows():
    quake_dict = {}
    stations_subset = station_list[stations_to_quakes_distances[:, quake_idx] <= cutoff_radius_fraction]
    year = quake['year']
    print(quake['lat'], quake['lon'])

    quake_dict['datetime'] = str(quake['datetime'].to_pydatetime())
    quake_dict['lat'] = quake['lat']
    quake_dict['lon'] = quake['lon']
    quake_dict['mw'] = quake['mw']
    quake_dict['stations'] = []

    for _, station in stations_subset.iterrows():
        if(pathlib.Path("stations/%s_%s.csv" % (station['name'], year)).exists()):
            quake_dict['stations'].append("%s_%s" % (station['name'], year))
            continue
        station_dataframes = ngl.ngl_rapid_5min(station['name'], year)
        if(station_dataframes is None):
            continue
        print(station['name'], year)
        station_df = ngl.ngl_rapid_5min_dataframes_merge_postprocess(station_dataframes, year)
        station_df.to_csv("stations/%s_%s.csv" % (station['name'], year), sep=" ", index=False)
        quake_dict['stations'].append("%s_%s" % (station['name'], year))

    list_of_quake_dicts.append(quake_dict.copy())


with open("quake_list.json", "w") as file:
    json.dump(list_of_quake_dicts, file)
