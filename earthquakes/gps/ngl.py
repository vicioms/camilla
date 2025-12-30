import astropy
import astropy.time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import urllib
import zipfile
import gzip
from io import StringIO
import datetime
import os
import pathlib

ngl_lists = {
    "full" : "https://geodesy.unr.edu/NGLStationPages/llh.out",
"24h_final" : "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt",
#"24h_rapid" : "https://geodesy.unr.edu/NGLStationPages/DataHoldingsRapid24hr.txt",
"5min_final" : "https://geodesy.unr.edu/NGLStationPages/DataHoldingsRapid5min.txt",
"5min_rapid" : "https://geodesy.unr.edu/NGLStationPages/DataHoldingsUltra5min.txt"
}

base_data_url_IGS20 = "https://geodesy.unr.edu/gps_timeseries/IGS20/"

formats_IGS20_24h_final = {'tenv' : 'tenv/%s.tenv',
                           'tenv3' : 'tenv3/IGS20/%s.tenv3',
                           'xyz' : 'txyz/%s.txyz2'}
#formats_IGS20_24h_rapid = {'tenv3' : 'tenv3/IGS20/%s.tenv3',}
formats_IGS20_5min_final = {'kenv' : lambda station, year : f'kenv/{station}/{station}.{year}.kenv.zip' ,}

tenv_column_names = ['station', 'date', 'decimal_year', 'MJD',
                     'gps_week', 'gps_week_day',
                     'unknown',
                     'delta_e_m', 'delta_n_m', 'delta_v_m',
                     'antenna_height_m',
                     'sigma_e_m', 'sigma_n_m', 'sigma_v_m',
                     'correlation_en', 'correlation_ev', 'correlation_nv']

xyz_column_names = ['station_name', 'date', 'decimal_year',
                    'x_m', 'y_m', 'z_m',
                    'x_sigma_m', 'y_sigma_m', 'z_sigma_m',
                    'xy_correlation_coefficient',
                    'yz_correlation_coefficient',
                    'xz_correlation_coefficient',
                    'antenna_height_m']


def _ngl_str_to_datetime(s, limit_for_2000):
    str_to_month = {'JAN':1, 'FEB':2, 'MAR':3, 'APR' : 4, 'MAY':5, 'JUN':6, 'JUL' : 7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
    d = int(s[5:])
    m = str_to_month[s[2:5]]
    y = s[:2]
    if(int(y) > limit_for_2000):
        y = int("19" + y)
    else:
        y = int("20" + y)
    return datetime.datetime(y, m, d)

def fetch_station_24h_final():
    station_list = pd.read_csv(ngl_lists['24h_final'], sep=r"\s+", on_bad_lines='skip', parse_dates=['Dtbeg', 'Dtend'])
    station_list.rename(columns={'Sta' : 'station', 'Lat(deg)' : 'lat', 'Long(deg)' : 'lon', 'Hgt(m)' : 'height', 'X(m)' : 'x', 'Y(m)' : 'y', 'Z(m)' : 'z', 'Dtbeg' : 'begin', 'Dtend' : 'end'   }, inplace=True)
    station_list['lon'] = (station_list['lon'] + 180)%360-180 # ensures values in [-180, 180]
    return station_list
def fetch_IGS20_24h_final(station, data_format = "tenv3",rootpath = None, overwrite = False):
    
    if data_format not in formats_IGS20_24h_final.keys():
        raise ValueError(f"data_format must be one of {list(formats_IGS20_24h_final.keys())}")
    if rootpath is not None:
        save_path = pathlib.Path(rootpath)
        if( save_path.exists() and save_path.is_dir()):
            save_path = save_path.joinpath(f"{station}.csv")
            if(save_path.exists() and not overwrite):
                return pd.read_csv(save_path, sep=",", parse_dates=['date'])
    else:
        save_path = None
    url = base_data_url_IGS20 + formats_IGS20_24h_final[data_format] % station
    if(data_format == "tenv3"):
        raw_df = pd.read_csv(url, sep=r"\s+", comment='#')
    else:
        raw_df = pd.read_csv(url, sep=r"\s+", comment='#', header=None)
        if(data_format == "xyz"):
            raw_df.columns = xyz_column_names
        elif(data_format == "tenv"):
            raw_df.columns = tenv_column_names
            num_years_after_2000 =  datetime.datetime.today().year - 2000
            raw_df['date'] = raw_df['date'].apply(lambda x: _ngl_str_to_datetime(x,num_years_after_2000))
    if(save_path is not None):
        raw_df.to_csv(save_path.absolute(), sep=",", index=False)
    return raw_df