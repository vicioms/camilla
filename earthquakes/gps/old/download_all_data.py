import numpy as np
import ngl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import haversine_distances
from datetime import datetime, timedelta
import pathlib
import json

station_list = ngl.ngl_process_list(ngl.ngl_24h_2w)
ngl.ngl_retrieve_24h