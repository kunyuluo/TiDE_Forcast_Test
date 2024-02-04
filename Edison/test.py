import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller, Mapper, InvertibleMapper
from darts.dataprocessing import Pipeline
from darts import concatenate
from darts.models import TiDEModel
from darts.metrics import mape
from Helper import DefaultValueFiller
import pickle

# data = pd.read_csv('../dummy_data.csv', low_memory=True)
#
# data['data_time'] = pd.to_datetime(data['data_time'])
# data.set_index('data_time', inplace=True)
# data_new = data.resample('15min').mean()

# target = data[['cp_power', 'oat']]
# target = TimeSeries.from_series(data['cp_power'])
# target = TimeSeries.from_dataframe(target)
# target.with_static_covariates()
# print(target)

# arbitrary continuous and categorical static covariates (single row)
# static_covs_single = pd.DataFrame(data={"cont": [0], "cat": ["a"]})
# print(static_covs_single)

# multivariate static covariates (multiple components). note that the number of rows matches the number of components of `series`
# static_covs_multi = pd.DataFrame(data={"cont": [0, 2, 1], "cat": ["a", "c", "b"]})
# print(static_covs_multi)

# series_single = target.with_static_covariates(static_covs_single)
# print(series_single)
# series_multi = target.with_static_covariates(static_covs_multi)
# print(series_multi.static_covariates)

# values = np.arange(start=0, stop=1, step=0.1)
# values[5:8] = np.nan
# series = TimeSeries.from_values(values)
# print(series)
# transformer = MissingValuesFiller()
# series_new = transformer.transform(series)
# print(series_new)

data = pd.read_csv('infinity_wallcontrol_power_data.csv', low_memory=False)
features_name = ['vfd_ac_line_power', 'oat', 'zone_rt']

filler = DefaultValueFiller(data, features_name, 1)
# feature_data = filler.get_feature_data()
# print(feature_data)
new_df = filler.fill_missing_value()
print(new_df)
new_df.to_csv('new_data_0201.csv', index=False)
