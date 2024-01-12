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
from TiDE_TF.Helper import DataProcessor
import pickle

data = pd.read_csv('infinity_wallcontrol_power_data.csv', low_memory=True)

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# fig = px.scatter(data, x=data['timestamp'], y=data['vfd_ac_line_power'])
# fig.show()

data_new = data[['timestamp', 'oat', 'zone_rt', 'vfd_ac_line_power']]
data_new = data_new.set_index('timestamp')
data_new.index = pd.to_datetime(data_new.index)
data_new = data_new.resample('H').mean()
data_new_series = TimeSeries.from_dataframe(data_new, fill_missing_dates=True, fillna_value=True, freq='60min')

# df.to_csv('data_new.csv')

transformer = MissingValuesFiller()
data_new_series = transformer.transform(data_new_series)
df = data_new_series.pd_dataframe()
print(df.iloc[3200, :])
DataProcessor.plot_variable_no_time(df, 'vfd_ac_line_power')

train, val = data_new_series.split_before(pd.Timestamp("20230308"))
valex, train = train.split_after(pd.Timestamp("20230101"))

# print(len(train))

validation = val
target = train['vfd_ac_line_power']
past_cov = concatenate([train['zone_rt'], train['oat']], axis=1)
# print(type(validation['oat'][0]))
# print(validation.pd_dataframe().values)
val_test = validation.pd_dataframe()
val_test['oat'] = [40]*24
val_test = TimeSeries.from_dataframe(val_test)
# power = validation['vfd_ac_line_power'].pd_dataframe().values.flatten()

# ts = TimeSeries.from_values(temps)
# print(ts)
# print(val_test)
# plt.plot(power)
# plt.show()

# tide_model = TiDEModel(
#     input_chunk_length=18,
#     output_chunk_length=24,
#     random_state=42,
#     n_epochs=200)
#
# tide_model.fit(target, past_covariates=past_cov)
# tide_pred = tide_model.predict(24, series=validation)
# error = mape(validation['vfd_ac_line_power'], tide_pred['vfd_ac_line_power'])
# print(tide_pred['vfd_ac_line_power'].pd_dataframe().values.flatten())
# print(f'MAPE:{error}')

# Save models
# *************************************************************************
# with open('model.pkl', 'wb') as f:
#     pickle.dump(tide_model, f)
