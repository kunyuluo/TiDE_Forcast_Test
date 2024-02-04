import matplotlib.pyplot as plt
from darts.datasets import AirPassengersDataset, ElectricityDataset
from darts.models import TiDEModel
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries, concatenate
from darts.metrics import mape, smape, mae
from darts.utils.timeseries_generation import datetime_attribute_timeseries

# Load the dataset
# **************************************************************************************
series_air = AirPassengersDataset().load()
# print(series_air.pd_dataframe())

# plt.plot(series_air.pd_dataframe())
# plt.show()

# Standardization
# **************************************************************************************
scaler_air = Scaler()
series_air_scaled = scaler_air.fit_transform(series_air)

# Build dynamic covariates (year and month)
# **************************************************************************************
air_year = datetime_attribute_timeseries(series_air_scaled, attribute="year")
air_month = datetime_attribute_timeseries(series_air_scaled, attribute="month")

# stack year and month to obtain series of 2 dimensions (year and month):
air_covariates = air_year.stack(air_month)
# print(air_covariates.pd_dataframe())

# Split data into train and validation
# **************************************************************************************
train_air, val_air = series_air_scaled[:-48], series_air_scaled[-48:]
train_air_covariates, val_air_covariates = air_covariates[:-48], air_covariates[-48:]

# scale them between 0 and 1:
scaler_covariates = Scaler()
train_air_covariates = scaler_covariates.fit_transform(train_air_covariates)
val_air_covariates = scaler_covariates.transform(val_air_covariates)

# air_covariates = concatenate([train_air_covariates, val_air_covariates])
print(train_air.pd_dataframe())
# print(train_air_covariates.pd_dataframe())
print(val_air.pd_dataframe())

# Build TiDE model
# **************************************************************************************
# model = TiDEModel(
#     input_chunk_length=24,
#     output_chunk_length=24,
#     n_epochs=200,
#     random_state=0,)
#
# model.fit(series=train_air,
#           past_covariates=train_air_covariates,
#           val_series=val_air,
#           val_past_covariates=val_air_covariates)
#
# pred = model.predict(n=24, series=train_air, past_covariates=train_air_covariates)
# print("MAPE = {:.2f}%".format(mape(val_air, pred)))
#
# plt.plot(series_air_scaled.pd_dataframe())
# plt.plot(pred.pd_dataframe())
# plt.show()

