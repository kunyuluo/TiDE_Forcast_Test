import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TiDEModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import StaticCovariatesTransformer
import pickle

# Global Variables
TIME_COL = "Date"
TARGET = "Weekly_Sales"
STATIC_COV = ["Store", "Dept"]
FREQ = "W-FRI"
SCALER = Scaler()
TRANSFORMER = StaticCovariatesTransformer()

# read train and test datasets and transform train dataset
train = pd.read_csv('train.csv')
train["Date"] = pd.to_datetime(train["Date"])
train[TARGET] = np.where(train[TARGET] < 0, 0, train[TARGET])
train_darts = TimeSeries.from_group_dataframe(
    df=train,
    group_cols=STATIC_COV,
    time_col=TIME_COL,
    value_cols=TARGET,
    static_cols=STATIC_COV,
    freq=FREQ,
    fill_missing_dates=True,
    fillna_value=0)

# read test dataset and determine Forecast Horizon
test = pd.read_csv('test.csv')
test["Date"] = pd.to_datetime(test["Date"])
FORECAST_HORIZON = len(test['Date'].unique())

# we get the holiday data that we have in both train and test dataset
holidays_df = pd.concat([train[["Date", "IsHoliday"]], test[["Date", "IsHoliday"]]]).drop_duplicates()
# convert bool to numeric
holidays_df["IsHoliday"] = holidays_df["IsHoliday"] * 1
# create dynamic covariates for each serie in the training darts
dynamic_covariates = []
for serie in train_darts:
    # add the month and week as a covariate
    covariate = datetime_attribute_timeseries(
        serie,
        attribute="month",
        one_hot=True,
        cyclic=False,
        add_length=FORECAST_HORIZON,
    )
    covariate = covariate.stack(
        datetime_attribute_timeseries(
            serie,
            attribute="week",
            one_hot=True,
            cyclic=False,
            add_length=FORECAST_HORIZON,
        )
    )

# create holidays with dates for training and test
holidays_serie = pd.merge(pd.DataFrame(covariate.time_index).rename(columns={'time': 'Date'}), holidays_df,
                              on='Date', how='left')
covariate = covariate.stack(
        TimeSeries.from_dataframe(holidays_serie, time_col="Date", value_cols="IsHoliday", freq=FREQ)
    )
dynamic_covariates.append(covariate)

# scale covariates
dynamic_covariates_transformed = SCALER.fit_transform(dynamic_covariates)
# scale data
data_transformed = SCALER.fit_transform(train_darts)
# transform static covariates
data_transformed = TRANSFORMER.fit_transform(data_transformed)

TiDE_params = {
    "input_chunk_length": 4, # number of weeks to lookback
    "output_chunk_length": FORECAST_HORIZON,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "decoder_output_dim": 1,
    "hidden_size": 15,
    "temporal_width_past": 4,
    "temporal_width_future": 4,
    "temporal_decoder_hidden": 26,
    "dropout": 0.1,
    "batch_size": 16,
    "n_epochs": 5,
    "likelihood": QuantileRegression(quantiles=[0.25, 0.5, 0.75]),
    "random_state": 42,
    "use_static_covariates": True,
    "optimizer_kwargs": {"lr": 1e-3},
    "use_reversible_instance_norm": False,
}

model = TiDEModel(**TiDE_params)
model.fit(data_transformed, future_covariates=dynamic_covariates_transformed, verbose=False)
# pred = SCALER.inverse_transform(model.predict(n=FORECAST_HORIZON, series=data_transformed, future_covariates=dynamic_covariates_transformed, num_samples=50))

# Save models
# *************************************************************************
with open('model_TiDE.pkl', 'wb') as f:
    pickle.dump(model, f)
