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
# train = train.drop_duplicates(subset=["Date", "Store", "Dept"])
# print(len(train['Dept'].unique()))

train_darts = TimeSeries.from_group_dataframe(
    df=train,
    group_cols=STATIC_COV,
    time_col=TIME_COL,
    value_cols=TARGET,
    static_cols=STATIC_COV,
    freq=FREQ,
    fill_missing_dates=True,
    fillna_value=0)

# print(len(train_darts))
# print(train_darts[0].pd_dataframe())
# print(train_darts[1].pd_dataframe())

# read test dataset and determine Forecast Horizon
test = pd.read_csv('test.csv')
test["Date"] = pd.to_datetime(test["Date"])
FORECAST_HORIZON = len(test['Date'].unique())

# Create Dynamic Covarites
holidays_df = pd.concat([train[["Date", "IsHoliday"]], test[["Date", "IsHoliday"]]]).drop_duplicates()
holidays_df["IsHoliday"] = holidays_df["IsHoliday"] * 1
# print(holidays_df.head(40))

dynamic_covariates = []
# for serie in train_darts:
# add the month and week as a covariate
covariate = datetime_attribute_timeseries(
    train_darts[0], attribute='month', one_hot=True, cyclic=False, add_length=FORECAST_HORIZON)
covariate = covariate.stack(datetime_attribute_timeseries(
    train_darts[0], attribute='week', one_hot=True, cyclic=False, add_length=FORECAST_HORIZON))
holidays_serie = pd.merge(pd.DataFrame(covariate.time_index).rename(columns={'time': 'Date'}), holidays_df,
                          on='Date', how='left')
covariate = covariate.stack(
                TimeSeries.from_dataframe(holidays_serie, time_col="Date", value_cols="IsHoliday", freq=FREQ))
# print(covariate.pd_dataframe())
dynamic_covariates.append(covariate)
# print(holidays_serie)

# scale covariates
dynamic_covariates_transformed = SCALER.fit_transform(dynamic_covariates)
print(dynamic_covariates[0].pd_dataframe())
# print(dynamic_covariates_transformed[0].pd_dataframe())

# scale data
data_transformed = SCALER.fit_transform(train_darts)
# print(train_darts[0].pd_dataframe())
# print(data_transformed[0].pd_dataframe())

# transform static covariates
data_transformed = TRANSFORMER.fit_transform(data_transformed)
print(data_transformed[0].pd_dataframe())
