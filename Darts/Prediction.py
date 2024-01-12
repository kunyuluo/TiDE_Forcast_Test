import pandas as pd
import numpy as np
import plotly.express as px
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
import pickle

# Load the model
# *************************************************************************
with open('model_TiDE.pkl', 'rb') as f:
    model = pickle.load(f)

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

print('Traning set is ready.')
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
print('Covariates is ready.')

# scale covariates
dynamic_covariates_transformed = SCALER.fit_transform(dynamic_covariates)
# scale data
data_transformed = SCALER.fit_transform(train_darts)
# transform static covariates
data_transformed = TRANSFORMER.fit_transform(data_transformed)

pred = SCALER.inverse_transform(
    model.predict(
        n=FORECAST_HORIZON, series=data_transformed, future_covariates=dynamic_covariates_transformed, num_samples=50))

# let's check one example
pred_df = pred[0].quantile_df(0.5).reset_index().rename(columns={'Weekly_Sales_0.5':'forecast'})
pred_df['Store'] = TRANSFORMER.inverse_transform(pred[0]).static_covariates['Store'][0]
pred_df['Dept'] = TRANSFORMER.inverse_transform(pred[0]).static_covariates['Store'][0]
pred_df = pd.concat([train[(train['Store'] == 1) & (train['Dept'] == 1)], pred_df])
# print(pred_df)

fig = px.line(
    pred_df,
    x=pred_df["Date"],
    y=['Weekly_Sales', 'forecast'],
    hover_data={"Date": "|%B %d, %Y"},
    width=1350,
    height=500,
)

fig.update_layout(
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    yaxis_title="Weekly Sales",
    xaxis_title="Delivery Week",
)

fig.show()
