import pickle
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.datasets import ETTh1Dataset
from darts.models.forecasting.tide_model import TiDEModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse, mape

# Load the model
# *************************************************************************
# with open('model_ETTh1.pkl', 'rb') as f:
#     tide = TiDEModel.load('model_ETTh1.pkl')
tide = TiDEModel.load('model_ETTh1.pkl')

series = ETTh1Dataset().load()
# print(series.pd_dataframe())

train, test = series[:-96], series[-96:]

train_scaler = Scaler()
scaled_train = train_scaler.fit_transform(train)

scaled_pred_tide = tide.predict(n=24)
pred_tide = train_scaler.inverse_transform(scaled_pred_tide)

preds_df = pred_tide.pd_dataframe()
test_df = test.pd_dataframe()

print(preds_df)
print(test_df)

tide_mae = mae(test, pred_tide)
tide_mse = mse(test, pred_tide)
tide_mape = mape(test, pred_tide)

print(tide_mae, tide_mse, tide_mape)

# cols_to_plot = ['OT', 'HULL', 'MUFL', 'MULL']
#
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
#
# for i, ax in enumerate(axes.flatten()):
#     col = cols_to_plot[i]
#
#     ax.plot(test_df[col], label='Actual', ls='-', color='blue')
#     ax.plot(preds_df[col], label='TiDE', ls='--', color='green')
#
#     ax.legend(loc='best')
#     ax.set_xlabel('Date')
#     ax.set_title(col)
#
# plt.tight_layout()
# fig.autofmt_xdate()
