import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TiDEModel
from darts.dataprocessing.transformers import Scaler
from darts import concatenate
from darts.metrics import mape
import optuna


def plot_results(test, preds, title_suffix=None, xlabel='Power Prediction', ylim=None):
    """
    Plots training data in blue, actual values in red, and predictions in green, over time.
    """
    fig, ax = plt.subplots(figsize=(18, 6))
    # x = df.Close[-498:].index
    # if test.shape[1] > 1:
    #     test = test[:, 0]

    plot_test = test[0:]
    plot_preds = preds[0:]

    # x = df[-(plot_test.shape[0] * plot_test.shape[1]):].index
    # plot_test = plot_test.reshape((plot_test.shape[0] * plot_test.shape[1], 1))
    plot_preds = plot_preds.reshape((plot_preds.shape[0] * plot_preds.shape[1], 1))

    ax.plot(plot_test, label='actual')
    ax.plot(plot_preds, label='preds')

    if title_suffix is None:
        ax.set_title('Predictions vs. Actual')
    else:
        ax.set_title(f'Predictions vs. Actual, {title_suffix}')

    ax.set_xlabel('Date')
    ax.set_ylabel(xlabel)
    ax.legend()
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.show()


file_path = '../simple_data.csv'
data = pd.read_csv(file_path, low_memory=True)

data['data_time'] = pd.to_datetime(data['data_time'])
# data['data_time'] = data['data_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
data.set_index('data_time', inplace=True)
data_new = data.resample('1min').mean()
data_series = TimeSeries.from_dataframe(data_new, fill_missing_dates=True, freq='1min')
# print(data_series.pd_dataframe())

train, val = data_series.split_before(0.8)
# print(train.pd_dataframe())

# Standardization
# *************************************************************************
scaler = Scaler()  # default uses sklearn's MinMaxScaler
train = scaler.fit_transform(train)
val = scaler.transform(val)
# print(val.pd_dataframe()['target'].head(10))

target = train['target']
past_cov = concatenate([train['cov_1'], train['cov_2']], axis=1)
val_target = val['target']
val_past_cov = concatenate([val['cov_1'], val['cov_2']], axis=1)

VAL_LEN = 10


# Define objectuve function.
def objective(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 12, 36)
    # out_len = trial.suggest_int("out_len", 1, 6)
    use_rin = trial.suggest_categorical("use_rin", [True, False])

    tide_model = TiDEModel(
        input_chunk_length=in_len,
        output_chunk_length=VAL_LEN,
        random_state=42,
        n_epochs=10,
        use_reversible_instance_norm=use_rin,)
        # force_reset=True,
        # save_checkpoints=True)

    tide_model.fit(
        target, past_covariates=past_cov, val_series=val_target, val_past_covariates=val_past_cov)

    # tide_model = TiDEModel.load_from_checkpoint('tide_model')

    # Evaluate how good it is on the validation set, using MAPE
    preds = tide_model.predict(series=target, past_covariates=past_cov, n=VAL_LEN)
    error = mape(val['target'], preds['target'])

    return error if error != np.nan else float("inf")

# past_cov_pred = concatenate([past_cov, val_past_cov], axis=0)
#
# n = 5
# tide_pred = tide_model.predict(n, series=target, past_covariates=past_cov_pred)
# print(tide_pred['target'].pd_dataframe())
#
# error = mape(val['target'], tide_pred['target'])
# print(f'MAPE:{error}')

# plot_results(val.pd_dataframe()['target'].head(n).values, tide_pred['target'].pd_dataframe().values)


# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# optimize hyperparameters by minimizing the MAPE on the validation set
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[print_callback])
