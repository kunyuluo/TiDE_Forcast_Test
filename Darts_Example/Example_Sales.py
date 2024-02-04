import torch
import numpy as np
import pandas as pd
import shutil
from darts.models import NHiTSModel, TiDEModel
from darts.datasets import AusBeerDataset
from darts.dataprocessing.transformers.scaler import Scaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.metrics import mae, mse, mape
import matplotlib.pyplot as plt

# Load the dataset
# *************************************************************************
series = AusBeerDataset().load()
# print(series.pd_dataframe())

# Split the dataset into train, validation, and test
# *************************************************************************
train, temp = series.split_after(0.6)
val, test = temp.split_after(0.5)
# print(test.pd_dataframe())

# Preview the data
# *************************************************************************
# plt.plot(train.pd_dataframe())
# plt.plot(val.pd_dataframe())
# plt.plot(test.pd_dataframe())
# plt.show()

# Standardization
# *************************************************************************
scaler = Scaler()  # default uses sklearn's MinMaxScaler
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)

# Model configuration
# *************************************************************************
optimizer_kwargs = {
    "lr": 1e-3,
}

# PyTorch Lightning Trainer arguments
pl_trainer_kwargs = {
    "gradient_clip_val": 1,
    "max_epochs": 200,
    "accelerator": "auto",
    "callbacks": [],
}

# learning rate scheduler
lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
lr_scheduler_kwargs = {
    "gamma": 0.999,
}

common_model_args = {
    "input_chunk_length": 12,  # lookback window
    "output_chunk_length": 12,  # forecast/lookahead window
    "optimizer_kwargs": optimizer_kwargs,
    "pl_trainer_kwargs": pl_trainer_kwargs,
    "lr_scheduler_cls": lr_scheduler_cls,
    "lr_scheduler_kwargs": lr_scheduler_kwargs,
    "likelihood": None,  # use a likelihood for probabilistic forecasts
    "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
    "force_reset": True,
    "batch_size": 256,
    "random_state": 42,
}
# Create the model
# *************************************************************************
model_tide = TiDEModel(**common_model_args, use_reversible_instance_norm=True)
model_tide.fit(series=train, val_series=val, verbose=True)

# Predictions
# *************************************************************************
pred_steps = common_model_args["output_chunk_length"] * 2
pred_input = test[:-pred_steps]
pred_output = model_tide.predict(n=pred_steps, series=pred_input)

print(f'mape is: ', mape(test, pred_output))

plt.plot(pred_input.pd_dataframe(), label='input')
plt.plot(test[-pred_steps:].pd_dataframe(), label='actual')
plt.plot(pred_output.pd_dataframe(), label='predicted')
plt.show()
