import pandas as pd
import numpy as np
from Helper import DataProcessor, TimeSeriesData
from Time_features import TimeCovariates

data = pd.read_csv('new_data.csv')
data['data_time'] = pd.to_datetime(data['data_time'])
data.set_index('data_time', inplace=True)
df_idx = data.index
# print(df_idx)

time_df = TimeCovariates(df_idx, normalized=False, holiday=False).get_covariates()
# print(time_df)

num_features = ['cp_power', 'oat', 'oah', 'downstream_chwsstpt']

dtl = TimeSeriesData(
      data_path='new_data.csv',
      datetime_col='data_time',
      num_cov_cols=num_features,
      cat_cov_cols=None,
      ts_cols=np.array(ts_cols),
      train_range=[0, boundaries[0]],
      val_range=[boundaries[0], boundaries[1]],
      test_range=[boundaries[1], boundaries[2]],
      hist_len=24,
      pred_len=18,
      batch_size=min(256, len(ts_cols)),
      freq='H',
      normalize=True,
      epoch_len=FLAGS.epoch_len,
      holiday=False,
      permute=True,
  )
