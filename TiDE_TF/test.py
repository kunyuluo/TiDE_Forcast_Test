import pandas as pd
import numpy as np
# from Helper import DataProcessor
from Data_Loader import TimeSeriesData
from Time_features import TimeCovariates

data = pd.read_csv('../dummy_data.csv')
data['data_time'] = pd.to_datetime(data['data_time'])
data.set_index('data_time', inplace=True)
df_idx = data.index
# print(df_idx)

time_df = TimeCovariates(df_idx, normalized=True, holiday=False).get_covariates()
# print(time_df)

num_features = ['cp_power', 'oat', 'oah', 'downstream_chwsstpt', 'category']
target = ['cp_power']
num_cov_cols = ['oat', 'oah', 'downstream_chwsstpt']
cat_cov_cols = ['category1', 'category2']

dtl = TimeSeriesData(
    data_path='../dummy_data.csv',
    datetime_col='data_time',
    num_cov_cols=num_cov_cols,
    cat_cov_cols=cat_cov_cols,
    ts_cols=np.array(target),
    train_range=[0, 1000],
    val_range=[1000, 1300],
    test_range=[1300, 1440],
    hist_len=24,
    pred_len=18,
    batch_size=min(256, len(target)),
    freq='H',
    normalize=False,
    epoch_len=None,
    holiday=False,
    permute=True,
)

# perm = np.arange(24, 982)
# perm = np.random.permutation(perm)
# num_ts = len(target)
# batch_size = 4
# permute = True
#
# for idx in perm[0:20]:
#     for _ in range(num_ts // batch_size + 1):
#         if permute:
#             tsidx = np.random.choice(num_ts, size=batch_size, replace=False)
#         else:
#             tsidx = np.arange(num_ts)
#         dtimes = np.arange(idx - 24, idx + 18)

print(dtl.train_gen())
