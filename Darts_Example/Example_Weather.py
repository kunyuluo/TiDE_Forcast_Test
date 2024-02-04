from darts.datasets import WeatherDataset
from darts.models import TiDEModel

series = WeatherDataset().load()
# print(series.pd_dataframe())

# predicting atmospheric pressure
target = series['p (mbar)'][:100]
print(target.pd_dataframe())

# optionally, use past observed rainfall (pretending to be unknown beyond index 100)
past_cov = series['rain (mm)'][:100]

# optionally, use future temperatures (pretending this component is a forecast)
future_cov = series['T (degC)'][:106]

model = TiDEModel(
    input_chunk_length=6,
    output_chunk_length=6,
    n_epochs=20
)

model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
pred = model.predict(6)
print(pred.values())
