import pickle

# Load the model
# *************************************************************************
with open('model_TiDE.pkl', 'rb') as f:
    model = pickle.load(f)

pred = SCALER.inverse_transform(
    model.predict(n=FORECAST_HORIZON, series=data_transformed, future_covariates=dynamic_covariates_transformed, num_samples=50))
