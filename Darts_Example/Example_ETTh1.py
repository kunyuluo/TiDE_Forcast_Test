from darts.datasets import ETTh1Dataset
from darts.models import TiDEModel
from darts.dataprocessing.transformers import Scaler

series = ETTh1Dataset().load()
# print(series.pd_dataframe())

train, test = series[:-96], series[-96:]

train_scaler = Scaler()
scaled_train = train_scaler.fit_transform(train)

tide = TiDEModel(
    input_chunk_length=48,
    output_chunk_length=24,
    num_encoder_layers=2,
    num_decoder_layers=2,
    decoder_output_dim=32,
    hidden_size=512,
    temporal_decoder_hidden=16,
    use_layer_norm=True,
    dropout=0.5,
    random_state=42,
    use_reversible_instance_norm=True)

tide.fit(
    scaled_train,
    epochs=5
)

# Save models
# *************************************************************************
# with open('model_ETTh1.pkl', 'wb') as f:
#     pickle.dump(tide, f)
tide.save('model_ETTh1.pkl')
