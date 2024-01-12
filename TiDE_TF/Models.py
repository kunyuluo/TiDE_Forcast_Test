import numpy as np
import tensorflow as tf
from tensorflow import keras

EPS = 1e-7

train_loss = keras.losses.MeanSquaredError()


class MLPResidual(keras.layers.Layer):
    """
    Simple one hidden state residual network.
    """
    def __init__(self, hidden_dim, output_dim, layer_norm=False, dropout_rate=0.0):
        super(MLPResidual, self).__init__()

        self.lin_a = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.lin_b = tf.keras.layers.Dense(output_dim, activation=None)
        self.lin_res = tf.keras.layers.Dense(output_dim, activation=None)

        if layer_norm:
            self.lnorm = tf.keras.layers.LayerNormalization()
        self.layer_norm = layer_norm

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    @tf.function
    def call(self, inputs):
        """
        Call method.
        """
        h_state = self.lin_a(inputs)
        out = self.lin_b(h_state)
        out = self.dropout(out)
        res = self.lin_res(inputs)

        if self.layer_norm:
            return self.lnorm(out + res)
        return out + res


def make_dnn_residual(hidden_dims, layer_norm=False, dropout_rate=0.0):
    """
    Multi-layer DNN residual model.
    """
    if len(hidden_dims) < 2:
        return keras.layers.Dense(hidden_dims[-1], activation=None)

    layers = []
    for i, hdim in enumerate(hidden_dims[:-1]):
        layers.append(MLPResidual(hdim, hidden_dims[i+1], layer_norm=layer_norm, dropout_rate=dropout_rate))

    return keras.Sequential(layers)


# l1 = tf.keras.layers.Dense(5)
# l2 = MLPResidual(5, 4)
# print(type(l1))
# print(type(l2))


class TiDEModel(keras.Model):
    """
    Main class for multi-scale DNN model
    """
    def __init__(
            self,
            model_config,
            pred_len,
            cat_sizes,
            num_ts,
            transform=False,
            cat_emb_size=4,
            layer_norm=False,
            dropout_rate=0.0):
        """
        :param model_config: configurations specific to the model.
        :param pred_len: prediction horizon length.
        :param cat_sizes: number of categories in each categorical covariate.
        :param num_ts: number of time-series in the dataset
        :param transform: apply reversible transform or not.
        :param cat_emb_size: embedding size of categorical variables.
        :param layer_norm: use layer norm or not.
        :param dropout_rate: level of dropout.
        """

        super().__init__()
        self.model_config = model_config
        self.transform = transform

        if self.transform:
            self.affine_weight = self.add_weight(
                name='affine_weight', shape=(num_ts,), initializer='ones', Trainable=True)
            self.affine_bias = self.add_weight(
                name='affine_bias', shape=(num_ts,), initializer='zeros', Trainable=True)

        self.pred_len = pred_len

        self.encoder = make_dnn_residual(
            model_config.get('hidden_dims'),
            layer_norm=layer_norm,
            dropout_rate=dropout_rate)

        self.decoder = make_dnn_residual(
            model_config.get('hidden_dims')[:, -1] + [model_config.get('decoder_output_dim') * self.pred_len],
            layer_norm=layer_norm,
            dropout_rate=dropout_rate)

        self.linear = tf.keras.layers.Dense(self.pred_len, activation=None)

        self.time_encoder = make_dnn_residual(
            model_config.get('time_encoder_dims'), layer_norm=layer_norm, dropout_rate=dropout_rate)

        self.final_decoder = MLPResidual(
            model_config.get('final_decoder_hidden'),
            output_dim=1,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate)

        self.cat_embs = []
        for cat_size in cat_sizes:
            self.cat_embs.append(tf.keras.layers.Embedding(cat_size, cat_emb_size))

        self.ts_embs = tf.keras.layers.Embedding(num_ts, 16)

    @tf.function
    def assemble_features(self, features, cfeatures):
        """Assemble all features"""
        all_features = [features]
        for i, emb in enumerate(self.cat_embs):
            all_features.append(tf.transpose(emb(cfeatures[i, :])))
        return tf.concat(all_features, axis=0)

    @tf.function
    def call(self, inputs):
        """Call function that takes in a batch of training data and features."""
        past_data = inputs[0]
        future_features = inputs[1]
        batch_size = past_data[0].shape[0]
        ts_idx = inputs[2]
        past_features = self.assemble_features(past_data[1], past_data[2])
        future_features = self.assemble_features(future_features[0], future_features[1])
        past_ts = past_data[0]

        if self.transform:
            affine_weight = tf.gather(self.affine_weight, ts_idx)
            affine_bias = tf.gather(self.affine_bias, ts_idx)
            batch_mean = tf.math.reduce_mean(past_ts, axis=1)
            batch_std = tf.math.reduce_std(past_ts, axis=1)
            batch_std = tf.where(tf.math.equal(batch_std, 0), tf.ones_like(batch_std), batch_std)

            past_ts = (past_ts - batch_mean[:, ]) / batch_std[:, ]
            past_ts = affine_weight[:, ] * past_ts + affine_bias[:, ]

        encoded_past_features = tf.transpose(self.time_encoder(tf.transpose(past_features)))
        encoded_future_features = tf.transpose(self.time_encoder(tf.transpose(future_features)))
