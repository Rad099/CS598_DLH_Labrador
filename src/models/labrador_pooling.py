from typing import Union, Dict, Any

import tensorflow as tf
from tensorflow import keras

from lab_transformers.models.labrador.model import Labrador


class LabradorPoolingWrapper(keras.Model):
    def __init__(
        self,
        base_model_path: Union[None, str],
        output_size: int,
        output_activation: str,
        model_params: Dict[str, Any],
        dropout_rate: float,
        add_extra_dense_layer: bool,
        train_base_model: bool = False,
        pooling_type: str = "mean",   # NEW: "mean" or "attn"
        attn_hidden_dim: int = 128,   # NEW: size of attention MLP
    ) -> None:
        """
        A wrapper for a Labrador model that allows for finetuning on a downstream task,
        with configurable pooling over the sequence of hidden states.

        pooling_type:
            "mean" -> masked global average pooling (original behavior)
            "attn" -> learned attention pooling over time
        """

        super(LabradorPoolingWrapper, self).__init__()

        self.transformer_params = model_params
        self.base_model_path = base_model_path
        self.output_size = output_size
        self.output_activation = output_activation
        self.train_base_model = train_base_model
        self.dropout_rate = dropout_rate
        self.max_seq_len = model_params["max_seq_length"]
        self.add_extra_dense_layer = add_extra_dense_layer

        # NEW: pooling configuration
        pooling_type = pooling_type.lower()
        if pooling_type not in {"mean", "attn"}:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")
        self.pooling_type = pooling_type

        # Base Labrador encoder
        self.base_model = Labrador(self.transformer_params)
        if self.base_model_path is not None:
            print(f"\n Loading weights from {self.base_model_path} \n", flush=True)
            self.base_model.load_weights(base_model_path)

        # We only use the encoder, not any head inside Labrador
        self.base_model.include_head = False
        self.base_model.trainable = train_base_model

        # Original mean-pooling layer (still used when pooling_type == "mean")
        self.global_avg_pool = keras.layers.GlobalAveragePooling1D()

        # NEW: attention pooling layers (used when pooling_type == "attn")
        # Simple MLP-style attention: tanh(W1 h_t) -> score_t = w2^T ...
        self.attn_dense1 = keras.layers.Dense(attn_hidden_dim, activation="tanh")
        self.attn_dense2 = keras.layers.Dense(1, activation=None)

        # Non-MIMIC tabular features projection
        self.dense_nonmimic = keras.layers.Dense(units=14, activation="relu")

        # Head
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)
        self.extra_dense_layer = keras.layers.Dense(units=1038, activation="relu")
        self.output_layer = keras.layers.Dense(
            units=self.output_size, activation=self.output_activation
        )

    def _compute_mask(
        self, categorical_input: tf.Tensor
    ) -> tf.Tensor:
        """
        Build a float mask of shape (batch, seq_len) where 1.0 = valid token, 0.0 = padding.
        Assumes padding index is 0 in categorical_input.
        """
        # categorical_input: (B, T)
        mask = tf.cast(tf.math.not_equal(categorical_input, 0), tf.float32)
        return mask  # (B, T)

    def _mean_pool(
        self, seq_outputs: tf.Tensor, mask: tf.Tensor
    ) -> tf.Tensor:
        """
        Masked mean pooling using the original GlobalAveragePooling1D layer.
        seq_outputs: (B, T, D)
        mask: (B, T) with 1.0 for valid tokens and 0.0 for padding.
        """
        # Keras GlobalAveragePooling1D expects an int mask
        int_mask = tf.cast(mask, tf.int32)
        seq_outputs._keras_mask = None  # avoid Keras auto-mask confusion
        pooled = self.global_avg_pool(seq_outputs, mask=int_mask)
        return pooled  # (B, D)

    def _attn_pool(
        self, seq_outputs: tf.Tensor, mask: tf.Tensor
    ) -> tf.Tensor:
        """
        Learned attention pooling over time with masking.

        seq_outputs: (B, T, D)
        mask: (B, T) with 1.0 for valid tokens and 0.0 for padding.
        """
        # Compute attention scores
        # (B, T, D) -> (B, T, attn_hidden_dim) -> (B, T, 1)
        attn_hidden = self.attn_dense1(seq_outputs)
        attn_scores = self.attn_dense2(attn_hidden)  # (B, T, 1)
        attn_scores = tf.squeeze(attn_scores, axis=-1)  # (B, T)

        # Apply mask: set scores of padded positions to large negative
        # so they get ~0 probability after softmax.
        minus_inf = tf.constant(-1e9, dtype=attn_scores.dtype)
        attn_scores = tf.where(tf.equal(mask, 1.0), attn_scores, minus_inf)

        # Normalize to get attention weights
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)  # (B, T)

        # Weighted sum of sequence outputs
        attn_weights_expanded = tf.expand_dims(attn_weights, axis=-1)  # (B, T, 1)
        pooled = tf.reduce_sum(attn_weights_expanded * seq_outputs, axis=1)  # (B, D)
        return pooled

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        # Truncate input sequences to max_seq_length
        base_model_inputs = {
            "categorical_input": inputs["categorical_input"][:, : self.max_seq_len],
            "continuous_input": inputs["continuous_input"][:, : self.max_seq_len],
        }

        # Encode lab sequence with pretrained Labrador
        seq_outputs = self.base_model(base_model_inputs, training=training)
        # seq_outputs: (B, T, D)

        # Build mask from categorical_input (padding assumed to be 0)
        mask = self._compute_mask(base_model_inputs["categorical_input"])  # (B, T)

        # Pooling over time: mean (original) or learned attention
        if self.pooling_type == "mean":
            x = self._mean_pool(seq_outputs, mask)
        elif self.pooling_type == "attn":
            x = self._attn_pool(seq_outputs, mask)
        else:
            raise ValueError(f"Unsupported pooling_type: {self.pooling_type}")

        # Optional non-MIMIC features (tabular features concatenated to pooled embedding)
        if "non_mimic_features" in inputs:
            non_mimic_features = self.dense_nonmimic(inputs["non_mimic_features"])
            x = tf.concat([x, non_mimic_features], axis=1)

        # Optional extra dense layer (matches original behavior)
        if self.add_extra_dense_layer:
            x = self.extra_dense_layer(x)

        # Dropout + final prediction layer
        x = self.dropout(x, training=training)
        outputs = self.output_layer(x)
        return outputs
