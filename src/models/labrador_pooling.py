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
        pooling_type: str = "mean",   # "mean", "attn", or "gated"
        attn_hidden_dim: int = 128,   # size of attention MLP
        gate_hidden_dim: int = 128,   # size of gate MLP (for "gated")
    ) -> None:
        """
        A wrapper for a Labrador model that allows for finetuning on a downstream task,
        with configurable pooling over the sequence of hidden states.

        pooling_type:
            "mean"  -> masked global average pooling (original behavior)
            "attn"  -> learned attention pooling over time
            "gated" -> learn a convex combination of mean and attn pooling
                       h = g * h_attn + (1 - g) * h_mean
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

        # Pooling configuration
        pooling_type = pooling_type.lower().strip()
        if pooling_type not in {"mean", "attn", "gated"}:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")
        self.pooling_type = pooling_type

        print(f"[DEBUG] Using pooling strategy: {self.pooling_type}")

        # Base Labrador encoder
        self.base_model = Labrador(self.transformer_params)
        if self.base_model_path is not None:
            print(f"\n Loading weights from {self.base_model_path} \n", flush=True)
            self.base_model.load_weights(base_model_path)

        # Use encoder only
        self.base_model.include_head = False
        self.base_model.trainable = train_base_model

        # Original mean-pooling layer (used in "mean" and "gated")
        self.global_avg_pool = keras.layers.GlobalAveragePooling1D()

        # Attention pooling layers (used in "attn" and "gated")
        # score_t = w2^T tanh(W1 h_t)
        self.attn_dense1 = keras.layers.Dense(attn_hidden_dim, activation="tanh")
        self.attn_dense2 = keras.layers.Dense(1, activation=None)

        # gate network (used only when pooling_type == "gated")
        # Input: concat[h_mean, h_attn]  -> gate g in (0,1)^D  (elementwise)
        self.gate_dense1 = keras.layers.Dense(gate_hidden_dim, activation="tanh")
        self.gate_dense2 = keras.layers.Dense(
            model_params["embedding_dim"], activation="sigmoid"
        )

        # Non-MIMIC tabular features projection
        self.dense_nonmimic = keras.layers.Dense(units=14, activation="relu")

        # Head
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)
        self.extra_dense_layer = keras.layers.Dense(units=1038, activation="relu")
        self.output_layer = keras.layers.Dense(
            units=self.output_size, activation=self.output_activation
        )

    def _compute_mask(self, categorical_input: tf.Tensor) -> tf.Tensor:
        """
        Build a float mask of shape (batch, seq_len) where 1.0 = valid token, 0.0 = padding.
        Assumes padding index is 0 in categorical_input.
        """
        mask = tf.cast(tf.math.not_equal(categorical_input, 0), tf.float32)
        return mask  # (B, T)

    def _mean_pool(self, seq_outputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Masked mean pooling using the original GlobalAveragePooling1D layer.
        seq_outputs: (B, T, D)
        mask: (B, T) with 1.0 for valid tokens and 0.0 for padding.
        """
        int_mask = tf.cast(mask, tf.int32)
        seq_outputs._keras_mask = None  # avoid Keras auto-mask confusion
        pooled = self.global_avg_pool(seq_outputs, mask=int_mask)
        return pooled  # (B, D)

    def _attn_pool(self, seq_outputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Learned attention pooling over time with masking.

        seq_outputs: (B, T, D)
        mask: (B, T) with 1.0 for valid tokens and 0.0 for padding.
        """
        # (B, T, D) -> (B, T, attn_hidden_dim) -> (B, T, 1)
        attn_hidden = self.attn_dense1(seq_outputs)
        attn_scores = self.attn_dense2(attn_hidden)  # (B, T, 1)
        attn_scores = tf.squeeze(attn_scores, axis=-1)  # (B, T)

        # Mask padded positions
        minus_inf = tf.constant(-1e9, dtype=attn_scores.dtype)
        attn_scores = tf.where(tf.equal(mask, 1.0), attn_scores, minus_inf)

        # Softmax over time
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)  # (B, T)

        # Weighted sum
        attn_weights_expanded = tf.expand_dims(attn_weights, axis=-1)  # (B, T, 1)
        pooled = tf.reduce_sum(attn_weights_expanded * seq_outputs, axis=1)  # (B, D)
        return pooled

    def _gated_pool(self, seq_outputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Gated pooling: learn a convex combination of mean and attention pooling.

        h_mean  = mean_pool(H)
        h_attn  = attn_pool(H)
        g       = gate(h_mean, h_attn) in (0,1)^D
        h_final = g * h_attn + (1 - g) * h_mean
        """
        h_mean = self._mean_pool(seq_outputs, mask)   # (B, D)
        h_attn = self._attn_pool(seq_outputs, mask)   # (B, D)

        # Concatenate and pass through a small MLP to get gate vector g
        h_concat = tf.concat([h_mean, h_attn], axis=-1)  # (B, 2D)
        g_hidden = self.gate_dense1(h_concat)            # (B, gate_hidden_dim)
        g = self.gate_dense2(g_hidden)                   # (B, D), values in (0,1)

        # Convex combination
        h_final = g * h_attn + (1.0 - g) * h_mean        # (B, D)
        return h_final

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        # Truncate input sequences to max_seq_length
        base_model_inputs = {
            "categorical_input": inputs["categorical_input"][:, : self.max_seq_len],
            "continuous_input": inputs["continuous_input"][:, : self.max_seq_len],
        }

        # Encode lab sequence with pretrained Labrador
        seq_outputs = self.base_model(base_model_inputs, training=training)
        # seq_outputs: (B, T, D)

        # Build mask from categorical_input
        mask = self._compute_mask(base_model_inputs["categorical_input"])  # (B, T)

        # Pooling over time
        if self.pooling_type == "mean":
            x = self._mean_pool(seq_outputs, mask)
        elif self.pooling_type == "attn":
            x = self._attn_pool(seq_outputs, mask)
        elif self.pooling_type == "gated":
            x = self._gated_pool(seq_outputs, mask)
        else:
            raise ValueError(f"Unsupported pooling_type: {self.pooling_type}")

        # Optional non-MIMIC features
        if "non_mimic_features" in inputs:
            non_mimic_features = self.dense_nonmimic(inputs["non_mimic_features"])
            x = tf.concat([x, non_mimic_features], axis=1)

        # Optional extra dense layer (original behavior)
        if self.add_extra_dense_layer:
            x = self.extra_dense_layer(x)

        x = self.dropout(x, training=training)
        outputs = self.output_layer(x)
        return outputs
