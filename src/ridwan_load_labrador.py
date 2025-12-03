import os
import tensorflow as tf

# Import the model + its custom layers
from lab_transformers.models.labrador.model import Labrador, TransformerBlock
from lab_transformers.models.labrador.continuous_embedding_layer import ContinuousEmbedding
from lab_transformers.models.labrador.prediction_heads import MLMPredictionHead

# Tell Keras how to reconstruct the custom objects
custom_objects = {
    "Labrador": Labrador,
    "TransformerBlock": TransformerBlock,
    "ContinuousEmbedding": ContinuousEmbedding,
    "MLMPredictionHead": MLMPredictionHead,
}

# Resolve the model directory relative to this file
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_weights", "labrador")

print("Loading model from:", MODEL_DIR)

# Load the pretrained model
model = tf.keras.models.load_model(
    MODEL_DIR,
    custom_objects=custom_objects,
)

print("Loaded model type:", type(model))

# --- Optional: tiny dummy forward pass sanity check ---

# Labrador expects a dict with:
#   - categorical_input: (batch_size, max_bag_length) ints
#   - continuous_input:  (batch_size, max_bag_length, 1) floats

batch_size = 2
seq_len = 64

categorical_input = tf.zeros((batch_size, seq_len), dtype=tf.int32)
continuous_input = tf.zeros((batch_size, seq_len), dtype=tf.float32)

inputs = {
    "categorical_input": categorical_input,
    "continuous_input": continuous_input,
}

outputs = model(inputs, training=False)

print("Forward pass OK. Output type:", type(outputs))