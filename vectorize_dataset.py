import tensorflow as tf
from tensorflow.keras import layers

import config


def create_text_vectorization_layer(raw_train_ds):
  vectorize_layer = layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=config.max_features,
    output_mode='int',
    output_sequence_length=config.sequence_length # output sequence length
    )
  # adapt the vectorization layer to the training data
  train_text = raw_train_ds.map(lambda x, y: x)
  vectorize_layer.adapt(train_text)
  return vectorize_layer


def vectorize_text(text, label, vectorize_layer):
  text = tf.expand_dims(text, -1) # add a dimension for the text vectorization layer
  return vectorize_layer(text), label


def vectorize_dataset(dataset, vectorize_layer):
  vectorized_dataset = dataset.map(lambda text, label: vectorize_text(text, label, vectorize_layer)) # vectorize text data
  vectorized_dataset = vectorized_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE) # performance optimization
  return vectorized_dataset