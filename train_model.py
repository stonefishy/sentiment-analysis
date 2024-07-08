import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import losses

import config


def build_model(train_ds, val_ds):
  model = tf.keras.Sequential([
    # Embedding layer with 64 dimensions
    layers.Embedding(config.max_features, config.embedding_dim),
    layers.Dropout(0.3),
    # average pooling layer to get the mean of all the word embeddings in a sentence
    layers.GlobalAveragePooling1D(), 
    layers.Dropout(0.2),
    # fully connected layer with 64 nerual units, relual activation function f(x)=max(0, x). Popular for deep neural networks
    layers.Dense(config.embedding_dim, activation='relu'),
    # dropout layer to prevent overfitting, 0.2 means 20% of the neurons will be dropped out
    layers.Dropout(0.2), 
    layers.Dense(config.embedding_dim, activation='relu'),
    layers.Dropout(0.1),
    # output layer with sigmoid activation function, output probabilities (0 - 1)
    layers.Dense(1, activation='sigmoid')]) 
  model.summary()

  # compile the model
  model.compile(
     # Loss function (二分类交叉熵损失函数), aim to calculate the difference between the predicted and actual values
      loss=losses.BinaryCrossentropy(), 
      # A adam optimizer(自适应矩估计优化器)， a popular optimizer for neural networks, aim to minimize the loss function
      optimizer='adam', 
      # Binary Accuracy metric(二分类准确率评估指标)，>0.5 is positive，<0.5 is negative
      metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)]
    ) 

  # train the model
  history = model.fit(
      train_ds, # training dataset  
      validation_data=val_ds, # validation dataset
      epochs=config.epochs # number of training loops over the entire dataset
    )
  
  return model, history


def save_model(vectorize_layer, model, model_saved_path):
  export_model = tf.keras.Sequential([
    vectorize_layer, # text vectorization layer
    model # the model trained before
  ])

  export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), 
    optimizer="adam", 
    metrics=['accuracy']
  )

  # save model
  export_model.save(model_saved_path)
  return export_model


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model
