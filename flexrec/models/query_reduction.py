'''Reduction of 2-D query embeddings into 1-D query embeddings'''

import tensorflow as tf

from typing import Dict, Iterable, List, Optional, Tuple, Union


class QueryReduction(tf.keras.Model):

  def __init__(self, 
               query_model: tf.keras.Model, 
               layer_sizes: Iterable[int]):
    '''
    Helper class for RetrievalModel
    
      Paramters:
        query_model: a QueryModel instance
        layer_sizes: hidden layer dimensions used for MLP reduction
    '''
    
    super().__init__()
    
    self.query_model = query_model
    self.query_reduction = tf.keras.Sequential()
    self.query_reduction.add(tf.keras.layers.Flatten())
    for layer_size in layer_sizes[:-1]:
      self.query_reduction.add(
        tf.keras.layers.Dense(
          layer_size, 
          activation="relu", 
          kernel_regularizer=tf.keras.regularizers.L2()
        )
      )
    self.query_reduction.add(
      tf.keras.layers.Dense(
        layer_sizes[-1], 
        kernel_regularizer=tf.keras.regularizers.L2()
      )
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.query_reduction(self.query_model(features))