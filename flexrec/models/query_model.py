'''Mapping of a query to the embedding space'''

import tensorflow as tf

from flexrec.features import QueryFeature
from typing import Dict, Iterable, List, Optional, Tuple, Union


class QueryModel(tf.keras.Model):

  def __init__(self, 
               features: Iterable[QueryFeature], 
               embedding_dim: int=32):
    '''
    Builds a query model
    
      Paramters:
        features: a collection of QueryFeature instances
        embedding_dim: latent space dimension
    '''
    
    super().__init__()
    
    self.feature_names = {feature.name for feature in features}
    self.embedding_dim = embedding_dim
    self.embeddings = {}
    self.continuous = {}
    
    for feature in features:
        
      if feature.dtype == QueryFeature.STR_TYPE:
        self.embeddings[feature.name] = tf.keras.Sequential([
          tf.keras.layers.StringLookup(vocabulary=feature.vocabulary, mask_token=None),
          tf.keras.layers.Embedding(
            len(feature.vocabulary) + 1, self.embedding_dim, 
            embeddings_regularizer=tf.keras.regularizers.L2()
          )
        ], name=feature.name)

      elif feature.dtype == QueryFeature.TXT_TYPE:
        self.embeddings[feature.name] = tf.keras.Sequential([
          tf.keras.layers.TextVectorization(vocabulary=feature.vocabulary),
          tf.keras.layers.Embedding(
            len(feature.vocabulary) + 2, self.embedding_dim, mask_zero=True, 
            embeddings_regularizer=tf.keras.regularizers.L2()
          ),
          tf.keras.layers.GlobalAveragePooling1D() # various other methods possible
        ], name=feature.name)

      elif feature.dtype == QueryFeature.INT_TYPE:
        self.embeddings[feature.name] = tf.keras.Sequential([
          tf.keras.layers.IntegerLookup(vocabulary=feature.vocabulary, mask_token=None),
          tf.keras.layers.Embedding(
            len(feature.vocabulary) + 1, self.embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.L2()
          )
        ], name=feature.name)

      elif feature.dtype == QueryFeature.CDI_TYPE:
        self.embeddings[feature.name] = tf.keras.Sequential([
          tf.keras.layers.Discretization(feature.buckets.tolist()),
          tf.keras.layers.Embedding(
            len(feature.buckets) + 2, self.embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.L2()
          )
        ], name=feature.name)

      elif feature.dtype == QueryFeature.CNO_TYPE:
        self.continuous[feature.name] = tf.keras.Sequential([
          tf.keras.layers.Normalization(mean=feature.mean, variance=feature.variance, axis=None),
          tf.keras.layers.Reshape((1,))
        ], name=feature.name)
    
    if len(self.continuous) > 0:
      self.mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(
          embedding_dim, 
          kernel_regularizer=tf.keras.regularizers.L2()
        )
      ])

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    
    embeddings = []
    for feature_name in self.embeddings:
      embeddings.append(
        self.embeddings[feature_name](features[feature_name])
      )
    
    continuous = []
    for feature_name in self.continuous:
      continuous.append(
        self.continuous[feature_name](features[feature_name])
      )
    
    if len(continuous) > 0:
      embeddings.append(
        self.mlp(tf.concat(continuous, axis=1))
      )
    
    return tf.stack(embeddings, axis=1)