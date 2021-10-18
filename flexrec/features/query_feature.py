'''An abstraction over each query feature'''

import numpy as np
import tensorflow as tf

from typing import Dict, Iterable, List, Optional, Tuple, Union


class QueryFeature:
    
  STR_TYPE: str = 'string'
  TXT_TYPE: str = 'text'
  INT_TYPE: str = 'integer'
  CDI_TYPE: str = 'continuous and discretized'
  CNO_TYPE: str = 'continuous and normalized'
  
  def __init__(self, 
               name: str, 
               data: tf.data.Dataset, 
               dtype: str, num_bins: int=1000):

    self.name = name
    self.data = data
    self.dtype = dtype
    
    if dtype in [QueryFeature.STR_TYPE, QueryFeature.TXT_TYPE, QueryFeature.INT_TYPE]:
      self.vocabulary = self.get_unique()
    else:
      self.vocabulary = None
    
    if dtype in [QueryFeature.CDI_TYPE]:
      self.buckets = self.get_buckets(num_bins)
    else:
      self.buckets = None
    
    if dtype in [QueryFeature.CNO_TYPE]:
      self.mean, self.variance = self.get_mean_variance()
    else:
      self.mean, self.variance = None, None

  def get_unique(self):
    return np.unique(list(self.data.as_numpy_iterator()))

  def get_buckets(self, num_bins: int):
    max_val = self.data.reduce(-np.Inf, tf.maximum).numpy()
    min_val = self.data.reduce(np.Inf, tf.minimum).numpy()
    return np.linspace(min_val, max_val, num=num_bins)

  def get_mean_variance(self):
    mean_metrics = tf.keras.metrics.Mean()
    mean_metrics.update_state(list(self.data.as_numpy_iterator()))
    mean = mean_metrics.result().numpy()
    centralized = self.data.map(lambda x: tf.square(x - mean))
    mean_metrics.reset_state()
    mean_metrics.update_state(list(centralized.as_numpy_iterator()))
    variance = mean_metrics.result().numpy()
    return mean, variance