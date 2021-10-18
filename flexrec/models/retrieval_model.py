'''Retrieval of candidates for a given query'''

import tensorflow as tf
import tensorflow_recommenders as tfrs

from flexrec.models import QueryReduction
from typing import Dict, Iterable, List, Optional, Tuple, Union


class RetrievalModel(tf.keras.Model):
    
  REGULARIZATION = 1e-3
  GRAVITATION = 1e-7
  NORMALIZATION = 0.98 # normalization factor in [0, 1]

  def __init__(self, 
               query_model: tf.keras.Model, 
               candidate_model: tf.keras.Model,
               candidates: Union[Dict[str, tf.Tensor], tf.data.Dataset], 
               layer_sizes: Optional[Iterable[int]]=None):

    super().__init__()

    self.deep_retrieval = False if (layer_sizes is None) else True

    self.query_feature_names = query_model.feature_names
    self.query_model = query_model
    self.query_reduction = QueryReduction(query_model, layer_sizes)\
                           if self.deep_retrieval else query_model

    self.candidate_feature_names = candidate_model.feature_names
    self.candidate_model = candidate_model
    self.candidate_reduction = QueryReduction(candidate_model, layer_sizes)\
                               if self.deep_retrieval else candidate_model

#     self.candidates = {
#       feature_name: tf.convert_to_tensor([
#         candidate[feature_name] for candidate in list(movies.as_numpy_iterator())
#       ]) for feature_name in retrieval_model.candidate_feature_names
#     } if isinstance(candidates, tf.data.Dataset) else candidates
    
    self.__metrics = tfrs.metrics.FactorizedTopK(
      candidates=candidates.batch(128).map(lambda x: self.candidate_reduction(x))
    )
    
#   def call(self, queries: Dict[str, tf.Tensor], k: int=1) -> tf.Tensor:
    
#     queries_reduced = self.query_reduction({
#       feature_name: queries[feature_name] for feature_name in self.query_feature_names
#     })
#     candidates_reduced = self.candidate_reduction(self.candidates)

#     dot = tf.linalg.matmul(queries_reduced, candidates_reduced, transpose_b=True)
#     query_norms = tf.math.reduce_sum(tf.square(queries_reduced), axis=-1)
#     candidate_norms = tf.math.reduce_sum(tf.square(candidates_reduced), axis=-1)

#     cos_scores = (dot / tf.reshape(tf.math.pow(candidate_norms, 0.5 * RetrievalModel.NORMALIZATION), (1, -1))) /\
#                  (tf.reshape(tf.math.pow(query_norms, 0.5 * RetrievalModel.NORMALIZATION), (-1, 1)))
#     _, indices = tf.math.top_k(cos_scores, k=k)
    
#     return [{feature_name: self.candidates[feature_name].numpy()[idx.numpy()] for feature_name in self.candidate_feature_names}
#             for idx in indices]

  def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:

    query_reduced = self.query_reduction({
      feature_name: features[feature_name] for feature_name in self.query_feature_names
    })
    candidate_reduced = self.candidate_reduction({
      feature_name: features[feature_name] for feature_name in self.candidate_feature_names
    })
    
    dot = tf.math.reduce_sum(query_reduced * candidate_reduced, axis=-1)
    query_norms = tf.math.reduce_sum(tf.square(query_reduced), axis=-1)
    candidate_norms = tf.math.reduce_sum(tf.square(candidate_reduced), axis=-1)
    
    cos_loss = tf.math.reduce_sum(
      -dot / tf.math.pow(query_norms * candidate_norms, 0.5 * RetrievalModel.NORMALIZATION)
    )
    gravitation_loss = tf.math.reduce_sum(query_norms + candidate_norms)
    
    loss = cos_loss, gravitation_loss
    
    update_metrics = not training
    
    if not update_metrics:
      return loss

    with tf.control_dependencies([
      self.__metrics.update_state(query_reduced, candidate_reduced)
    ]):
      return loss

  def train_step(self, features: Dict[str, tf.Tensor]):

    with tf.GradientTape() as tape:
      cosine_loss, gravitation_loss = self.compute_loss(features, training=True)
      regularization_loss = sum(self.losses)
      total_loss = cosine_loss +\
                   RetrievalModel.GRAVITATION * gravitation_loss +\
                   RetrievalModel.REGULARIZATION * regularization_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {}
    metrics['cosine_loss'] = cosine_loss
    metrics['gravitation_loss'] = RetrievalModel.GRAVITATION * gravitation_loss
    metrics['regularization_loss'] = RetrievalModel.REGULARIZATION * regularization_loss
    metrics['total_loss'] = total_loss

    return metrics

  def test_step(self, features: Dict[str, tf.Tensor]):
        
    cosine_loss, gravitation_loss = self.compute_loss(features, training=False)
    regularization_loss = sum(self.losses)
    total_loss = cosine_loss +\
                 RetrievalModel.GRAVITATION * gravitation_loss +\
                 RetrievalModel.REGULARIZATION * regularization_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics['cosine_loss'] = cosine_loss
    metrics['gravitation_loss'] = RetrievalModel.GRAVITATION * gravitation_loss
    metrics['regularization_loss'] = RetrievalModel.REGULARIZATION * regularization_loss
    metrics['total_loss'] = total_loss

    return metrics