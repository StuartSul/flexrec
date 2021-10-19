'''Retrieval of candidates for a given query'''

import tensorflow as tf
import tensorflow_recommenders as tfrs

from flexrec.models import QueryReduction
from typing import Dict, Iterable, List, Optional, Tuple, Union


class RetrievalModel(tf.keras.Model):

  def __init__(self, 
               query_model: tf.keras.Model, 
               candidate_model: tf.keras.Model,
               candidates: tf.data.Dataset, 
               layer_sizes: Optional[Iterable[int]]=None,
               normalization: float=0.99,
               regularization: float=1e-4,
               gravitation: float=1e-6):

    super().__init__()

    self.query_feature_names = query_model.feature_names
    self.query_model = query_model
    self.query_reduction = QueryReduction(query_model, layer_sizes)

    self.candidate_feature_names = candidate_model.feature_names
    self.candidate_model = candidate_model
    self.candidate_reduction = QueryReduction(candidate_model, layer_sizes)

    self.candidates = candidates # must be callable as reduction.predict(candidates.batch())
    self._candidates_batched = candidates.batch(candidates.cardinality()).get_single_element()
    self._candidates_cached = None # cached embeddings of candidates
    
    self.normalization = normalization
    self.gravitation = gravitation
    self.regularization = regularization
    
    self.tok_k_metrics = {
      k: tf.keras.metrics.Mean(name=f"top_{k}_accuracy") for k in (1, 5, 10, 50, 100)
    }
    
  def cosine_score(self, query_reduced: tf.Tensor, candidate_reduced: tf.Tensor, pairwise: bool):
        
    query_norm = tf.math.reduce_sum(tf.square(query_reduced), axis=-1)
    candidate_norm = tf.math.reduce_sum(tf.square(candidate_reduced), axis=-1)
        
    if pairwise:
      dot = tf.math.reduce_sum(query_reduced * candidate_reduced, axis=-1)
      score = dot / tf.math.pow(query_norm * candidate_norm, 0.5 * self.normalization)
    else:
      dot = tf.linalg.matmul(query_reduced, candidate_reduced, transpose_b=True)
      score = (
        dot / tf.reshape(tf.math.pow(candidate_norm, 0.5 * self.normalization), (1, -1))
      ) / (
        tf.reshape(tf.math.pow(query_norm, 0.5 * self.normalization), (-1, 1))
      )

    return score, query_norm, candidate_norm

  def call(self, query: Dict[str, tf.Tensor], k: int=100) -> tf.Tensor:

    query_reduced = self.query_reduction({
      feature_name: query[feature_name] for feature_name in self.query_feature_names
    })
    candidate_reduced = self.candidate_reduction(self._candidates_batched)

    cosine_score, _, _ = self.cosine_score(query_reduced, candidate_reduced, pairwise=False)
    _, indices = tf.math.top_k(cosine_score, k=k)
    
    return indices

  def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:

    query_reduced = self.query_reduction({
      feature_name: features[feature_name] for feature_name in self.query_feature_names
    })
    candidate_reduced = self.candidate_reduction({
      feature_name: features[feature_name] for feature_name in self.candidate_feature_names
    })

    cosine_score, query_norm, candidate_norm = self.cosine_score(
      query_reduced, candidate_reduced, pairwise=True
    )

    cosine_loss = tf.math.reduce_sum(-cosine_score)
    gravitation_loss = tf.math.reduce_sum(query_norm) + tf.math.reduce_sum(candidate_norm)
    loss = cosine_loss, gravitation_loss

    if not training:
      self.update_top_k_metrics(query_reduced, cosine_score)

    return loss

  def update_top_k_metrics(self, 
                           query_reduced: tf.Tensor,
                           true_candidate_score: tf.Tensor):

    candidate_reduced = self.candidate_reduction(self._candidates_batched)
    
    cosine_score, _, _ = self.cosine_score(query_reduced, candidate_reduced, pairwise=False)

    for k in self.tok_k_metrics:
      self.tok_k_metrics[k].update_state(
        tf.math.in_top_k(
          targets=tf.zeros(tf.shape(cosine_score)[0], dtype=tf.int32),
          predictions=tf.concat([tf.reshape(true_candidate_score, (-1, 1)), cosine_score], axis=1),
          k=k
        )
      )

  def reset_metrics(self):
    for m in self.metrics:
      m.reset_state()
    for k in self.tok_k_metrics:
      self.tok_k_metrics[k].reset_state()

  def train_step(self, features: Dict[str, tf.Tensor]):

    with tf.GradientTape() as tape:
      cosine_loss, gravitation_loss = self.compute_loss(features, training=True)
      regularization_loss = sum(self.losses)
      total_loss = cosine_loss +\
                   self.gravitation * gravitation_loss +\
                   self.regularization * regularization_loss
    
    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {}
    metrics['cosine_loss'] = cosine_loss
    metrics['gravitation_loss'] = self.gravitation * gravitation_loss
    metrics['regularization_loss'] = self.regularization * regularization_loss
    metrics['total_loss'] = total_loss

    return metrics

  def test_step(self, features: Dict[str, tf.Tensor]):

    cosine_loss, gravitation_loss = self.compute_loss(features, training=False)
    regularization_loss = sum(self.losses)
    total_loss = cosine_loss +\
                 self.gravitation * gravitation_loss +\
                 self.regularization * regularization_loss

    metrics = {self.tok_k_metrics[k].name: self.tok_k_metrics[k].result() for k in self.tok_k_metrics}
    metrics['cosine_loss'] = cosine_loss
    metrics['gravitation_loss'] = self.gravitation * gravitation_loss
    metrics['regularization_loss'] = self.regularization * regularization_loss
    metrics['total_loss'] = total_loss

    return metrics