'''A multi-objective trainer for natural transfer learning'''

import tensorflow as tf

from typing import Dict, Iterable, List, Optional, Tuple, Union


class MultitaskTrainer(tf.keras.Model):

  REGULARIZATION = 0.01

  def __init__(self, 
               retrieval_model: tf.keras.Model, 
               ranking_model: tf.keras.Model, 
               retrieval_weight: float, 
               ranking_weight: float):

    super().__init__()

    if retrieval_model.query_model is not ranking_model.query_model or\
       retrieval_model.candidate_model is not ranking_model.candidate_model:
      raise RuntimeError('Retrieval and ranking models must use identical embedding tables.')

    self.query_feature_names = retrieval_model.query_model.feature_names
    self.candidate_feature_names = retrieval_model.candidate_model.feature_names
    self.retrieval_model = retrieval_model
    self.ranking_model = ranking_model
    self.retrieval_weight = retrieval_weight
    self.ranking_weight = ranking_weight

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    query_embeddings = self.retrieval_model.query_model({
      feature_name: features[feature_name] for feature_name in self.query_feature_names
    })
    candidate_embeddings = self.retrieval_model.candidate_model({
      feature_name: features[feature_name] for feature_name in self.candidate_feature_names
    })

    if self.retrieval_model.deep_retrieval:
      query_reduced = self.retrieval_model.query_reduction.query_reduction(query_embeddings)
      candidate_reduced = self.retrieval_model.candidate_reduction.query_reduction(candidate_embeddings)
    else:
      query_reduced = query_embeddings
      candidate_reduced = candidate_embeddings
    
    x = tf.concat([query_embeddings, candidate_embeddings], axis=1)
    if self.ranking_model.interaction is not None:
      x = self.ranking_model.interaction(x)
    x = self.ranking_model.mlp(x)
    scores = self.ranking_model.logit_layer(x)
    
    return query_reduced, candidate_reduced, scores

  def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("user_rating")
    query_reduced, candidate_reduced, scores = self(features)

    retrieval_loss = self.retrieval_model.task(
      query_reduced, 
      candidate_reduced,
      compute_metrics=not training
    )
    ranking_loss = self.ranking_model.task(
      labels=labels,
      predictions=scores,
    )

    return retrieval_loss, ranking_loss

  def train_step(self, features: Dict[str, tf.Tensor]):
        
    with tf.GradientTape() as tape:
      retrieval_loss, ranking_loss = self.compute_loss(features, training=True)
      regularization_loss = sum(self.losses)
      gravitation_loss = 0.
      total_loss = self.retrieval_weight * retrieval_loss +\
                   self.ranking_weight * ranking_loss +\
                   MultitaskTrainer.REGULARIZATION * regularization_loss +\
                   gravitation_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {}
    metrics['retrieval_loss'] = retrieval_loss
    metrics['ranking_weight'] = ranking_loss
    metrics['regularization_loss'] = regularization_loss
    metrics['gravitation_loss'] = gravitation_loss
    metrics['total_loss'] = total_loss
    
    return metrics

  def test_step(self, features: Dict[str, tf.Tensor]):

    retrieval_loss, ranking_loss = self.compute_loss(features, training=True)
    regularization_loss = sum(self.losses)
    gravitation_loss = 0.
    total_loss = self.retrieval_weight * retrieval_loss +\
                 self.ranking_weight * ranking_loss +\
                 MultitaskTrainer.REGULARIZATION * regularization_loss +\
                 gravitation_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics['retrieval_loss'] = retrieval_loss
    metrics['ranking_weight'] = ranking_loss
    metrics['regularization_loss'] = regularization_loss
    metrics['gravitation_loss'] = gravitation_loss
    metrics['total_loss'] = total_loss

    return metrics