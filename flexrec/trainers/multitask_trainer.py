'''A multi-objective trainer for natural transfer learning'''

import tensorflow as tf

from typing import Dict, Iterable, List, Optional, Tuple, Union


class MultitaskTrainer(tf.keras.Model):

  def __init__(self, 
               retrieval_models: List[tf.keras.Model], 
               ranking_models: List[tf.keras.Model], 
               retrieval_weights: Iterable[float], 
               ranking_weights: Iterable[float],
               regularization: float=1e-2):

    super().__init__()

    all_models = retrieval_models + ranking_models
    for model_1 in all_models:
      for model_2 in all_models:
        if model_1.query_model is not model_2.query_model or\
           model_1.candidate_model is not model_2.candidate_model:
          raise RuntimeError('All models must use identical embedding tables.')

    self.retrieval_models = retrieval_models
    self.ranking_models = ranking_models
    self.retrieval_weights = retrieval_weights
    self.ranking_weights = ranking_weights
    self.regularization = regularization
    
    # extract feature names, query model, and query reduction from one model
    self.query_feature_names = retrieval_models[0].query_model.feature_names
    self.candidate_feature_names = retrieval_models[0].candidate_model.feature_names
    self.query_model = retrieval_models[0].query_model
    self.candidate_model = retrieval_models[0].candidate_model

  def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
    
    query_embeddings = self.query_model({
      feature_name: features[feature_name] for feature_name in self.query_feature_names
    })
    candidate_embeddings = self.candidate_model({
      feature_name: features[feature_name] for feature_name in self.candidate_feature_names
    })

    retrieval_loss = 0.
    for idx, retrieval_model in enumerate(self.retrieval_models):
      query_reduced = retrieval_model.query_reduction.query_reduction(query_embeddings)
      candidate_reduced = retrieval_model.candidate_reduction.query_reduction(candidate_embeddings)
      cosine_score, query_norm, candidate_norm = retrieval_model.cosine_score(
        query_reduced, candidate_reduced, pairwise=False
      )
      labels = tf.eye(tf.shape(cosine_score)[0])
      cosine_loss = retrieval_model.loss_(y_true=labels, y_pred=cosine_score)
      gravitation_loss = tf.math.reduce_sum(query_norm) + tf.math.reduce_sum(candidate_norm)
      retrieval_loss += cosine_loss * self.retrieval_weights[idx] +\
                        gravitation_loss * retrieval_model.gravitation +\
                        sum(retrieval_model.losses) * retrieval_model.regularization
        
    ranking_loss = 0.
    for ranking_model in self.ranking_models:
      labels = features.pop(ranking_model.label_name)
      scores = ranking_model.dcn(tf.concat([query_embeddings, candidate_embeddings], axis=1))
      ranking_loss += ranking_model.loss_(y_true=labels, y_pred=scores) * self.ranking_weights[idx] +\
                      sum(ranking_model.losses) * ranking_model.regularization

    return retrieval_loss, ranking_loss

  def train_step(self, features: Dict[str, tf.Tensor]):
        
    with tf.GradientTape() as tape:
      retrieval_loss, ranking_loss = self.compute_loss(features, training=True)
      total_loss = retrieval_loss + ranking_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {}
    metrics['retrieval_loss'] = retrieval_loss
    metrics['ranking_loss'] = ranking_loss
    metrics['total_loss'] = total_loss
    
    return metrics

  def test_step(self, features: Dict[str, tf.Tensor]):

    retrieval_loss, ranking_loss = self.compute_loss(features, training=True)
    total_loss = retrieval_loss + ranking_loss

    metrics = {}
    metrics['retrieval_loss'] = retrieval_loss
    metrics['ranking_loss'] = ranking_loss
    metrics['total_loss'] = total_loss

    return metrics