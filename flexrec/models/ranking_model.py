'''Ranking of candidates for a given query'''

import tensorflow as tf
import tensorflow_recommenders as tfrs

from typing import Dict, Iterable, List, Optional, Tuple, Union


class RankingModel(tf.keras.Model):

  def __init__(self, 
               query_model: tf.keras.Model, 
               candidate_model: tf.keras.Model,
               layer_sizes: Optional[Iterable[int]]=None,
               use_interaction: bool=True, 
               projection_dim: int=None,
               regularization: float=1e-3,
               label_name: str='label'):
    
    super().__init__()
    
    self.query_feature_names = query_model.feature_names
    self.query_model = query_model

    self.candidate_feature_names = candidate_model.feature_names
    self.candidate_model = candidate_model
    
    self.label_name = label_name

    self.dcn = tf.keras.Sequential()
    self.dcn.add(tf.keras.layers.Flatten())
    if use_interaction:
      self.dcn.add(
        tfrs.layers.dcn.Cross(
          projection_dim=projection_dim, # needs to be (input size)/2 ~ (input size)/4
          kernel_initializer="glorot_uniform",
          kernel_regularizer=tf.keras.regularizers.L2()
        )
      )
    if layer_sizes is not None:
      for layer_size in layer_sizes:
        self.dcn.add(
          tf.keras.layers.Dense(
            layer_size, activation="relu", kernel_regularizer=tf.keras.regularizers.L2()
          )
        )
    self.dcn.add(
      tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.L2()
      )
    )
    
    self.regularization = regularization

    self.loss_ = tf.keras.losses.MeanSquaredError(
      reduction=tf.keras.losses.Reduction.SUM
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    
    query_embeddings = self.query_model({
      feature_name: features[feature_name] for feature_name in self.query_feature_names
    })
    candidate_embeddings = self.candidate_model({
      feature_name: features[feature_name] for feature_name in self.candidate_feature_names
    })
    
    return self.dcn(tf.concat([query_embeddings, candidate_embeddings], axis=1))

  def compute_loss(self, features: Dict[str, tf.Tensor], training=False):
    labels = features.pop(self.label_name)
    scores = self(features)
    return self.loss_(y_true=labels, y_pred=scores)

  def train_step(self, features: Dict[str, tf.Tensor]):

    with tf.GradientTape() as tape:
      loss = self.compute_loss(features, training=True)
      regularization_loss = sum(self.losses)
      total_loss = loss + self.regularization * regularization_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = self.regularization * regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

  def test_step(self, features: Dict[str, tf.Tensor]):

    loss = self.compute_loss(features, training=False)
    regularization_loss = sum(self.losses)
    total_loss = loss + self.regularization * regularization_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = self.regularization * regularization_loss
    metrics["total_loss"] = total_loss

    return metrics