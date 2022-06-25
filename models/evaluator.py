import typing
import tensorflow as tf

from pathlib import Path

class Evaluator(object):
	"""
	Class to evaluate transformer models

	:param model: Model to evaluate
	:param val_data: Validation dataset
	:param valid_loss: Loss function to evaluate model
	:param valid_metric: Validation metrics
	"""
	def __init__(self, model, val_data: tf.data.Dataset, valid_loss: tf.keras.losses.Loss,
				 valid_metric: tf.keras.metrics.Metric):

		self.model = model
		self.val_data = val_data
		self.valid_loss = valid_loss
		self.valid_metric = valid_metric


	@tf.function
	def valid_step(self, val_inputs: tf.Tensor, val_targets: tf.Tensor):
		"""
		Performs validation on validation inputs and targets

		:param val_inputs: Validation inputs
		:param val_targets: Validation targets
		"""

		raise NotImplementedError("Subclass the Evaluator class and implement valid step for your model")


	def evaluate(self, epoch: int = 1):
		"""
		Evaluate model on validation set and returns validation metric

		:param epoch: A single epoch
		"""
		for inputs, targets in self.val_data: # Check why proper valid data is not passed
			self.valid_step(inputs, targets)

		# Validation logs
		valid_metric_value = self.valid_metric.result()
		valid_loss_value = self.valid_loss

		return valid_loss_value
