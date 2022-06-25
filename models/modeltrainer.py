import typing
import tensorflow as tf

from models.trainer import Trainer

class ModelTrainer(Trainer):
	"""
	Inherits Trainer and Evaluator class to implement train_step and valid_step

	:param model: Model to train
	:param train_data: Training data
	:param val_data: Validation data
	:param optimizer: Optimizer to use when training
	:param train_loss: Loss function for training
	:param valid_loss: Loss function for validation
	:param train_metric: Training metric
	:param valid_metric: Validation metric
	"""
	def __init__(self, model, train_data: tf.data.Dataset, val_data: tf.data.Dataset,
				 optimizer: tf.keras.optimizers.Optimizer,
				 train_loss: tf.keras.losses.Loss,
				 valid_loss: tf.keras.losses.Loss,
				 train_metric: tf.keras.metrics.Metric,
				 valid_metric: tf.keras.metrics.Metric):

		super().__init__(model, train_data, val_data, optimizer, train_loss, valid_loss, train_metric, valid_metric)


	# @tf.function
	def train_step(self, train_inputs: tf.Tensor, train_targets: tf.Tensor):
		"""
		Returns loss computed after training each batch
		:param train_inputs: training inputs
		:param train_targets: training labels
		"""
		# For each batch, we open a GradientTape() scope
		with tf.GradientTape() as tape:

			# Inside this scope, we call the model (forward pass) and compute the loss
			logits = self.model.call(train_inputs, training=True)

			loss_value = self.train_loss(train_targets, logits)

		# Outside the scope, we retrieve the gradients of the weights of the model with regard to the loss
		grads = tape.gradient(loss_value, self.model.trainable_weights)

		# Finally, we use the optimizer to update the weights of the model based on the gradients
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

		# Update training metric.
		self.train_metric.update_state(train_targets, logits)

		return loss_value


	# @tf.function
	def valid_step(self, valid_inputs: tf.Tensor, valid_targets: tf.Tensor):
		"""
		Uses trained model to make prediction on validation set
		:param valid_inputs: validation inputs
		:param valid_targets: validation labels
		"""
		val_logits = self.model(valid_inputs, training=False)

		# Update val metrics
		self.valid_metric.update_state(valid_targets, val_logits)
		self.valid_loss(valid_targets, val_logits)







