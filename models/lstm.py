import typing
import tensorflow as tf

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Activation
from tensorflow.keras.models import Sequential

class Lstm(tf.keras.Model):
	"""
	The model is sequential with an embedding layer (input), LSTM layer, Dropout and a Dense layer (fully connected output layer)

	:param num_classes: Number of classes
	:param embedding: embedding layer
	:param lstm_units: lstm dimension
	:param dropout: fraction of inputs to drop
	"""

	def __init__(self, num_classes: int, embedding, lstm_units: int, dropout: float):
		super(Lstm, self).__init__()
		self.model = Sequential()
		self.embedding_layer = embedding
		self.lstm_layer = LSTM(lstm_units)
		self.dropout_layer = Dropout(dropout)
		self.output_layer = Dense(num_classes)
		

	def call(self, inputs: tf.data.Dataset, training=False):
		"""
		Adds embedding, lstm, dropout and dense layer to the sequential model

		Returns the model
		"""
		self.model.add(self.embedding_layer)
		self.model.add(self.lstm_layer)
		self.model.add(Activation('relu'))
		self.model.add(self.dropout_layer)
		self.model.add(self.output_layer)
		self.model.add(Activation('softmax'))

		return self.model(inputs, training=training)