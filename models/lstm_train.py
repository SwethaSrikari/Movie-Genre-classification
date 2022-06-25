import typing
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Embedding

from models.modeltrainer import ModelTrainer
from models.lstm import Lstm

def lstm_train(vocab_size: int, num_classes: int, train_dataset, val_dataset):
	"""
	Builds a trainer to train a LSTM model

	:params vocab_size: Vocabulary size
	:params num_classes: Number of classes
	:params train_dataset: Train dataset
	:params val_dataset: Validation dataset
	"""
	# Hyperparameters
	embedding_dim = 100
	lstm_units = 128
	dropout = 0.5

	# Embedding
	# if embedding == 'none':
	embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
	# elif embedding == 'glove100':
	# 	print(f'Using {embedding} embeddings')
	# 	embedding_matrix = create_embedding_matrix(embedding_dir=embedding_dir, embedding_dim=100, tokenizer=tokenizer)
	# 	embedding = Embedding(vocab_size,
	# 						  100,
	# 						  embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
	# 						  trainable=False,
	# 						  mask_zero=True)

	# Instiantiate model
	model = Lstm(num_classes=num_classes, embedding=embedding,
				 lstm_units=lstm_units, dropout=dropout)

	# Optimizer
	optimizer = tf.keras.optimizers.Adam()

	# Loss functions and metrics
	train_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
	valid_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
	train_metric = tf.keras.metrics.CategoricalAccuracy()
	valid_metric = tf.keras.metrics.CategoricalAccuracy()

	# Trainer
	trainer = ModelTrainer(model=model, train_data=train_dataset,
						   val_data=val_dataset,optimizer=optimizer,
						   train_loss=train_loss, valid_loss=valid_loss,
						   train_metric=train_metric, valid_metric=valid_metric)

	return trainer