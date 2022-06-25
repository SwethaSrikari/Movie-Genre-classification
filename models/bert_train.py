import typing
import argparse
import numpy as np
import tensorflow as tf

from models.modeltrainer import ModelTrainer
from models.bert import BERT

def bert_train( num_classes: int, train_dataset, val_dataset, bert_model_name: str = 'small_bert/bert_en_uncased_L-2_H-128_A-2'):
	"""
	Initializes a trainer to train a bert model

	:params num_classes: Number of classes
	:params bert_model_name: Name of the bert model
	:params train_dataset: Train dataset
	:params val_dataset: Validation dataset
	"""

	# Model
	model = BERT(num_classes, bert_model_name)

	# Optimizer
	optimizer = tf.keras.optimizers.Adam()

	# Loss function
	train_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
	valid_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

	# Metrics
	train_metric = tf.keras.metrics.CategoricalAccuracy()
	valid_metric = tf.keras.metrics.CategoricalAccuracy()

	# Trainer
	trainer = ModelTrainer(model=model, train_data=train_dataset,
						   val_data=val_dataset,optimizer=optimizer,
						   train_loss=train_loss, valid_loss=valid_loss,
						   train_metric=train_metric, valid_metric=valid_metric)

	return trainer