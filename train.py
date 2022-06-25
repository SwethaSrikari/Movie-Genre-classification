import typing
import argparse
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding

from utils.preprocessing import clean_dataset_sklearn, clean_dataset_sequential, clean_dataset_transformer
from models.sl_models import pipeline
from models.lstm_train import lstm_train
from models.bert_train import bert_train

def train(data_dir: str, batch_size: int, epochs: int, seed: int, scaler_type: str, transform_type: str,
		  model_name: str, debug: bool=False):
	"""
	Script to train models for text classification

	:params data_dir: Directory where data is located
	:params batch_size: Batch size for deep learning models
	:params epochs: number of epochs to train
	:params seed: random state seed
	:params scaler_type: type of sklearn scaler
	:params transform_type: type of vectorization
	:params model_name: classification model
	:params debug: uses less data for faster debugging
	"""

	# Assign model_type for model specific data extraction and training
	if model_name in ['MultinomialNB', 'LogisticRegression', 'SGDClassifier', 'ComplementNB',
					  'GaussianNB', 'LinearSVC', 'DecisionTreeClassifier', 'RandomForestClassifier',
					  'KNeighborsClassifier', 'DummyClassifier']:
		model_type = 'sklearn'

	elif model_name in ['LSTM']:
		model_type = 'sequential'
	elif model_name in ['BERTsmall']:
		model_type = 'pre-trained transformer'
	else:
		raise ValueError(f'Unknown model {model_name}')

	if model_type == 'sklearn':
		X, y = clean_dataset_sklearn(data_dir, seed, debug)

		# Train and validation split
		x_train, x_val, y_train, y_val = train_test_split(X, y, shuffle=True,
														  test_size=0.2,
														  stratify=y,
														  random_state=271)

		
		classifier = pipeline(scaler_type=scaler_type, transform_type=transform_type, model_name=model_name)

		classifier.fit(x_train, y_train)
		predicted = classifier.predict(x_val)
		print(f"{model_name}")
		print(f"Accuracy using {model_name} is" , np.mean(predicted == y_val) * 100, "%")

	elif model_type == 'sequential':
		tokenizer, num_classes, train_dataset, val_dataset = clean_dataset_sequential(data_dir, seed, batch_size, debug)
		vocab_size = len(tokenizer.get_vocabulary())

		# Train
		trainer = lstm_train(vocab_size, num_classes, train_dataset, val_dataset)
		trainer.fit(epochs)

	elif model_type == 'pre-trained transformer':
		num_classes, train_dataset, val_dataset = clean_dataset_transformer(data_dir, seed, batch_size, debug)

		#Train
		trainer = bert_train(num_classes, train_dataset, val_dataset)
		trainer.fit(epochs)

	
	print('-------Training complete---------')









if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str,
						default='dataset/',
						help='Dataset directory')
	# parser.add_argument('--logs_dir', type=str, default='',
	# 					help='Path to training logs')
	# parser.add_argument('--embedding', type=str, default='none', help='Type of word embeddings to use', choices=['none', 'glove100'])
	# parser.add_argument('--embedding_dir', type=str, default='', help='Embeddings directory')
	parser.add_argument('--batch_size', type=int, default=32, help='The desired batch size to use when training')
	parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for')
	parser.add_argument('--seed', type=int, default=46, help='random state seed')
	parser.add_argument('--scaler_type', type=str, default='minmax', help='Scaler to use')
	parser.add_argument('--transform_type', type=str, default='count-tfidf', help='Type of vectorization')
	parser.add_argument('--model_name', type=str, default='DummyClassifier', help='Model to use',
						choices=['MultinomialNB', 'LogisticRegression', 'SGDClassifier', 'ComplementNB',
						'GaussianNB', 'LinearSVC', 'DecisionTreeClassifier', 'RandomForestClassifier',
						'KNeighborsClassifier', 'DummyClassifier', 'LSTM', 'BERTsmall'])
	parser.add_argument('--debug', type=bool, default=False, help='Uses less data to help debug quickly')
	args = parser.parse_args()
	train(args.data_dir, args.batch_size, args.epochs, args.seed, args.scaler_type, args.transform_type, args.model_name, args.debug)