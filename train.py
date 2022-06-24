print('good')
import typing
import argparse
import numpy as np

from sklearn.model_selection import train_test_split

from utils.preprocessing import clean_dataset_sklearn, clean_dataset_sequential, clean_dataset_transformer
from models.sl_models import pipeline
print('good')
def train(data_dir, batch_size, seed, scaler_type, transform_type, model_name, model_type, debug):
	"""
	Script to train models for text classification

	:params data_dir: Directory where data is located
	:params scaler_type: type of sklearn scaler
	:params transform_type: type of vectorization
	:params model_name: classification model
	:params debug: uses less data for faster debugging
	"""

	# Check model_type and models match or not
	if model_name in ['MultinomialNB', 'LogisticRegression', 'SGDClassifier', 'ComplementNB',
					  'GaussianNB', 'LinearSVC', 'DecisionTreeClassifier', 'RandomForestClassifier',
					  'KNeighborsClassifier', 'DummyClassifier'] and model_type != 'sklearn':
		raise ValueError('Model is not a sklearn model')

	elif model_name in ['LSTM'] and model_type != 'sequential':
		raise ValueError('Model is not a sequential model')
	elif model_name in ['BERTsmall'] and model_type != 'pre-trained transformer':
		raise ValueError('Model is not a transformer model')


	if model_type == 'sklearn':
		X, y = clean_dataset_sklearn(data_dir, seed, debug)

		print('data size', len(X))

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
		train_dataset, val_dataset = clean_dataset_sequential(data_dir, seed, batch_size, debug)
		for x,y in train_dataset.take(1):
			print(x.shape)
			print(y.shape)

	elif model_type == 'pre-trained transformer':
		train_dataset, val_dataset = clean_dataset_transformer(data_dir, seed, batch_size, debug)
		for x,y in train_dataset.take(1):
			print(x.shape)
			print(y.shape)








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
	# parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for')
	parser.add_argument('--seed', type=int, default=46, help='random state seed')
	parser.add_argument('--scaler_type', type=str, default='minmax', help='Scaler to use')
	parser.add_argument('--transform_type', type=str, default='count-tfidf', help='Type of vectorization')
	parser.add_argument('--model_name', type=str, default='DummyClassifier', help='Model to use',
						choices=['MultinomialNB', 'LogisticRegression', 'SGDClassifier', 'ComplementNB',
						'GaussianNB', 'LinearSVC', 'DecisionTreeClassifier', 'RandomForestClassifier',
						'KNeighborsClassifier', 'DummyClassifier', 'LSTM', 'BERTsmall'])
	parser.add_argument('--model_type', type=str, default='sklearn', help='Type of model - sklearn, sequential or pre-trained transformer',
						choices=['sklearn', 'sequential', 'pre-trained transformer'])
	parser.add_argument('--debug', type=bool, default=False, help='Uses less data to help debug quickly')
	args = parser.parse_args()
	train(args.data_dir, args.batch_size, args.seed, args.scaler_type, args.transform_type, args.model_name, args.model_type, args.debug)