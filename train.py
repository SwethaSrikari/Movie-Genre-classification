import typing
import argparse
import numpy as np

from sklearn.model_selection import train_test_split

from utils.preprocessing import clean_dataset
from models.sl_models import pipeline

def train(data_dir: str, scaler_type:str, transform_type: str, model_name: str, debug: bool=False):
	"""
	Script to train models for text classification

	:params data_dir: Directory where data is located
	:params scaler_type: type of sklearn scaler
	:params transform_type: type of vectorization
	:params model_name: classification model
	:params debug: uses less data for faster debugging
	"""

	data = clean_dataset(data_dir, '')

	if debug:
		print('Debugging')
		data = data[:10000]

	print('data size', len(data))

	# Train and validation split
	X = data['summary']
	y = data['genre']

	x_train, x_val, y_train, y_val = train_test_split(X, y, shuffle=True,
													  test_size=0.2,
													  stratify=y,
													  random_state=271)

	classifier = pipeline(scaler_type=scaler_type, transform_type=transform_type, model_name=model_name)

	classifier.fit(x_train, y_train)
	predicted = classifier.predict(x_val)
	print(f"Accuracy using {model_name} is" , np.mean(predicted == y_val) * 100, "%")








if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str,
						default='dataset/',
						help='Dataset directory')
	# parser.add_argument('--logs_dir', type=str, default='',
	# 					help='Path to training logs')
	# parser.add_argument('--embedding', type=str, default='none', help='Type of word embeddings to use', choices=['none', 'glove100'])
	# parser.add_argument('--embedding_dir', type=str, default='', help='Embeddings directory')
	# parser.add_argument('--batch_size', type=int, default=64, help='The desired batch size to use when training')
	# parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for')
	# parser.add_argument('--seed', type=int, default=271, help='random state seed')
	parser.add_argument('--scaler_type', type=str, default='minmax', help='Scaler to use')
	parser.add_argument('--transform_type', type=str, default='count-tfidf', help='Type of vectorization')
	parser.add_argument('--model_name', type=str, default='DummyClassifier', help='Model to use',
						choices=['MultinomialNB', 'LogisticRegression', 'SGDClassifier', 'ComplementNB',
						'GaussianNB', 'LinearSVC', 'DecisionTreeClassifier', 'RandomForestClassifier',
						'KNeighborsClassifier', 'DummyClassifier'])   
	parser.add_argument('--debug', type=bool, default=False, help='Uses less data to help debug quickly')
	args = parser.parse_args()
	train(args.data_dir, args.scaler_type, args.transform_type, args.model_name, args.debug)