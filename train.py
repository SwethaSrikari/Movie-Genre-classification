import typing
import argparse
import numpy as np

from utils.preprocessing import clean_dataset

def train(data_dir: str, debug: bool=False):
	"""
	Script to train models for text classification

	:params data_dir: Directory where data is located
	:params debug: uses less data for faster debugging
	"""

	print('inside train')
	data = clean_dataset(data_dir, '')

	if debug:
		print('Debugging')
		data = data[:100]

	print('data size', len(data))







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
	parser.add_argument('--debug', type=bool, default=False, help='Uses less data to help debug quickly')
	args = parser.parse_args()
	train(args.data_dir, args.debug)