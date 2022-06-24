import os
import string
import contractions
import typing
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
nltk.download('omw-1.4')

from pathlib import Path

def load_dataset(data_dir: str, seed: int, debug: bool = False) -> pd.DataFrame: 
	"""
	Loads data from a csv file and returns a pandas DataFrame

	:params data_dir: Path to the data files
	"""

	# Load data
	# csv file has id, movie, genre and summary columns seperated by :::
	data = pd.read_csv(data_dir + 'train_data.txt', sep = ' ::: ', names = ['id', 'movie', 'genre', 'summary'], engine='python')


	# For now, let's deal with balanced classification problem

	# Keeps only the examples of genre with at least 1000 examples 
	# and the examples of genre with less than 1000 examples are dropped
	morethan1000 = data.groupby('genre').filter(lambda x: len(x) >=1000)

	if debug:
		print('Debugging ------')
		data = morethan1000.groupby('genre').apply(lambda x: x.sample(n=10, random_state=seed)).reset_index(drop = True)
	else:
		# Since the genre distribution (number of examples in each genre is different), we sample 1000 examples from each genre for training
		data = morethan1000.groupby('genre').apply(lambda x: x.sample(n=1000, random_state=seed)).reset_index(drop = True)

	# Map string classes to an integer
	classes = np.unique(data['genre'])
	label_map = {l:i for i,l in enumerate(classes)}
	data = data.replace({'genre':label_map})
	num_classes = len(classes)

	return data

# 1. Expands word contractions
def expand_contractions(inp_sentence: str) -> str:
	"""
	Expands all contracted words and returns the 'expanded' sentence

	Returns sentences after expanding contracted words

	:param inp_sentence: string of words (sentence) to be processed
	"""


	# creating an empty list
	expanded_words = []   
	for word in inp_sentence.split():
		# using contractions.fix to expand the shortened words
		expanded_words.append(contractions.fix(word))

	return " ".join(expanded_words)

# 2. Handles special characters
def handle_special_chars(inp_sentence: str) -> str:
	"""
	Handles non-ascii characters, hyphens
	:param inp_sentence: string of words (sentence) to be processed
	"""
	# Replaces '-' with an empty space to reduce extra vocabulary
	inp_sentence = inp_sentence.replace('-', ' ')
	# Replaces '/' with an empty space to reduce extra vocabulary
	inp_sentence = inp_sentence.replace('/', ' ')
	# Removes non-ascii characters
	inp_sentence = inp_sentence.encode('utf-8').decode('ascii', 'ignore')

	return inp_sentence

# 3. Removes stopwords #TODO
def remove_stopwords(inp_sentence: str) -> str:
	"""
	Removes stop words
	:param inp_sentence: string of words (sentence) to be processed
	"""
	words = [word for word in inp_sentence.split() if word not in stopwords.words('english')]
	inp_sentence = " ".join(words)

	return inp_sentence

# 4. Lemmatizes words
lemmatizer = WordNetLemmatizer()
def lemmatize_sent(inp_sentence: str) -> str:
	"""
	https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/

	Lemmatizes words in the inp_sentence
	:param inp_sentence: string of words (sentence) to be processed
	"""

	lemmatized_sentence = " ".join([lemmatizer.lemmatize(w) for w in inp_sentence.split()])

	return lemmatized_sentence

# 5. Tokenizes words
def tokenizer_of_input(dataframe_col: pd.core.series.Series):
	"""
	Tokenizing and vectorizing the text for the model to understand
	Converts to lowercase
	returns a tokenizer
	:param dataframe_col: A dataframe column
	"""
	tokenizer = TextVectorization(standardize='lower', output_mode='int', ragged=True) 
	tokenizer.adapt(dataframe_col.tolist())

	return tokenizer


def clean_dataset_sklearn(data_dir: str, seed: int, debug: bool = False) -> pd.DataFrame:
	"""
	This function pre-processes text data
	1. Expands word contractions
	2. Handles special characters
	3. Removes stopwords
	4. Lemmatizes words
	5. Tokenizes words ### TODO
	6. Maps class names (string) to an integer

	Returns X - predictors and y - labels

	:params data_dir: Data directory
	:params model_type: sklearn model or tensorflow sequential model or pre-trained transformer
	:params debug: Debugging for faster testing
	"""
	
	data = load_dataset(data_dir, seed, debug)


	# Expands contractions
	data['summary'] = data["summary"].apply(expand_contractions)
	# Handle special characters
	data["summary"] = data["summary"].apply(handle_special_chars)
	# Remove punctuation
	data['summary'] = data['summary'].str.replace('[{}]'.format(string.punctuation), '', regex=True)
	# Lemmatize
	data['summary'] = data['summary'].apply(lemmatize_sent)

	X = data['summary']
	y = data['genre']
	

	return X, y

def clean_dataset_sequential(data_dir: str, seed: int, batch_size: int, debug: bool = False):

	data = load_dataset(data_dir, seed, debug)
	num_classes = len(np.unique(data['genre']))
	
	# Expands contractions
	data['summary'] = data["summary"].apply(expand_contractions)
	# Handle special characters
	data["summary"] = data["summary"].apply(handle_special_chars)
	# Remove punctuation
	data['summary'] = data['summary'].str.replace('[{}]'.format(string.punctuation), '', regex=True)
	# Lemmatize
	data['summary'] = data['summary'].apply(lemmatize_sent)
	# Tokenize
	tokenizer = tokenizer_of_input(data['summary'])
	data['summary'] = data['summary'].apply(tokenizer)

	X = data['summary'].tolist()
	# One-hot encoding of classes
	y = tf.keras.utils.to_categorical(data["genre"].values, num_classes=num_classes)
	y = [tf.constant(l) for l in y] # Converting labels to tensor

	# Creating training and validation sets
	x_train, x_val, y_train, y_val = train_test_split(X, y, shuffle=True,
	                                                  test_size=0.2, random_state=271)

	# Creating tf dataset from generator (for variable length inputs 'from_tensor_slices' doesn't work)
	# train dataset
	train_dataset = tf.data.Dataset.from_generator(lambda: ((x, y) for (x,y) in zip(x_train, y_train)),
	                                               output_types=(tf.as_dtype(x_train[0].dtype), tf.as_dtype(y_train[0].dtype)),
	                                               output_shapes=([None,], [None,]))
	# Valid dataset
	val_dataset = tf.data.Dataset.from_generator(lambda: ((x, y) for (x,y) in zip(x_val, y_val)),
	                                             output_types=(tf.as_dtype(x_val[0].dtype), tf.as_dtype(y_val[0].dtype)),
	                                             output_shapes=([None,], [None,]))

	# Dynamic padding pads the sequences to the maximum sequence length of each batch
	# Prepare the training dataset.
	# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	train_dataset = train_dataset.shuffle(buffer_size=1024).padded_batch(batch_size)

	# Prepare the validation dataset.
	# val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
	val_dataset = val_dataset.padded_batch(batch_size)

	AUTOTUNE = tf.data.AUTOTUNE
	train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
	val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

	return train_dataset, val_dataset

def clean_dataset_transformer(data_dir: str, seed: int, batch_size: int, debug: bool = False):

	data = load_dataset(data_dir, seed, debug)
	num_classes = len(np.unique(data['genre']))
	# Raw data is fed to fine-tune pre-trained transformer models so pre-processing is omitted
	X = data['summary']
	y = tf.keras.utils.to_categorical(data["genre"].values, num_classes=num_classes)

	x_train, x_val, y_train, y_val = train_test_split(X, y, shuffle=True,
                                                  test_size=0.2,
                                                  stratify=y,
                                                  random_state=271)

	training_dataset = (tf.data.Dataset.from_tensor_slices((tf.cast(x_train.values, tf.string), 
															tf.cast(y_train, tf.int32)))).batch(batch_size)

	validation_dataset = (tf.data.Dataset.from_tensor_slices((tf.cast(x_val.values, tf.string), 
															  tf.cast(y_val, tf.int32)))).batch(batch_size)

	AUTOTUNE = tf.data.AUTOTUNE
	training_dataset = training_dataset.cache().prefetch(buffer_size=AUTOTUNE)
	validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

	return training_dataset, validation_dataset

