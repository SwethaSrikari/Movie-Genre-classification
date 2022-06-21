import os
import string
import contractions
import typing
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from pathlib import Path

def load_dataset(data_dir: str) -> pd.DataFrame: 
	"""
	Loads data from a csv file and returns a pandas DataFrame

	:params data_dir: Path to the data files
	"""

	# Load data
	# csv file has id, movie, genre and summary columns seperated by :::
	data = pd.read_csv(data_dir + 'train_data.txt', sep = ' ::: ', names = ['id', 'movie', 'genre', 'summary'], engine='python')

	return data

def clean_dataset(data_dir: str, model_type: str) -> pd.DataFrame:
	"""
	This function pre-processes text data
	1. Expands word contractions
	2. Handles special characters
	3. Removes stopwords
	4. Lemmatizes words ### TODO
	5. Tokenizes words ### TODO
	6. Maps class names (string) to an integer

	:params data_dir: Data directory
	:params model_type: sklearn model or transformer
	"""
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
		# Removes non-ascii characters
		inp_sentence = inp_sentence.encode('utf-8').decode('ascii', 'ignore')

		return inp_sentence

	# 3. Removes stopwords
	def remove_stopwords(inp_sentence: str) -> str:
		"""
		Removes stop words
		:param inp_sentence: string of words (sentence) to be processed
		"""
		words = [word for word in inp_sentence.split() if word not in stopwords.words('english')]
		inp_sentence = " ".join(words)

		return inp_sentence

	# 4. Lemmatizes words ### TODO
	def lemmatize_sent(inp_sentence: str) -> str:
		"""
		https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/

		Lemmatizes words in the inp_sentence
		:param inp_sentence: string of words (sentence) to be processed
		"""

		nlp = spacy.load('en_core_web_sm')

		sent = nlp(inp_sentence)

		lemmatized_sentence = " ".join([token.lemma_ for token in sent])

		return lemmatized_sentence

	# 5. Tokenizes words ### TODO
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

	data = load_dataset(data_dir)
	# Expands contractions
	data['summary'] = data["summary"].apply(expand_contractions)
	# Handle special characters
	data["summary"] = data["summary"].apply(handle_special_chars)
	# Remove punctuation
	data['summary'] = data['summary'].str.replace('[{}]'.format(string.punctuation), '', regex=True)

	# 6. Map string classes to an integer
	classes = np.unique(data['genre'])
	label_map = {l:i for i,l in enumerate(classes)}
	data = data.replace({'genre':label_map})

	return data
