from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def models(model_name: str):
	"""
	Returns sklearn model
	:params model_name: Model name
	"""

	if model_name == 'MultinomialNB':
		return ('clf', MultinomialNB())

	elif model_name == 'LogisticRegression':
		return ('clf', LogisticRegression(max_iter=1000))

	elif model_name == 'SGDClassifier':
		return ('clf', SGDClassifier())

	elif model_name == 'ComplementNB':
		return ('clf', ComplementNB())

	elif model_name == 'GaussianNB':
		return ('clf', GaussianNB())

	elif model_name == 'LinearSVC':
		return ('clf', LinearSVC())

	elif model_name == 'DecisionTreeClassifier':
		return ('clf', DecisionTreeClassifier())

	elif model_name == 'RandomForestClassifier':
		return ('clf', RandomForestClassifier())

	elif model_name == 'KNeighborsClassifier':
		return ('clf', KNeighborsClassifier())

	elif model_name == 'DummyClassifier':
		return ('clf', DummyClassifier())

	else:
		raise ValueError("Model not found, check with model choices")

def transformation(transform_type: str = 'count-tfidf'):
	"""
	Returns a sklearn Transformer
	:params transform_type : To use embedding vectors or word-count/tfidf (default = 'count-tfidf')
	"""

	if transform_type == 'count-tfidf':
		return ('vect', CountVectorizer(lowercase=True, stop_words='english')), ('tfidf', TfidfTransformer())

def scalers(scaler_type: str = 'minmax'):
	"""
	Returns a sklearn scaler
	:params scaler_type: type of scaler to use (default = 'minmax')
	"""

	if scaler_type == 'minmax':
		return ('scaler', MinMaxScaler())

def pipeline(scaler_type: str, transform_type: str, model_name: str):
	"""
	Returns a  sklearn pipeline for training
	"""

	scaler = scalers(scaler_type)
	vectorizer, transformer = transformation(transform_type)
	model = models(model_name)

	pipeline = Pipeline([vectorizer,
						 transformer,
						 model])

	return pipeline




