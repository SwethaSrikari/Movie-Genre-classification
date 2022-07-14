# Text-classification
Multi class text classification using different word embeddings, scikit-learn models, sequential models, pre-trained transformer models

**NOTE: For now, the goal is to explore different types of models that can be used for text classification and see how each one performs. I'll add more models as I keep exploring new ones**

# Dataset
Genre classfication dataset with the movie name, summary of the movie and its genre seperated by ':::' is saved in a text file which can be downloaded from [here](https://www.kaggle.com/code/rohitganji13/film-genre-classification-using-nlp/data).

For classification, summaries of the movies are used as features and genres as the targets.

# About the dataset
Run this [notebook](https://colab.research.google.com/drive/1xlkqsWmmkw7ruoazaEWiox_Od-8MMGff#scrollTo=leGprgl9aN9s) to know more about the data and about the correlation between certain words and a genre.

# Requirements
To run the `train.py` script and train different models, install the packages listed in the `requiremnets.txt`.

# Training
To train models, run `train.py` script within `Text-classification` folder with appropriate arguments.

```
$ python train.py --data_dir <path to dataset> --batch_size <batch_size> --epochs <number of epochs to train for> --seed <random state seed> --scaler_type <scaler to use> --transform_type <type of tokenizer> --model_name <model to use for training> --debug <for debugging>
```

If this sounds tedious, you can play with this [notebook](https://colab.research.google.com/drive/1ipyhpEdEbV1tzU5Z2oinmbAU1upPS0Qb#scrollTo=5-VTHCr7wE6P) that already has various scikit-learn models, sequential models and pre-trained models implemented.

**NOTE : tensorflow-text is not compatible with conda. It has to be fixed to be able to run `BERT` model. Until then, use this [colab notebook](https://colab.research.google.com/drive/1ipyhpEdEbV1tzU5Z2oinmbAU1upPS0Qb#scrollTo=5-VTHCr7wE6P) to explore different `BERT` models.**

# References
Classify text with BERT - https://www.tensorflow.org/text/tutorials/classify_text_with_bert

Scikit-learn - https://scikit-learn.org/stable/

Expanding word contractions - https://www.geeksforgeeks.org/nlp-expand-contractions-in-text-processing/

Tensorflow's ragged tensor to handle inputs of different length - https://www.tensorflow.org/guide/ragged_tensor

Tensorflow dataset - https://www.tensorflow.org/guide/data

Tensorflow's text generation - https://www.tensorflow.org/text/tutorials/text_generation

Tensorflow training - https://www.tensorflow.org/guide/keras/train_and_evaluate

Writing training loop from scratch - https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

Saving and loading models - https://www.tensorflow.org/tutorials/keras/save_and_load

Glove embeddings using spacy - https://lvngd.com/blog/spacy-word-vectors-as-features-in-scikit-learn/

Fasttext embeddings - https://stackoverflow.com/questions/65300462/write-a-fasttext-customised-transformer