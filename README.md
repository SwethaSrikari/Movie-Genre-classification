# Text-classification
Multi class text classification using scikit-learn models, sequential models, pre-trained transformer models

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

If this sounds tedious, you can play with the [colab notebook] (https://colab.research.google.com/drive/1ipyhpEdEbV1tzU5Z2oinmbAU1upPS0Qb#scrollTo=5-VTHCr7wE6P) that already has various scikit-learn models, sequential models and pre-trained models implemented.