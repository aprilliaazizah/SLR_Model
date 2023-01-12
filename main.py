import sklearn
import joblib
from keras.models import load_model
from flask import Flask, render_template, request

import numpy as np
import string
import pickle

import json
import nltk
import pandas as pd

from nltk.corpus import stopwords
", ".join(stopwords.words('english'))

model, X_train, y_train = joblib.load('model.h5')
scaler = joblib.load('tfidf.joblib')

app = Flask(__name__)

def to_lower_case(texts):
	texts["title_abstract"] = [entry.lower() for entry in texts["title_abstract"]]
	return texts

def remove_punctuation(texts):
    """custom function to remove the punctuation"""
    PUNCT_TO_REMOVE = string.punctuation
    return texts.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def remove_stopwords(texts):
    """custom function to remove the stopwords"""
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(texts).split() if word not in STOPWORDS])

def preprocess_text(texts):
    texts = to_lower_case(texts)
    texts = texts["title_abstract"].apply(lambda texts: remove_punctuation(texts))
    return texts


@app.route('/predict',methods=["GET", "POST"])

def predict():
    text = request.args.get('text')
    data_teks = {'title_abstract': text}
    df = (pd.DataFrame(data_teks, index=[0]))
    x = preprocess_text(df)


    sentences_Tfidf = scaler.transform(x)
    predictions = model.predict(sentences_Tfidf)
    print(predictions)

    if predictions > 0.5:
        value = "include"
    else:
        value = "exclude"
    return value

if __name__ == "__main__":
    # Run locally
    app.run(debug=False)
