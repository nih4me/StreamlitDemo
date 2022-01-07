import os
import streamlit as st

import pickle # Serialiser des objets (y comporis des modeles)

from pathlib import Path

import nltk
from nltk import tag
from nltk.stem import PorterStemmer

import gdown

ASSETS_PATH = Path("assets")
VECTORIZERS_PATH = ASSETS_PATH / "vectorizers"
MODELS_PATH = ASSETS_PATH / "models"

VECTORIZERS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

VECTORIZER_URL = "https://drive.google.com/uc?id=1aqcjGPbEBW9hFh8W3XmW5F6zbK6nEcOI"
MODEL_URL = "https://drive.google.com/uc?id=1QPxJrMlhCmo0_VxCr6T1roEcmNmMXbgK"

# Telechargements
nltk.download('stopwords')

st.cache(persist=True)
def getVectorizer(url):
    target = str(VECTORIZERS_PATH / "tfidf_vectorizer.pkl")
    if not os.path.exists(target):
        gdown.download(
            url,
            target)
    # Restaurer le TFIDFVectorizer
    vectorizer = pickle.load(open(target, mode='rb'))
    return vectorizer

st.cache(persist=True)
def getModel(url):
    target = str(MODELS_PATH / "model.pk")
    if not os.path.exists(target):
        gdown.download(
            url,
            target)
    # Restaurer le modèle
    model = pickle.load(open(target, mode='rb'))
    return model

vectorizer = getVectorizer(VECTORIZER_URL)
model = getModel(MODEL_URL)

def predict_hate(message):
    st = PorterStemmer()

    message = ' '.join([st.stem(t) for t in message.split()])
    message_features = vectorizer.transform([message]).todense()

    prediction = model.predict(message_features)[0]
    proba_neg, proba_pos = model.predict_proba(message_features)[0]
    return prediction, proba_neg, proba_pos

# Streamlit 
st.title('Demo Streamlit')
st.subheader('Prediction de discours haineux')
st.write('**Déploiement d\'un classifier pour prédire si un discours contient des propos haineux ou pas.**')

message = st.text_area('Discours à analyser', '')

if st.button('Prédire'):
    pred, proba_neg, proba_pos = predict_hate(message)
    data = dict(status='OK', message=message, haineux=str(pred), proba_neg=proba_neg, proba_pos=proba_pos)
    st.json(data)
