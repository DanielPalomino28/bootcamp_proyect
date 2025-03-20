import pickle
import joblib
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Descargar recursos de NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Preprocesamiento de texto
def procesamiento_texto(texto):
    if pd.isna(texto):
        return ""
    texto = re.sub(r'[^a-zA-Z\s]', '', texto.lower())
    words = word_tokenize(texto)
    words = [word for word in words if word not in stopwords.words('spanish')]
    lemmatizador = WordNetLemmatizer()
    words = [lemmatizador.lemmatize(word) for word in words]
    return ' '.join(words)

# Cargar modelo y vectorizador
with open("spam_model.pkl", "rb") as f:
  modelo_cargado = pickle.load(f)

with open("spam_vectorizer.pkl", "rb") as f:
  vectorizador_cargado = pickle.load(f)

# Procesar y predecir sobre una nueva entrada
nuevo_texto = "No matter how far you get" 
texto_procesado = procesamiento_texto(nuevo_texto)
X = vectorizador_cargado.transform([texto_procesado])
prediction = modelo_cargado.predict(X)

print("Predicción:", prediction)
