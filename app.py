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
from flask import Flask, request, render_template

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

app = Flask(__name__)
@app.route("/",methods=["GET","POST"])
def index():
   if request.method == "POST":
      text = request.form["textToClasify"]
      processText = procesamiento_texto(text)
      X = vectorizador_cargado.transform([processText])
      pred = modelo_cargado.predict(X)
      return render_template("result.html", prediction=pred, textToPrint = text)
   return render_template("index.html")

if __name__ == "__main__":
   app.run(debug=True)
