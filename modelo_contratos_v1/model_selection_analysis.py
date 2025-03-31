import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import joblib
import os
import matplotlib.pyplot as plt

# 1. Cargar el dataset
file_path = r"C:\Users\danie\Documents\Bootcamp\data_prepared_renamed.csv"
df = pd.read_csv(file_path, low_memory=False)
print("✅ Archivo cargado correctamente.")

# 2. Definir las columnas a descartar (no viables para la entrada del usuario o poco informativas)
# Se descartan las columnas de tiempo y de identificación que no aportan información útil para el usuario
cols_to_drop = ['Código_Directorio', 'Registro_Encuesta', 'PERIODO', 'MES', 'PER', 'FEX_C18', 'P3044S2', 'P6420S2']
df = df.drop(columns=cols_to_drop, errors='ignore')
print(f"🔹 Se han descartado las siguientes columnas: {cols_to_drop}")

# 3. Preprocesamiento básico: Para este análisis, convertiremos las columnas que sean objeto a valores numéricos
# (Si ya están codificadas, se mantiene la codificación; de lo contrario, se aplica factorize)
for col in df.columns:
    if df[col].dtype == 'object':
        # Reemplazar valores '.' por NaN y luego factorizar la columna
        df[col] = df[col].replace('.', np.nan)
        df[col] = pd.factorize(df[col])[0]

# 4. Analicemos dos posibles objetivos:
# a) Modelo de regresión con P6500 (Ingreso_Mensual)
# b) Modelo de clasificación con P6450 (Tipo_Contrato)

# --- a) Análisis con P6500 (Ingreso_Mensual) como objetivo (modelo numérico)
target_reg = 'Ingreso_Mensual'
if target_reg not in df.columns:
    raise ValueError(f"La variable objetivo {target_reg} no se encuentra en el DataFrame.")

X_reg = df.drop(columns=[target_reg])
y_reg = df[target_reg]

# Dividir datos para regresión (80/20)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Entrenar un árbol de regresión para obtener importancia de variables
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = regressor.predict(X_test_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"\nModelo de regresión con '{target_reg}' - R2 score: {r2:.2f}")

# Importancia de variables para la regresión
imp_reg = pd.DataFrame({
    'feature': X_reg.columns,
    'importance': regressor.feature_importances_
}).sort_values(by='importance', ascending=False)
print("\nImportancia de variables (Regresión - Ingreso_Mensual):")
print(imp_reg.head(10))


# --- b) Análisis con P6450 (Tipo_Contrato) como objetivo (modelo categórico)
target_clf = 'Tipo_Contrato'
if target_clf not in df.columns:
    raise ValueError(f"La variable objetivo {target_clf} no se encuentra en el DataFrame.")

X_clf = df.drop(columns=[target_clf])
y_clf = df[target_clf]

# Dividir datos para clasificación (80/20)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Entrenar un árbol de clasificación para obtener importancia de variables
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train_clf, y_train_clf)
y_pred_clf = classifier.predict(X_test_clf)
acc = accuracy_score(y_test_clf, y_pred_clf)
print(f"\nModelo de clasificación con '{target_clf}' - Precisión: {acc:.2f}")

# Importancia de variables para la clasificación
imp_clf = pd.DataFrame({
    'feature': X_clf.columns,
    'importance': classifier.feature_importances_
}).sort_values(by='importance', ascending=False)
print("\nImportancia de variables (Clasificación - Tipo_Contrato):")
print(imp_clf.head(10))

# 5. Seleccionar las variables viables para la entrada en la app web
# Por ejemplo, descartamos variables de identificación y aquellas que el análisis indique tienen menor importancia.
# Supongamos que para la app web, queremos que el usuario ingrese información referente a:
# - Medio_Empleo_Obtenido
# - Misma_Empresa
# - Tipo_Empresa_Contratante
# - Empresa_Registrada
# - Cuenta_Contador
# Estos campos tienen un significado intuitivo para el usuario y son relevantes según el análisis.

selected_features = ['Medio_Empleo_Obtenido', 'Misma_Empresa', 'Tipo_Empresa_Contratante', 
                     'Empresa_Registrada', 'Cuenta_Contador']
print("\nPara la app web se recomienda solicitar como entrada los siguientes campos:")
print(selected_features)

# Opcional: Graficar la importancia de variables para el modelo de clasificación
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.barh(imp_clf['feature'].head(10)[::-1], imp_clf['importance'].head(10)[::-1], color='skyblue')
plt.xlabel("Importancia")
plt.title("Top 10 Variables - Modelo Clasificación (Tipo_Contrato)")
plt.show()

""" Nota: Este script sirve para argumentar la elección de las variables a usar en la app web.
# Según el análisis, para predecir el tipo de contrato de un usuario, se podrían utilizar solo las variables que el usuario pueda ingresar (sin incluir identificadores o códigos internos).
# De esta manera, se simplifica la interfaz y se evita la necesidad de ingresar información que no aporta a la predicción."""