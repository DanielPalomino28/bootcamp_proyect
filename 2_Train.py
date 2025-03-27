import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import numpy as np

# Carga del dataset preparado
file_path = r"C:\Users\danie\Documents\Bootcamp\data_prepared.csv"
df = pd.read_csv(file_path, low_memory=False)
print("✅ Archivo cargado correctamente.")

# Eliminar columnas innecesarias
cols_to_drop = ['PERIODO', 'MES', 'PER', 'FEX_C18', 'P3044S2', 'P6420S2']
df = df.drop(columns=cols_to_drop, errors='ignore')

# Renombrar columnas
rename_dict = {
    'P6400': 'Misma_Empresa',
    'P6410': 'Tipo_Empresa_Contratante',
    'P3045S2': 'Empresa_Registrada',
    'P3046': 'Cuenta_Contador',
    'P3363': 'Medio_Empleo_Obtenido',
    'P6450': 'Tipo_Contrato'
}
df = df.rename(columns=rename_dict)

# Limpiar datos
df = df.replace('.', np.nan).dropna()

# Seleccionar características y variable objetivo
selected_features = ['Medio_Empleo_Obtenido', 'Misma_Empresa', 'Tipo_Empresa_Contratante', 
                     'Empresa_Registrada', 'Cuenta_Contador']
df = df[selected_features + ['Tipo_Contrato']]

# Convertir variables categóricas a numéricas
for col in selected_features:
    if df[col].dtype == 'object':
        df[col] = pd.factorize(df[col])[0]
if df['Tipo_Contrato'].dtype == 'object':
    df['Tipo_Contrato'] = pd.factorize(df['Tipo_Contrato'])[0]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df[selected_features], df['Tipo_Contrato'], 
                                                    test_size=0.2, random_state=42)

# Definir modelo Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Validación cruzada
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"📌 Validación cruzada - Precisión media: {cv_scores.mean():.2f}, Desviación estándar: {cv_scores.std():.2f}")

# Evaluación en el conjunto de prueba
y_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"📌 Precisión en test: {test_accuracy:.2f}")

# Crear la carpeta "models" si no existe
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)

# Guardar el modelo en la carpeta "models"
model_path = os.path.join(models_dir, "best_random_forest.pkl")
joblib.dump(rf_model, model_path)
print(f"✅ Modelo guardado en: {model_path}")
