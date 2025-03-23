import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Ruta del dataset preparado (asegúrate de que el archivo exista y esté actualizado)
file_path = r"C:\Users\danie\Documents\Bootcamp\data_prepared_renamed.csv"
df = pd.read_csv(file_path, low_memory=False)
print("✅ Archivo cargado correctamente.")

# Definir el target y las variables seleccionadas para el modelo simplificado
target = 'Tipo_Contrato'
selected_features = ['Medio_Empleo_Obtenido', 'Misma_Empresa', 'Tipo_Empresa_Contratante', 
                     'Empresa_Registrada', 'Cuenta_Contador']

# Validar que todas las columnas seleccionadas existan
for col in selected_features + [target]:
    if col not in df.columns:
        raise ValueError(f"La columna {col} no se encuentra en el DataFrame.")

# Seleccionar las características y la variable objetivo
X = df[selected_features]
y = df[target]

# Dividir datos en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Datos divididos en entrenamiento y prueba.")

# Entrenar el modelo de árbol de decisión
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🌳 Modelo simplificado entrenado. Precisión: {accuracy:.2f}")

# Guardar el modelo simplificado en la misma carpeta que el dataset
output_path = os.path.join(os.path.dirname(file_path), "simplified_decision_tree_model.pkl")
joblib.dump(model, output_path)
print(f"✅ Modelo simplificado guardado como: {output_path}")
