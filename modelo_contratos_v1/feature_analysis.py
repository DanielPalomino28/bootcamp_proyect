import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Ruta del dataset y del modelo (ajustar según corresponda)
data_path = r"C:\Users\danie\Documents\Bootcamp\data_prepared_renamed.csv"
model_path = os.path.join(os.path.dirname(data_path), "decision_tree_model.pkl")

# Cargar dataset y modelo
df = pd.read_csv(data_path, low_memory=False)
model = joblib.load(model_path)

# Definir la variable objetivo y las características
target = 'Tipo_Contrato'
X = df.drop(columns=[target])

# Extraer las importancias del modelo
importances = model.feature_importances_
features = X.columns

# Crear un DataFrame para ordenar las importancias
feat_importance = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("Importancia de las variables:")
print(feat_importance)

# Seleccionar las 4 características más importantes
top_features = feat_importance.head(4)
print("\nLas 4 características más importantes son:")
print(top_features)

# Opcional: gráfico de barras para visualizar la importancia de las variables
plt.figure(figsize=(8, 6))
plt.barh(top_features['feature'][::-1], top_features['importance'][::-1], color='skyblue')
plt.xlabel("Importancia")
plt.title("Top 4 Características según el Árbol de Decisión")
plt.show()
