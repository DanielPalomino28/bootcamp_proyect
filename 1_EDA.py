import pandas as pd
import os

# Carga del archivo
file_path = r"C:\Users\danie\Documents\Bootcamp\data.csv" ##Cambiar por la ruta dependiendo donde se ejecuta el script
df = pd.read_csv(file_path, low_memory=False)
print("✅ Archivo cargado correctamente.")

# 1. Eliminar columnas con más del 40% de nulos
threshold = len(df) * 0.4
cols_before = df.shape[1]
df = df.dropna(thresh=threshold, axis=1)
cols_after = df.shape[1]
print(f"🔹 Columnas eliminadas por exceso de nulos: {cols_before - cols_after}")

# 2. Rellenar los campos vacíos
# Para columnas numéricas: rellenar con la mediana
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
print(f"🔹 Campos numéricos vacíos rellenados con la mediana.")

# Para columnas categóricas: rellenar con la moda
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
print(f"🔹 Campos categóricos vacíos rellenados con la moda.")

# Guardar el dataset preparado
output_path = os.path.join(os.path.dirname(file_path), "data_prepared.csv")
df.to_csv(output_path, index=False)
print(f"✅ EDA completado. Archivo guardado como: {output_path}")
