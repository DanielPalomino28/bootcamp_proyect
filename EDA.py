import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Carga del archivo
file_path = r"C:\Users\DRA01\Downloads\data.csv"
df = pd.read_csv(file_path, low_memory=False)

# 1. Eliminar columnas con más del 40% de nulos
threshold = len(df) * 0.4
columns_before = df.shape[1]
df = df.dropna(thresh=threshold, axis=1)
columns_after = df.shape[1]
print(f"Paso 1: Eliminadas {columns_before - columns_after} columnas con más del 40% de valores nulos.")

# 2. Rellenar los campos vacíos
# Para columnas numéricas: rellenar con la mediana
num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns

num_nulls_before = df[num_cols].isnull().sum().sum()
cat_nulls_before = df[cat_cols].isnull().sum().sum()

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

num_nulls_after = df[num_cols].isnull().sum().sum()
cat_nulls_after = df[cat_cols].isnull().sum().sum()

print(f"Paso 2: Rellenados {num_nulls_before + cat_nulls_before} valores nulos ({num_nulls_before} numéricos y {cat_nulls_before} categóricos).")

# 3. Convertir columnas categóricas a numéricas
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
print(f"Paso 3: Convertidas {len(cat_cols)} columnas categóricas a numéricas.")

# 4. Escalar los datos
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print(f"Paso 4: Datos escalados correctamente. Tamaño: {len(df_scaled)}")
print(df_scaled.info())

# Guardar el nuevo archivo
output_path = os.path.join(os.path.dirname(file_path), "data_processed.csv")
df_scaled.to_csv(output_path, index=False)
print(f"Proceso completado. Archivo guardado como: {output_path}")
