import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
import numpy as np

# 🔹 Cargar dataset
file_path = r"C:\Users\DRA01\Downloads\datos_procesados.csv"
df = pd.read_csv(file_path)
print("✅ Archivo cargado correctamente.")

# 🔹 Convertir columna Tipo_Trabajo a numérico
df['Tipo_Trabajo'] = df['Tipo_Trabajo'].astype('category').cat.codes

# 🔹 Convertir Salario a float32
df['Salario'] = df['Salario'].astype(np.float32)

# 🔹 Definir rangos válidos de valores
columnas_info_filtrado = {
    "Medio_Transporte_Trabajo": {"min": 1, "max": 14},
    "Lugar_Principal_Trabajo": {"min": 1, "max": 8},
    "empresa_formal": {"min": 0, "max": 1},
    "Recibe_Subsidio": {"min": 0, "max": 1},
    "Recibe_Prima": {"min": 0, "max": 1},
    "Tipo_Trabajo": {"min": 1, "max": 9},
    "Salario": {"min": 50000, "max": 10000000} 
}

# 🔹 Corregir valores fuera del rango en lugar de eliminarlos
for col, info in columnas_info_filtrado.items():
    df[col] = df[col].apply(lambda x: np.random.randint(info["min"], info["max"] + 1) if x < info["min"] or x > info["max"] else x)

print("✅ Valores fuera de rango corregidos con valores aleatorios dentro del rango permitido.")

# 🔹 Definir features y target
selected_features = [
    "Medio_Transporte_Trabajo", 
    "Lugar_Principal_Trabajo", 
    "empresa_formal", 
    "Recibe_Subsidio",
    "Recibe_Prima",
    "Tipo_Trabajo"
]
target = "Salario"

# 🔹 Normalizar Salario
scaler = StandardScaler()
df['Salario'] = scaler.fit_transform(df[['Salario']])

# 🔹 Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df[selected_features], df[target], 
                                                    test_size=0.2, random_state=42)

# 🔹 Definir modelo Random Forest con paralelización
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 🔹 Validación cruzada
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"📌 Validación cruzada - Precisión media: {cv_scores.mean():.2f}, Desviación estándar: {cv_scores.std():.2f}")

# 🔹 Evaluación en el conjunto de prueba
y_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"📌 Precisión en test: {test_accuracy:.2f}")

# 🔹 Ajuste de hiperparámetros con RandomizedSearchCV
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1), 
                                   param_distributions=param_grid, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

print(f"✅ Mejor modelo encontrado: {random_search.best_params_}")

# 🔹 Evaluación del mejor modelo
y_pred_best = random_search.best_estimator_.predict(X_test)
best_test_accuracy = accuracy_score(y_test, y_pred_best)
print(f"📌 Precisión del mejor modelo en test: {best_test_accuracy:.2f}")

# 🔹 Guardar el mejor modelo
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "best_random_forest_v5.pkl")
joblib.dump(random_search.best_estimator_, model_path)
print(f"✅ Modelo optimizado guardado en: {model_path}")
