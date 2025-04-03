import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
import numpy as np

# 🔹 Cargar dataset
file_path = r"Z:\datos_procesados_v3.csv"
df = pd.read_csv(file_path)
print("✅ Archivo cargado correctamente.")

# 🔹 Convertir Salario en categorías
df["Salario_Categoria"] = pd.cut(df["Salario"], 
                                 bins=[0, 1500000, 3000000, 5000000, 8000000, 10000000], 
                                 labels=[0, 1, 2, 3, 4])


# 🔹 Definir features y targets
selected_features = [
    "Medio_Transporte_Trabajo", 
    "Lugar_Principal_Trabajo", 
    "empresa_formal", 
    "Recibe_Subsidio",
    "Recibe_Prima",
    "Tipo_Trabajo",
    "Horas_Semanales_Trabajo"
]
targets = ["Salario_Categoria", "empresa_formal", "Desea_Cambiar_Trabajo"]

# 🔹 Evaluar la importancia de las columnas para cada target
importance_results = {}
'''
for target in targets:
    print(f"\n🔹 Evaluando importancia de variables para: {target}")

    # Dividir en train/test (asegurando stratificación)
    X_train, X_test, y_train, y_test = train_test_split(df[selected_features], df[target], 
                                                        test_size=0.2, random_state=42, stratify=df[target])
    # Entrenar modelo Random Forest para evaluación de importancia
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)

    # Obtener importancia de las características
    feature_importances = pd.DataFrame({
        'Feature': selected_features,
        'Importance': rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    importance_results[target] = feature_importances
    print(feature_importances)

'''
# 🔹 Entrenar y guardar modelos para cada target
for target in targets:
    print(f"\n🚀 Entrenando modelo para {target}...")

    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(df[selected_features], df[target], 
                                                        test_size=0.2, random_state=42, stratify=df[target])

    # Ajuste de hiperparámetros
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1), 
                                       param_distributions=param_grid, n_iter=20, cv=5, 
                                       scoring='balanced_accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    # Evaluación
    y_pred = random_search.best_estimator_.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"📌 Balanced Accuracy en test para {target}: {bal_acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Guardar modelo
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"best_random_forest_{target}.pkl")
    joblib.dump(random_search.best_estimator_, model_path)
    print(f"✅ Modelo de {target} guardado en: {model_path}")
