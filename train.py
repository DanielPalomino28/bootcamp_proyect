import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Carga del dataset preparado
file_path = r"C:\Users\DRA01\Downloads\data_prepared.csv"
df = pd.read_csv(file_path, low_memory=False)
print("✅ Archivo cargado correctamente.")

# Eliminar columnas que no se utilizarán
cols_to_drop = ['PERIODO', 'MES', 'PER', 'FEX_C18', 'P3044S2', 'P6420S2']
df = df.drop(columns=cols_to_drop, errors='ignore')
print(f"🔹 Columnas eliminadas: {cols_to_drop}")

# Renombrar columnas para una mejor interpretación
rename_dict = {
    'DIRECTORIO': 'Código_Directorio',
    'SECUENCIA_P': 'Secuencia_Persona',
    'ORDEN': 'Orden_Persona',
    'HOGAR': 'Código_Hogar',
    'REGIS': 'Registro_Encuesta',
    'AREA': 'Área_Urbano_Rural',
    'CLASE': 'Clase_Hogar',
    'DPTO': 'Departamento',
    'FT': 'Factor_Ajuste',
    'P6440': 'Tiene_Contrato',         # ¿Tiene contrato? (1=Sí; 2=No; 9=No responde)
    'P6450': 'Tipo_Contrato',           # Target: 1=Verbal, 2=Escrito, 9=No sabe/No responde
    'P6400': 'Misma_Empresa',           # ¿La empresa contratante es la misma donde trabaja?
    'P6410': 'Tipo_Empresa_Contratante',# Tipo de empresa que contrató a la persona
    'P6424S2': 'Prima_Navidad',         # Prima de navidad: 1=Sí; 2=No; 9=No responde
    'P6424S3': 'Derecho_Cesantia',      # Derecho de cesantía: 1=Sí; 2=No; 9=No responde
    'P3045S2': 'Empresa_Registrada',    # ¿La empresa está registrada? (1=Sí; 2=No; 9=No responde)
    'P3046': 'Cuenta_Contador',         # ¿Cuenta con servicios de contador? (1=Sí; 2=No; 9=No responde)
    'P3363': 'Medio_Empleo_Obtenido',    # Medio principal por el que consiguió el empleo
    'P6500': 'Ingreso_Mensual',         # Ingreso antes de descuentos (valor numérico)
    'P6585S3A2': 'Subsidio_Familiar_Incluido'  # ¿Incluyó este valor en ingresos? (1=Sí; 2=No; 9=No responde)
}

df = df.rename(columns=rename_dict)
print("🔹 Columnas renombradas según su descripción.")

# Definir la variable objetivo y las características
target = 'Tipo_Contrato'  # P6450: Tipo de contrato (verbal vs. escrito)
if target not in df.columns:
    raise ValueError(f"La variable objetivo {target} no se encuentra en el DataFrame.")

X = df.drop(columns=[target])
y = df[target]

# Dividir datos en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Datos divididos en entrenamiento y prueba.")

# Crear y entrenar el modelo de árbol de decisión
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("🌳 Modelo de árbol de decisión entrenado.")

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📌 Precisión del modelo: {accuracy:.2f}")

# Guardar el modelo entrenado
model_path = os.path.join(os.path.dirname(file_path), "decision_tree_model.pkl")
joblib.dump(model, model_path)
print(f"✅ Modelo guardado como: {model_path}")

# Guardar el DataFrame modificado (opcional)
output_path = os.path.join(os.path.dirname(file_path), "data_prepared_renamed.csv")
df.to_csv(output_path, index=False)
print(f"✅ Dataset modificado guardado como: {output_path}")
