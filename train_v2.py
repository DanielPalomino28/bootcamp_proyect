# Definir features y target para el entrenamiento
features = [
    "Medio_Conseguido_Empleo", 
    "Medio_Transporte_Trabajo", 
    "Lugar_Principal_Trabajo", 
    "Tipo_Trabajo", 
    "Recibe_Subsidio"
]
target = ["Salario", "Desea_Cambiar_Trabajo", "empresa_formal"]
# Seleccionar los datos de entrada y salida
X = df[features]
y = df[target]

# División de los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ejemplo de entrenamiento con RandomForest
from sklearn.ensemble import RandomForestRegressor  # O RandomForestClassifier, según el tipo de target
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print("Modelo entrenado.")