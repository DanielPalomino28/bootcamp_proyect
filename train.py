# 5. Dividir los datos en entrenamiento y prueba
X = df_scaled.drop('target_column', axis=1)  # Reemplaza 'target_column' por tu columna objetivo
y = df_scaled['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Entrenar un modelo de clasificación básica
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Precisión del modelo: {accuracy:.2f}')
