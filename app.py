from flask import Flask, request, render_template
import pandas as pd
import joblib

# Cargar el modelo entrenado y columnas esperadas
model_path = r"C:\Users\DRA01\Downloads\decision_tree_model.pkl"
modelo_cargado = joblib.load(model_path)

# Cargar el dataset preparado para obtener columnas esperadas
file_path = r"C:\Users\DRA01\Downloads\data_prepared.csv"
df = pd.read_csv(file_path)
columnas = list(df.columns[:-1])  # Excluye la columna objetivo

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Capturar los datos ingresados en el formulario
        try:
            entrada_usuario = [request.form[col] for col in columnas]

            # Crear un DataFrame con los datos del usuario
            nueva_instancia = pd.DataFrame([entrada_usuario], columns=columnas)

            # Hacer la predicción
            prediccion = modelo_cargado.predict(nueva_instancia)

            return render_template("result.html", prediction=prediccion[0], data=entrada_usuario)

        except Exception as e:
            return f"❌ Error procesando la entrada: {e}"

    return render_template("index.html", columns=columnas)

if __name__ == "__main__":
    app.run(debug=True)
