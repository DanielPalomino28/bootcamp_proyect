from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Cargar el modelo simplificado entrenado
model_path = os.path.join(r"C:\Users\danie\Documents\Bootcamp", "simplified_decision_tree_model.pkl")
model = joblib.load(model_path)

# Definir las opciones para cada campo (ajustar según la codificación oficial)
medios_empleo = {
    "1": "Bolsa de Trabajo",
    "2": "Red de Contactos",
    "3": "Medios Digitales",
    "4": "Otro"
}
misma_empresa = {
    "1": "Sí",
    "2": "No"
}
tipo_empresa = {
    "1": "Empresa Pública",
    "2": "Empresa Privada",
    "3": "Empresa Mixta",
    "4": "Otra"
}
empresa_registrada = {
    "1": "Sí",
    "2": "No"
}
cuenta_contador = {
    "1": "Sí",
    "2": "No"
}

# Mapeo para el resultado (ajustar según la codificación del target)
tipo_contrato_mapping = {
    1: "Verbal",
    2: "Escrito",
    9: "No sabe/No responde"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Recoger los datos del formulario y convertirlos a enteros
        input_data = {
            "Medio_Empleo_Obtenido": int(request.form["Medio_Empleo_Obtenido"]),
            "Misma_Empresa": int(request.form["Misma_Empresa"]),
            "Tipo_Empresa_Contratante": int(request.form["Tipo_Empresa_Contratante"]),
            "Empresa_Registrada": int(request.form["Empresa_Registrada"]),
            "Cuenta_Contador": int(request.form["Cuenta_Contador"])
        }
        # Convertir el input a DataFrame (asegurando la misma estructura que se usó en el entrenamiento)
        input_df = pd.DataFrame([input_data])
        # Hacer la predicción
        prediction_code = model.predict(input_df)[0]
        prediction_label = tipo_contrato_mapping.get(prediction_code, "Desconocido. (Código: {prediction_code})")
        # Mostrar el resultado
        return render_template("result.html", prediction=prediction_label)
    return render_template("index.html",
                           medios_empleo=medios_empleo,
                           misma_empresa=misma_empresa,
                           tipo_empresa=tipo_empresa,
                           empresa_registrada=empresa_registrada,
                           cuenta_contador=cuenta_contador)

if __name__ == "__main__":
    app.run(debug=True)
