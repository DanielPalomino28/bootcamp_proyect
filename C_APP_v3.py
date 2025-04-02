from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Cargar modelos entrenados (ajusta las rutas según corresponda)
base_model_path = r"C:\Users\danie\bootcamp_proyect\models"
model_salario_cat = joblib.load(os.path.join(base_model_path, "best_random_forest_Salario_Categoria.pkl"))
model_empresa_formal = joblib.load(os.path.join(base_model_path, "best_random_forest_empresa_formal.pkl"))
model_desea_cambiar = joblib.load(os.path.join(base_model_path, "best_random_forest_Desea_Cambiar_Trabajo.pkl"))

# Opciones para el formulario (actualizadas según la codificación real)
medio_transporte = {
    "1": "Bus intermunicipal",
    "2": "Bus urbano",
    "3": "A pie",
    "4": "Metro",
    "5": "Transporte articulado",
    "6": "Taxi",
    "7": "Transporte de la empresa",
    "8": "Automóvil particular",
    "9": "Lancha, planchón, canoa",
    "10": "Caballo",
    "11": "Moto",
    "12": "Mototaxi",
    "13": "Bicicleta",
    "14": "No se desplaza"
}

lugar_principal = {
    "1": "En esta vivienda",
    "2": "En otras viviendas",
    "3": "En kiosco - caseta",
    "4": "En un vehículo",
    "5": "De puerta en puerta",
    "6": "Sitio al descubierto en la calle (ambulante y estacionario)",
    "7": "Local fijo, oficina, fábrica, etc.",
    "8": "En el campo o área rural, mar o río",
    "9": "En una obra en construcción",
    "10": "En una mina o cantera",
    "11": "Otro"
}

empresa_formal_opts = {
    "1": "Sí",
    "2": "No"
}

recibe_subsidio_opts = {
    "1": "Sí",
    "2": "No"
}

recibe_prima_opts = {
    "1": "Sí",
    "2": "No"
}

tipo_trabajo_opts = {
    "1": "Obrero o empleado de empresa particular",
    "2": "Obrero o empleado del gobierno",
    "3": "Empleado doméstico",
    "4": "Trabajador por cuenta propia",
    "5": "Patrón o empleador",
    "6": "Trabajador familiar sin remuneración",
    "7": "Trabajador sin remuneración en empresas o negocios de otros hogares",
    "8": "Jornalero o peón",
    "9": "Otro"
}

# Mapeo para resultados de predicción
salario_cat_mapping = {
    0: "Bajo",
    1: "Medio-bajo",
    2: "Medio",
    3: "Medio-alto",
    4: "Alto"
}
empresa_formal_mapping = {
    1: "Sí",
    2: "No"
}
desea_cambiar_mapping = {
    0: "No desea cambiar de trabajo",
    1: "Desea cambiar de trabajo"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Recoger los datos del formulario y convertirlos a los tipos esperados
        input_data = {
            "Medio_Transporte_Trabajo": int(request.form["Medio_Transporte_Trabajo"]),
            "Lugar_Principal_Trabajo": int(request.form["Lugar_Principal_Trabajo"]),
            "empresa_formal": int(request.form["empresa_formal"]),
            "Recibe_Subsidio": int(request.form["Recibe_Subsidio"]),
            "Recibe_Prima": int(request.form["Recibe_Prima"]),
            "Tipo_Trabajo": int(request.form["Tipo_Trabajo"]),
            "Horas_Trabajadas": float(request.form["Horas_Trabajadas"])
        }
        # Convertir el input a DataFrame para que tenga la misma estructura que en el entrenamiento
        input_df = pd.DataFrame([input_data])
        
        # Realizar las predicciones con cada modelo
        pred_salario_cat = model_salario_cat.predict(input_df)[0]
        pred_empresa_formal = model_empresa_formal.predict(input_df)[0]
        pred_desea_cambiar = model_desea_cambiar.predict(input_df)[0]
        
        # Mapear los códigos a etiquetas legibles
        salario_cat_label = salario_cat_mapping.get(pred_salario_cat, f"Desconocido ({pred_salario_cat})")
        empresa_formal_label = empresa_formal_mapping.get(pred_empresa_formal, f"Desconocido ({pred_empresa_formal})")
        desea_cambiar_label = desea_cambiar_mapping.get(pred_desea_cambiar, f"Desconocido ({pred_desea_cambiar})")
        
        # Crear un diccionario con los resultados para enviarlo a la plantilla
        results = {
            "Salario_Categoria": salario_cat_label,
            "Empresa_Formal": empresa_formal_label,
            "Desea_Cambiar_Trabajo": desea_cambiar_label
        }
        
        return render_template("result.html", results=results)
    
    return render_template("index.html",
                           medio_transporte=medio_transporte,
                           lugar_principal=lugar_principal,
                           empresa_formal_opts=empresa_formal_opts,
                           recibe_subsidio_opts=recibe_subsidio_opts,
                           recibe_prima_opts=recibe_prima_opts,
                           tipo_trabajo_opts=tipo_trabajo_opts)

if __name__ == "__main__":
    app.run(debug=True)
