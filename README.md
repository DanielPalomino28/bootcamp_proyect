# 🎯 **Proyecto de Predicción de Tipo de Contrato**

### 🧠 **Descripción del Proyecto**
Este proyecto desarrolla una aplicación web sencilla y elegante utilizando Flask y Bootstrap. Su objetivo principal es predecir el tipo de contrato laboral de una persona, basado en un modelo de Machine Learning entrenado previamente. El modelo ha sido ajustado y simplificado para permitir una interacción rápida y eficiente a través de una interfaz web.

---

### 🛠️ **Tecnologías utilizadas**
- **Python 3.10+**  
- **Flask** (framework web ligero)  
- **Pandas** (procesamiento de datos)  
- **Joblib** (para cargar el modelo guardado)  
- **Bootstrap 4.5+** (estilización rápida y adaptable)  
- **Sklearn** (entrenamiento del modelo)

---

### 🗂️ **Estructura del proyecto**

```
/proyecto-prediccion-contrato
│
├── app.py                      # Código principal de la aplicación Flask
├── train_simplified.py         # Script para entrenar el modelo simplificado
├── simplified_decision_tree_model.pkl # Modelo entrenado guardado
│
├── /templates                  # Plantillas HTML para la interfaz web
│   ├── index.html              # Página principal (formulario de entrada)
│   └── result.html             # Página de resultados
│
└── README.md                   # Este archivo de documentación
```

---

### ⚙️ **Instalación y ejecución**

1️⃣ **Clonar el repositorio**  
```bash
git clone https://github.com/DanielPalomino28/bootcamp_proyect.git
```

2️⃣ **Instalar dependencias**  
Desde la raíz del proyecto, ejecuta:  
```bash
pip install -r requirements.txt
```
*(Asegúrate de tener un entorno virtual configurado si prefieres aislar las dependencias).*

3️⃣ **Ejecutar la aplicación**  
```bash
python app.py
```
Esto iniciará un servidor local, generalmente en: [http://localhost:5000](http://localhost:5000)

---

### 🧠 **¿Cómo funciona?**

1️⃣ **Ingreso de datos**  
El usuario completa un formulario con opciones predefinidas (selectores), garantizando que los datos sean válidos y coherentes.

2️⃣ **Predicción**  
Al enviar el formulario, los datos ingresados se procesan y se pasan al modelo cargado (`simplified_decision_tree_model.pkl`).

3️⃣ **Resultado**  
La app devuelve la predicción del tipo de contrato ("Verbal", "Escrito" o "No sabe/No responde") en una página amigable.

---

### 📌 **Variables consideradas**
El modelo fue ajustado y solo utiliza las variables más importantes:

- **Medio en que obtuvo su empleo** (Bolsa de Trabajo, Red de Contactos, etc.)
- **Si trabaja en la misma empresa donde fue contratado** (Sí/No)
- **Tipo de empresa que contrató** (Pública, Privada, Mixta, Otra)
- **Si la empresa está registrada** (Sí/No)
- **Si cuenta con un contador** (Sí/No)

---

### 👥 **Créditos**
Proyecto desarrollado por el equipo:  
**Daniel * 3 + Juan * 2**  
🚀 Presentado en el **Bootcamp de Inteligencia Artificial, nivel explorador**

---

### 📌 **Posibles mejoras futuras**
- ✅ Agregar validación de errores más detallada.
- ✅ Permitir subir archivos CSV para predicción masiva.
- ✅ Implementar visualización gráfica de resultados.
- ✅ Desplegar la aplicación en un servidor (Heroku, Render, etc.).

---

¿Listo para predecir contratos? 🎯✨  
Siéntete libre de personalizar la app y agregar más funcionalidades. 💪✨