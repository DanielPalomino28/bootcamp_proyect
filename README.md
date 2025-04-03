# 🎯 **Predicción Multiobjetivo en Ámbito Laboral: Modelos de Machine Learning para Clasificar Salario, Formalidad Laboral y Preferencia de Cambio de Empleo**

### 🧠 **Descripción del Proyecto**
Este proyecto desarrolla una aplicación web sencilla y elegante utilizando Flask y Bootstrap. Su objetivo principal es predecir las siguientes vaiables: Clasificar Salario, Formalidad Laboral y Preferencia de Cambio de Empleo, basado en un modelo de Machine Learning entrenado previamente. El modelo ha sido ajustado y simplificado para permitir una interacción rápida y eficiente a través de una interfaz web.

---

### 🛠️ **Tecnologías utilizadas**
- **Python 3.10+**  
- **Flask** (framework web ligero)  
- **Pandas** (procesamiento de datos)  
- **Joblib** (para cargar el modelo guardado)  
- **Bootstrap 4.5+** (estilización rápida y adaptable)  
- **Sklearn** (entrenamiento del modelo)

---

### 🐂 **Estructura del proyecto**

```
/bootcamp_proyect
│
├── C_APP_v3.py                     # Código principal de la aplicación Flask
├── B_Train_v3.py                   # Script para entrenar el modelo simplificado
├── A_EDA_v3.py                     # Script para analisar y realizar limpieza 
├── models.rar                      # Modelos entrenados que son necesarios para el despliegue (comprimidos)
│
├── /templates                      # Plantillas HTML para la interfaz web
│   ├── index.html                  # Página principal (formulario de entrada)
│   └── result.html                 # Página de resultados
│
├── README.md                       # Este archivo de documentación
└── graficos_v3                     # Carpteta que contiene los gráficos de analisis para el dataset procesado (EDA)

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

3️⃣ **Ejecutar la aplicación**  
```bash
python C_APP_v3.py
```
Esto iniciará un servidor local, generalmente en: [http://localhost:5000](http://localhost:5000)

---

### 🧠 **¿Cómo funciona?**

1️⃣ **Ingreso de datos**  
El usuario completa un formulario con opciones predefinidas (selectores), garantizando que los datos sean válidos y coherentes.

2️⃣ **Predicción**  
Al enviar el formulario, los datos ingresados se procesan y se pasan a los modelos cargados (descomprimir `models.rar`).

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

