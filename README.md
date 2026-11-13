# 🩺 DermaNet AI — Clasificación de Lesiones Dermatológicas (ACA N° 2)

Aplicación interactiva desarrollada en **Streamlit** basada en aprendizaje por transferencia (*Transfer Learning* + *Fine-Tuning*) sobre la arquitectura **EfficientNetB0** entrenada con el conjunto de datos clínico **Skin Cancer MNIST: HAM10000**.

## 📂 Estructura del Repositorio

* `app.py`: Código principal de la interfaz web interactiva, inferencia neuronal, visualización de explicabilidad **Grad-CAM** y registro en **SQLite**.
* `dermanet_efficientnetb0.keras`: Modelo neuronal entrenado y serializado listo para inferencia.
* `class_info.json`: Diccionario de mapeo entre códigos diagnósticos (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`) y nombres clínicos.
* `requirements.txt`: Dependencias necesarias para ejecutar la aplicación.
* `sample_images/`: Imágenes dermatoscópicas del conjunto de prueba (*Test Set*) para validación rápida.

## 🚀 Instrucciones para Cargar el Modelo y Ejecutar Localmente

1. **Clonar el repositorio:**
   git clone https://github.com/Juancho1202-tech/Streamlit-project-ACA-2.git
   cd Streamlit-project-ACA-2

2. **Instalar dependencias:**
   pip install -r requirements.txt

3. **Ejecutar la aplicación Streamlit (carga automáticamente `dermanet_efficientnetb0.keras`):**
   streamlit run app.py
