import os
import json
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib
import streamlit as st

st.set_page_config(
    page_title="DermaNet AI | Diagnóstico Dermatoscópico",
    page_icon="🩺",
    layout="wide"
)

IMG_SIZE = 224
MODEL_PATH = "dermanet_efficientnetb0.keras"
CLASS_INFO_PATH = "class_info.json"
SAMPLE_DIR = "sample_images"
DB_PATH = "predicciones_historial.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historial_diagnosticos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_hora TEXT,
            origen_imagen TEXT,
            codigo_dx TEXT,
            diagnostico TEXT,
            confianza REAL
        )
    """)
    conn.commit()
    return conn

conn = init_db()

@st.cache_resource
def load_model_and_metadata():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)
    return model, info["class_names"], info["lesion_type_dict"]

def make_gradcam_heatmap(img_array, model, pred_index):
    base_model = None
    idx_base = 0
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            base_model = layer
            idx_base = i
            break

    if base_model is None:
        raise ValueError("No se encontró el backbone convolucional.")

    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    grad_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output]
    )

    post_layers = model.layers[idx_base + 1:]

    with tf.GradientTape() as tape:
        conv_outputs, x = grad_model(tf.cast(img_array, tf.float32), training=False)
        tape.watch(conv_outputs)
        for layer in post_layers:
            x = layer(x, training=False)
        class_channel = x[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(pil_img, heatmap, alpha=0.45):
    img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    jet_img = Image.fromarray(np.uint8(255 * jet_heatmap)).resize((IMG_SIZE, IMG_SIZE))
    jet_array = np.array(jet_img, dtype=np.float32)

    superimposed = jet_array * alpha + img.astype(np.float32)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return Image.fromarray(superimposed)

st.title("🩺 DermaNet AI: Clasificación de Lesiones Dermatológicas")
st.markdown(
    "Sistema de apoyo diagnóstico basado en **Deep Learning (`EfficientNetB0` + Fine-Tuning)** "
    "entrenado sobre el conjunto clínico **HAM10000**."
)

with st.sidebar:
    st.header("⚙️ Fuente de Imagen")
    modo = st.radio(
        "Selecciona cómo cargar la imagen:",
        ["Galería de muestras (Test Set)", "Subir imagen propia (.jpg / .png)"]
    )

    pil_image = None
    source_label = ""

    if modo == "Galería de muestras (Test Set)":
        if os.path.exists(SAMPLE_DIR):
            samples = sorted([f for f in os.listdir(SAMPLE_DIR) if f.endswith((".jpg", ".png"))])
            if samples:
                selected_sample = st.selectbox("Elige una lesión de prueba:", samples)
                pil_image = Image.open(os.path.join(SAMPLE_DIR, selected_sample)).convert("RGB")
                source_label = f"Muestra: {selected_sample}"
            else:
                st.warning("No se encontraron imágenes en `sample_images/`.")
        else:
            st.warning("Crea la carpeta `sample_images/` con las muestras exportadas.")
    else:
        uploaded_file = st.file_uploader("Sube una imagen dermatoscópica", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")
            source_label = f"Archivo: {uploaded_file.name}"

    st.divider()
    st.info(
        "**Aviso Académico:** Herramienta desarrollada para el proyecto ACA N° 2 (CUN). "
        "No reemplaza el criterio médico profesional."
    )

try:
    model, class_names, lesion_type_dict = load_model_and_metadata()
except Exception as e:
    st.error(f"Error al cargar el modelo o metadatos: {e}")
    st.stop()

if pil_image is not None:
    img_resized = pil_image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top_code = class_names[top_idx]
    top_name = lesion_type_dict[top_code]
    top_conf = float(preds[top_idx]) * 100

    conn.execute(
        "INSERT INTO historial_diagnosticos (fecha_hora, origen_imagen, codigo_dx, diagnostico, confianza) VALUES (?, ?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), source_label, top_code, top_name, round(top_conf, 2))
    )
    conn.commit()

    col1, col2, col3 = st.columns([1, 1, 1.3])

    with col1:
        st.subheader("Imagen Analizada")
        st.image(img_resized, caption=source_label, use_container_width=True)

    with col2:
        st.subheader("Explicabilidad (Grad-CAM)")
        try:
            heatmap = make_gradcam_heatmap(img_array, model, top_idx)
            gradcam_img = overlay_gradcam(pil_image, heatmap)
            st.image(gradcam_img, caption="Regiones de mayor atención neuronal", use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo generar el mapa Grad-CAM: {e}")

    with col3:
        st.subheader("Resultado Diagnóstico")
        st.metric(
            label="Predicción Principal",
            value=top_name,
            delta=f"{top_conf:.2f}% de confianza"
        )

        df_probs = pd.DataFrame({
            "Código": class_names,
            "Diagnóstico": [lesion_type_dict[c] for c in class_names],
            "Probabilidad (%)": np.round(preds * 100, 2)
        }).sort_values("Probabilidad (%)", ascending=False)

        st.bar_chart(df_probs.set_index("Diagnóstico")["Probabilidad (%)"])

    with st.expander("📊 Ver tabla detallada de probabilidades y registro SQL"):
        st.dataframe(df_probs, use_container_width=True, hide_index=True)
        st.markdown("**Últimas 5 consultas registradas en SQLite (`historial_diagnosticos`):**")
        df_hist = pd.read_sql_query(
            "SELECT fecha_hora, origen_imagen, diagnostico, confianza FROM historial_diagnosticos ORDER BY id DESC LIMIT 5",
            conn
        )
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
else:
    st.info("👈 Selecciona una imagen de muestra o sube una fotografía desde el panel izquierdo para iniciar el análisis.")
