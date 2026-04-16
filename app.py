import streamlit as st 
import numpy as np
import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Credenciales
USER = "postgres.dyqjvckxitsrgmytbdnw"
PASSWORD = "Petunia_Bob_Takemichi"
HOST = "aws-1-us-east-1.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")
st.title("🌸 Predictor de Especies de Iris")

# -------------------------------
# HISTÓRICO (MEMORIA)
# -------------------------------
if "historial" not in st.session_state:
    st.session_state.historial = []

# -------------------------------
# CARGA Y ENTRENAMIENTO
# -------------------------------
@st.cache_resource
def load_models():
    try:
        conn = psycopg2.connect(
            user=USER, password=PASSWORD,
            host=HOST, port=PORT, dbname=DBNAME
        )

        df = pd.read_sql("SELECT * FROM ml.tb_iris", conn)
        conn.close()

        X = df[['l_s', 'a_s', 'l_p', 'a_p']]
        y = df['prediccion']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        return model, scaler

    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

model, scaler = load_models()

# -------------------------------
# INTERFAZ
# -------------------------------
if model is not None:
    st.header("Ingresa las características de la flor:")

    sepal_length = st.number_input("Longitud del Sépalo (cm)", 0.0, 10.0, 5.0, 0.1)
    sepal_width  = st.number_input("Ancho del Sépalo (cm)",    0.0, 10.0, 3.0, 0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", 0.0, 10.0, 4.0, 0.1)
    petal_width  = st.number_input("Ancho del Pétalo (cm)",    0.0, 10.0, 1.0, 0.1)

    if st.button("Predecir Especie"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confianza = max(probabilities)

        st.success(f"Especie predicha: **{prediction}**")
        st.write(f"Confianza: **{confianza:.1%}**")

        st.write("Probabilidades:")
        for species, prob in zip(model.classes_, probabilities):
            st.write(f"- {species}: {prob:.1%}")

        # -------------------------------
        # GUARDAR EN HISTÓRICO
        # -------------------------------
        st.session_state.historial.append({
            "Sepalo_L": sepal_length,
            "Sepalo_A": sepal_width,
            "Petalo_L": petal_length,
            "Petalo_A": petal_width,
            "Prediccion": prediction,
            "Confianza": round(confianza, 4)
        })

# -------------------------------
# MOSTRAR HISTÓRICO
# -------------------------------
st.subheader("📊 Histórico de predicciones")

if st.session_state.historial:
    df_hist = pd.DataFrame(st.session_state.historial)
    st.dataframe(df_hist, use_container_width=True)
else:
    st.info("Aún no hay predicciones.")

# -------------------------------
# LIMPIAR HISTÓRICO
# -------------------------------
if st.button("🗑️ Limpiar historial"):
    st.session_state.historial = []
    st.success("Historial eliminado")