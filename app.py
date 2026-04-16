import streamlit as st
import joblib
import pickle
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

@st.cache_resource
def load_models():
    try:
        # Conectar a Supabase
        conn = psycopg2.connect(
            user=USER, password=PASSWORD,
            host=HOST, port=PORT, dbname=DBNAME
        )
        # Leer datos
        df = pd.read_sql("SELECT * FROM tb_iris", conn)
        conn.close()

        # Entrenar modelo con los datos de Supabase
        X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = df['species']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        target_names = y.unique()
        return model, scaler, target_names

    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

model, scaler, target_names = load_models()

if model is not None:
    st.header("Ingresa las características de la flor:")
    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width  = st.number_input("Ancho del Sépalo (cm)",    min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width  = st.number_input("Ancho del Pétalo (cm)",    min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    if st.button("Predecir Especie"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        st.success(f"Especie predicha: **{prediction}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")
        st.write("Probabilidades:")
        for species, prob in zip(model.classes_, probabilities):
            st.write(f"- {species}: {prob:.1%}")