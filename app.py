import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Pitié-Salpêtrière - Prédictions Admissions",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('hospital_pitie_salpetriere_COMPLETE_v2.csv', sep=';')
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du CSV : {e}")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('modele_reactif_v1.pkl')
        return model
    except FileNotFoundError:
        st.error("modele_reactif_v1.pkl introuvable")
        return None

def main():
    df = load_data()
    model = load_model()
    
    if df is None:
        st.stop()
    
    st.title("Hôpital Pitié-Salpêtrière")
    st.subheader("Système de Prédiction des Admissions aux Urgences")
    
    if model is not None:
        st.success("Modèle chargé avec succès")
    
    st.info(f"Données chargées : {len(df)} entrées")

if __name__ == "__main__":
    main()
