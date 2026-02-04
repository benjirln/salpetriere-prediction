import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

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

def get_scenario_params(scenario):
    scenarios = {
        "Normal": {"temp": 15, "grippe": 20, "admissions_j1": 200, "occupation_rea": 50},
        "Hiver": {"temp": 0, "grippe": 60, "admissions_j1": 280, "occupation_rea": 75},
        "Épidémie": {"temp": 10, "grippe": 90, "admissions_j1": 350, "occupation_rea": 85},
        "Grève": {"temp": 15, "grippe": 25, "admissions_j1": 150, "occupation_rea": 40},
        "Afflux Massif": {"temp": 25, "grippe": 30, "admissions_j1": 400, "occupation_rea": 90}
    }
    return scenarios.get(scenario, scenarios["Normal"])

def prepare_features(temp, grippe, admissions_j1, occupation_rea, jour_semaine=1, mois=1, scenario="Normal"):
    scenario_mapping = {"Normal": 0, "Hiver": 1, "Épidémie": 2, "Grève": 3, "Afflux Massif": 4}
    scenario_cat = scenario_mapping.get(scenario, 0)
    
    features = {
        'temperature': temp,
        'grippe_saison': grippe,
        'epidemies': 1 if grippe > 60 else 0,
        'jour_semaine': jour_semaine,
        'mois': mois,
        'scenario_cat': scenario_cat,
        'absences_personnel': 5 if jour_semaine in [6, 7] else 2,
        'admissions_veille': admissions_j1,
        'vacances_zone_c': 0,
        'trend_fievre': 30,
        'trend_covid': 15,
        'trend_gastro': 20,
        'trend_grippe': 25,
        'trend_toux': 20,
    }
    return pd.DataFrame([features])

def predict_admissions(model, features_df, force_simulation=False):
    try:
        if force_simulation or model is None:
            temp = features_df['temperature'].values[0]
            grippe = features_df['grippe_saison'].values[0]
            adm_veille = features_df['admissions_veille'].values[0]
            jour_semaine = features_df['jour_semaine'].values[0]
            
            base = adm_veille
            impact_grippe = (grippe / 100) * 80
            impact_temp = 0
            
            if temp < 5:
                impact_temp = (5 - temp) * 3
            elif temp > 30:
                impact_temp = (temp - 30) * 2
            
            ajust_weekend = -20 if jour_semaine in [6, 7] else 0
            
            prediction = base + impact_grippe + impact_temp + ajust_weekend
            return max(int(prediction * 0.95), 100)
        
        prediction = model.predict(features_df)
        return max(int(prediction[0]), 0)
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")
        return 200

def main():
    df = load_data()
    model = load_model()
    
    if df is None:
        st.stop()
    
    st.title("Hôpital Pitié-Salpêtrière")
    st.subheader("Système de Prédiction des Admissions aux Urgences")
    
    if model is not None:
        st.success("Modèle chargé avec succès")

if __name__ == "__main__":
    main()
