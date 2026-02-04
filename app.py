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

st.markdown("""
<style>
    .main {
        background-color: #FFFFFF;
    }
    
    h1, h2, h3, h4 {
        color: #1E3A8A !important;
    }
    
    p, div, span, label {
        color: #1E40AF !important;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 4px solid #3B82F6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.1);
        margin: 10px 0;
    }
    
    .kpi-title {
        color: #1E40AF;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        color: #1E3A8A;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .kpi-subtitle {
        color: #60A5FA;
        font-size: 12px;
    }
    
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
    }
    
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background-color: #2563EB;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        color: #1E40AF;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


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
        model = joblib.load('modele_reactif_v2.pkl')
        return model
    except FileNotFoundError:
        st.error("modele_reactif_v2.pkl introuvable")
        return None

def get_scenario_params(scenario):
    scenarios = {
        "Normal": {
            "temp": 15, 
            "grippe": 20, 
            "admissions_j1": 200, 
            "occupation_rea": 50,
            "ide_dispo": 18,
            "stock_masques_jours": 10
        },
        "Hiver": {
            "temp": 0, 
            "grippe": 60, 
            "admissions_j1": 280, 
            "occupation_rea": 75,
            "ide_dispo": 15,
            "stock_masques_jours": 7
        },
        "Épidémie": {
            "temp": 10, 
            "grippe": 90, 
            "admissions_j1": 350, 
            "occupation_rea": 85,
            "ide_dispo": 16,
            "stock_masques_jours": 3
        },
        "Grève": {
            "temp": 15, 
            "grippe": 25, 
            "admissions_j1": 150, 
            "occupation_rea": 40,
            "ide_dispo": 10,
            "stock_masques_jours": 12
        },
        "Afflux Massif": {
            "temp": 25, 
            "grippe": 30, 
            "admissions_j1": 400, 
            "occupation_rea": 90,
            "ide_dispo": 20,
            "stock_masques_jours": 5
        }
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
        'reanimation_occupes': occupation_rea,
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

def calculate_kpis(admissions):
    besoins_ide = int(np.ceil(admissions / 15))
    besoins_medecins = int(np.ceil(admissions / 30))
    consommation_masques = admissions * 9
    taux_occupation = min(int((admissions / 250) * 100), 100)

    return {
        'besoins_ide': besoins_ide,
        'besoins_medecins': besoins_medecins,
        'consommation_masques': consommation_masques,
        'taux_occupation': taux_occupation
    }

def generate_alerts(kpis, stock_masques_jours=5, ide_dispo=18):
    alerts = []

    if kpis['besoins_ide'] > ide_dispo:
        diff = kpis['besoins_ide'] - ide_dispo
        alerts.append({
            'type': 'error',
            'message': f"**Alerte RH** : Rappel de personnel nécessaire (+{diff} IDE)"
        })

    if kpis['taux_occupation'] > 90:
        alerts.append({
            'type': 'error',
            'message': f"**Saturation {kpis['taux_occupation']}%** : Plan Blanc suggéré"
        })
    elif kpis['taux_occupation'] > 75:
        alerts.append({
            'type': 'warning',
            'message': f"**Occupation élevée ({kpis['taux_occupation']}%)** : Surveillance renforcée"
        })

    consommation_quotidienne = kpis['consommation_masques']
    if stock_masques_jours < 3:
        alerts.append({
            'type': 'error',
            'message': f"**Logistique** : Commande urgente masques nécessaire (stock < 3 jours)"
        })
    elif stock_masques_jours < 7:
        alerts.append({
            'type': 'warning',
            'message': f"**Stock faible** : Anticiper commande masques ({stock_masques_jours} jours restants)"
        })

    if len(alerts) == 0:
        alerts.append({
            'type': 'success',
            'message': "**Situation maîtrisée** : Toutes les ressources sont suffisantes"
        })

    return alerts

def main():
    df = load_data()
    model = load_model()

    if df is None:
        st.stop()

    st.title("Hôpital Pitié-Salpêtrière")
    st.subheader("Système de Prédiction des Admissions aux Urgences")

    st.sidebar.markdown("### Simulateur de Scénarios")

    scenario = st.sidebar.selectbox(
        "Scénario",
        ["Normal", "Hiver", "Épidémie", "Grève", "Afflux Massif"],
        help="Sélectionnez un scénario prédéfini"
    )

    params = get_scenario_params(scenario)

    st.sidebar.markdown("#### Conditions Météo & Santé")

    temperature = st.sidebar.slider(
        "Température (°C)",
        min_value=-10,
        max_value=40,
        value=params['temp'],
        help="Température prévue pour demain"
    )

    intensite_grippe = st.sidebar.slider(
        "Intensité Grippe",
        min_value=0,
        max_value=100,
        value=params['grippe'],
        help="Indicateur épidémique (0 = faible, 100 = épidémie)"
    )

    st.sidebar.markdown("#### Données de la Veille (J-1)")

    admissions_j1 = st.sidebar.number_input(
        "Admissions J-1",
        min_value=50,
        max_value=500,
        value=params['admissions_j1'],
        step=10,
        help="Nombre d'admissions de la veille"
    )

    occupation_rea = st.sidebar.number_input(
        "Occupation Réa J-1 (%)",
        min_value=0,
        max_value=100,
        value=params['occupation_rea'],
        step=5,
        help="Taux d'occupation de la réanimation"
    )


    ide_dispo = st.sidebar.number_input("IDE disponibles", min_value=5, max_value=50, value=params['ide_dispo'])
    stock_masques_jours = st.sidebar.number_input("Stock masques (jours)", min_value=0, max_value=30, value=params['stock_masques_jours'])

    features_df = prepare_features(
        temp=temperature,
        grippe=intensite_grippe,
        admissions_j1=admissions_j1,
        occupation_rea=occupation_rea,
        jour_semaine=datetime.now().weekday() + 1,
        mois=datetime.now().month,
        scenario=scenario
    )

    admissions_predites = predict_admissions(model, features_df, force_simulation=False)
    kpis = calculate_kpis(admissions_predites)
    alerts = generate_alerts(kpis, stock_masques_jours, ide_dispo)

    st.markdown("### Tableau de Bord Décisionnel - Prévisions J+1")
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ADMISSIONS PRÉVUES</div>
            <div class="kpi-value">{admissions_predites}</div>
            <div class="kpi-subtitle">Prédiction IA</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">BESOINS IDE</div>
            <div class="kpi-value">{kpis['besoins_ide']}</div>
            <div class="kpi-subtitle">1 IDE / 15 admissions</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        color = "#EF4444" if kpis['taux_occupation'] > 90 else "#F59E0B" if kpis['taux_occupation'] > 75 else "#10B981"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">TAUX OCCUPATION LITS</div>
            <div class="kpi-value" style="color: {color};">{kpis['taux_occupation']}%</div>
            <div class="kpi-subtitle">Service Urgences: 250 lits</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">CONSOMMATION MASQUES</div>
            <div class="kpi-value">{kpis['consommation_masques']:,}</div>
            <div class="kpi-subtitle">9 masques / patient</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Centre d'Alertes")

    for alert in alerts:
        if alert['type'] == 'error':
            st.error(alert['message'])
        elif alert['type'] == 'warning':
            st.warning(alert['message'])
        else:
            st.success(alert['message'])

if __name__ == "__main__":
    main()