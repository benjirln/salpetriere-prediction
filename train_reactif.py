import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os



file_path = 'hospital_pitie_salpetriere_COMPLETE_v2.csv'
if not os.path.exists(file_path):
    exit()

df = pd.read_csv(file_path, sep=';')

df['admissions_veille'] = df['admissions_urgences'].shift(1)
df['scenario_cat'] = df['scenario'].astype('category').cat.codes

df = df.dropna()

if 'vacances_zone_c' in df.columns:
    df['vacances_zone_c'] = df['vacances_zone_c'].astype(int)

features = [
    'temperature',         # Variable météo
    'grippe_saison',       # Variable épidémique
    'epidemies',           # Indicateur binaire
    'jour_semaine',        # Jour de la semaine
    'mois',                # Mois
    'scenario_cat',        # Scénario encodé
    'absences_personnel',  # RH
    'admissions_veille',   # Continuité temporelle
    'vacances_zone_c',     # Vacances scolaires Paris
    'trend_fievre',        # Google Trends fièvre
    'trend_covid',         # Google Trends COVID
    'trend_gastro',        # Google Trends gastro
    'trend_grippe',        # Google Trends grippe
    'trend_toux',          # Google Trends toux
]

X = df[features]
y = df['admissions_urgences']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200,      # 200 arbres pour précision
    max_depth=15,          # Profondeur limitée pour éviter surapprentissage
    min_samples_split=5,   # Split minimum
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Précision score r2  : {r2:.4f}")
print(f"Erreur Moyenne MAE  : {mae:.2f} patients")


base_features = pd.DataFrame([{
    'temperature': 15,
    'grippe_saison': 20,
    'epidemies': 0,
    'jour_semaine': 3,
    'mois': 2,
    'scenario_cat': 0,  
    'absences_personnel': 2,
    'admissions_veille': 200,
    'vacances_zone_c': 0,        
    'trend_fievre': 30,
    'trend_covid': 15,
    'trend_gastro': 20,
    'trend_grippe': 25,
    'trend_toux': 20,
}])
pred_base = model.predict(base_features)[0]

test_froid = base_features.copy()
test_froid['temperature'] = -5
pred_froid = model.predict(test_froid)[0]
diff_froid = pred_froid - pred_base

test_epidemie = base_features.copy()
test_epidemie['grippe_saison'] = 90
test_epidemie['epidemies'] = 1
pred_epidemie = model.predict(test_epidemie)[0]
diff_epidemie = pred_epidemie - pred_base


joblib.dump(model, 'modele_reactif_v1.pkl')

