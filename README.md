# Prédictions Admissions - Hôpital Pitié-Salpêtrière

Application d'intelligence artificielle pour prédire les admissions aux urgences et anticiper les besoins en ressources (lits, personnel, matériel).

## Installation

1.  **Cloner ou télécharger le projet**
2.  **Installer les dépendances** :
    ```bash
    pip install streamlit pandas numpy joblib scikit-learn plotly matplotlib seaborn
    ```

## 🧠 Entraînement du Modèle

vous enntraîner le modèle avec :

```bash
python train_reactif.py
```
Cela générera un fichier `.pkl` basé sur le fichier `hospital_pitie_salpetriere_COMPLETE_v2.csv`.

## Lancer l'Application

Pour démarrer le tableau de bord interactif :

```bash
python -m streamlit run app.py
```

