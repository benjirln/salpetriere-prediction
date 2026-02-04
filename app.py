import streamlit as st

st.set_page_config(
    page_title="Pitié-Salpêtrière - Prédictions Admissions",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Hôpital Pitié-Salpêtrière")
    st.subheader("Système de Prédiction des Admissions aux Urgences")

if __name__ == "__main__":
    main()
