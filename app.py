import streamlit as st
import joblib
import pandas as pd
import mlflow 

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Prédiction défaut crédit",
    page_icon="💳",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Aller vers",
    ["Accueil", "Prédiction"]
)

# =========================
# PAGE ACCUEIL
# =========================
if page == "Accueil":
    st.title("Projet MLOps - Prédiction de défaut de crédit")

    st.markdown("""
    Cette application permet de prédire le risque de défaut d’un client  
    à partir de ses informations financières.

    📊 Modèle utilisé : **XGBoost (meilleur modèle via MLflow)** 👉 Utilisez la sidebar pour accéder à la prédiction.
    """)

    
    st.markdown("---")
    st.markdown("###  Classement officiel des modèles")
    
    try:
        
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Récupération des dossiers
        exp_log = mlflow.get_experiment_by_name("Modele_Regression_Logistique")
        exp_rf = mlflow.get_experiment_by_name("Modele_Random_Forest")
        exp_xgb = mlflow.get_experiment_by_name("Modele_XGBoost")
        
        liste_id = []
        if exp_log: liste_id.append(exp_log.experiment_id)
        if exp_rf: liste_id.append(exp_rf.experiment_id)
        if exp_xgb: liste_id.append(exp_xgb.experiment_id)
        
        if liste_id:
            runs = mlflow.search_runs(experiment_ids=liste_id)
            runs = runs.dropna(subset=["metrics.f1_score"])
            
            colonnes_a_garder = {
                "tags.mlflow.runName": "Nom du Modèle",
                "metrics.accuracy": "Accuracy",
                "metrics.recall": "Recall",
                "metrics.precision": "Precision",
                "metrics.f1_score": "F1-Score"
            }
            
            tableau = runs[list(colonnes_a_garder.keys())].rename(columns=colonnes_a_garder)
            tableau = tableau.sort_values(by="F1-Score", ascending=False).round(4)
            tableau = tableau.reset_index(drop=True)
            tableau.index += 1
            
            # Affichage graphique du tableau dans Streamlit
            st.dataframe(tableau, use_container_width=True)
        else:
            st.info("Aucune donnée d'entraînement trouvée.")
    except Exception as e:
        st.warning(f"Impossible de charger l'historique MLflow. Erreur : {e}")
    # ----------------------------------------

# =========================
# PAGE PREDICTION
# =========================
elif page == "Prédiction":

    st.title("📊 Prédiction de défaut de crédit")
    st.markdown("### 📋 Informations client")

    # Charger modèle
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

   
    col1, col2 = st.columns(2)

    with col1:
        loan_amt_outstanding = st.number_input("Montant du prêt restant", value=5000.0)
        income = st.number_input("Revenu", value=30000.0)
        years_employed = st.number_input("Années d'emploi", value=5.0)
        fico_score = st.number_input("FICO score", value=650.0)

        predict_button = st.button("Prédire", use_container_width=True)

    with col2:
        st.markdown("### 📈 Résultat")

        if predict_button:

            input_df = pd.DataFrame([{
                "loan_amt_outstanding": loan_amt_outstanding,
                "income": income,
                "years_employed": years_employed,
                "fico_score": fico_score
            }])

            input_scaled = scaler.transform(input_df)

            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1]

            st.write(f"Probabilité de défaut : **{proba:.2%}**")

            if prediction == 1:
                st.error("⚠️ Risque de défaut élevé")
            else:
                st.success("✅ Faible risque de défaut")