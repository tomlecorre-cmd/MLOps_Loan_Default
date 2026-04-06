import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:8080")

exp_log = mlflow.get_experiment_by_name("Modele_Regression_Logistique")
exp_rf = mlflow.get_experiment_by_name("Modele_Random_Forest")
exp_xgb = mlflow.get_experiment_by_name("Modele_XGBoost")

liste_id_experiences = []
if exp_log: liste_id_experiences.append(exp_log.experiment_id)
if exp_rf: liste_id_experiences.append(exp_rf.experiment_id)
if exp_xgb: liste_id_experiences.append(exp_xgb.experiment_id)

if not liste_id_experiences:
    print("Aucune expérience trouvée")
else:
    runs = mlflow.search_runs(experiment_ids=liste_id_experiences)

    if runs.empty:
        print("Aucun entraînement trouvé")
    else:
        runs = runs.dropna(subset=["metrics.f1_score"])

        colonnes_a_garder = {
            "tags.mlflow.runName": "Nom du Modèle",
            "metrics.accuracy": "Accuracy",
            "metrics.recall": "Recall",
            "metrics.precision": "Precision",
            "metrics.f1_score": "F1-Score"
        }

        tableau = runs[list(colonnes_a_garder.keys())].rename(columns=colonnes_a_garder)
        tableau = tableau.sort_values(by="F1-Score", ascending=False)
        tableau = tableau.round(4)
        tableau = tableau.reset_index(drop=True)
        tableau.index += 1

        print(" CLASSEMENT GÉNÉRAL DES MODÈLES :")
        print("-" * 80)
        print(tableau.to_string())
        print("-" * 80 + "\n")

        meilleur_run = runs.sort_values(by="metrics.f1_score", ascending=False).iloc[0]
        print(f" LE GRAND GAGNANT EST : {meilleur_run['tags.mlflow.runName']}")