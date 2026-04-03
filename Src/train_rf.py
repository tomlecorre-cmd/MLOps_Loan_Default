import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from preprocessing import prepare_data

X_train, X_test, y_train, y_test = prepare_data()
mlflow.set_tracking_uri("http://127.0.0.1:8080")

mlflow.set_experiment("Modele_Random_Forest")

# Les 3 configurations à tester
tests_a_faire = [
    {"nom": "RF_50_Arbres_Petits", "n_estimators": 50, "max_depth": 5},
    {"nom": "RF_100_Arbres_Moyens", "n_estimators": 100, "max_depth": 10},
    {"nom": "RF_100_Arbres_Profonds", "n_estimators": 100, "max_depth": None}
]


#Boucle d'entraînement
for test in tests_a_faire:
    print(f"Lancement de : {test['nom']}")
    
    with mlflow.start_run(run_name=test['nom']):
        
        mlflow.log_param("n_estimators", test["n_estimators"])
        mlflow.log_param("max_depth", str(test["max_depth"]))
        
        # Création et entraînement
        model = RandomForestClassifier(
            n_estimators=test["n_estimators"], 
            max_depth=test["max_depth"], 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        }
        
    
        mlflow.log_metrics(metrics)
        
        # Sauvegarde du modèle
        mlflow.sklearn.log_model(model, "modele_rf")

print("Entraînement terminé")