import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from preprocessing import prepare_data

X_train, X_test, y_train, y_test = prepare_data()
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Modele_Regression_Logistique")

penalites_a_tester = [None, "l1", "l2"]

# On lance la boucle
for pen in penalites_a_tester:
    nom_du_run = f"Test_Penalite_{pen}"
    print(f"Lancement de : {nom_du_run}")
    
    with mlflow.start_run(run_name=nom_du_run):
        
        
        mlflow.log_param("penalty", str(pen))
        
        # On crée le modèle
        model = LogisticRegression(penalty=pen, solver="saga", max_iter=2000, random_state=42)
        model.fit(X_train, y_train)
        
        # On calcule les prédictions
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        }
        
        # On envoie le modèle à MLflow
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "modele")

print("C'est fini")