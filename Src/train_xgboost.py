import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import prepare_data
import joblib


X_train, X_test, y_train, y_test = prepare_data()

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Modele_XGBoost")

for max_depth in [3, 6]:
    nom_run = f"XGB_depth_{max_depth}"
    print(f"Lancement : {nom_run}")

    with mlflow.start_run(run_name=nom_run):

        model = XGBClassifier(
            max_depth=max_depth,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall",    recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score",  f1_score(y_test, y_pred))

        mlflow.sklearn.log_model(model, "model")
        print(f"  -> F1 : {f1_score(y_test, y_pred):.4f}")

print("Terminé !")


# Sauvegarder le meilleur modèle (celui avec max_depth=6)
joblib.dump(model, "models/xgb_model.pkl")