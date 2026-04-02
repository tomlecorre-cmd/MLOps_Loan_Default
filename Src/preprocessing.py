import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def prepare_data():
    
    df = pd.read_csv('data/Loan_Data.csv')
    

    # On supprime  les données trop corrélées pour rendre le modèle réaliste
    colonnes_a_supprimer = ['customer_id', 'credit_lines_outstanding', 'total_debt_outstanding']
    
    for col in colonnes_a_supprimer:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    
    X = df.drop(columns=['default'])
    y = df['default']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #Sauvegarde du scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Les données sont prêtes.")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    prepare_data()