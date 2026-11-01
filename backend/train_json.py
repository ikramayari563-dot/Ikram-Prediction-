import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import json

def train_on_json():
    json_path = "data/creditcard_with_users.json"
    if not os.path.exists(json_path):
        print(f"Erreur : {json_path} non trouvé.")
        return
    
    print("Chargement des données JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Prétraitement
    # Les colonnes V1-V28 sont déjà normalisées (PCA), on garde Time et Amount
    features = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    X = df[features]
    y = df['Class']
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE
    print("Application de SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # On utilise Random Forest comme modèle principal car performant
    print("Entraînement du modèle Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)
    
    y_pred = model.predict(X_test)
    print("\nRésultats :")
    print(classification_report(y_test, y_pred))
    
    # Sauvegarde
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rf_json_model.joblib")
    
    # On sauvegarde aussi un scaler pour Time et Amount au cas où
    scaler = StandardScaler()
    scaler.fit(df[['Time', 'Amount']])
    joblib.dump(scaler, "models/scaler_json.joblib")
    
    print("Modèle JSON sauvegardé.")

if __name__ == "__main__":
    train_on_json()
