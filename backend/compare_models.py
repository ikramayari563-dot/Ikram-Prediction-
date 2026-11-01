"""
Comparaison des modèles (Decision Tree, Random Forest, KNN, SVM, Isolation Forest)
avec et sans SMOTE sur le dataset Kaggle.
"""
import os
import joblib
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from .preprocessing import prepare, fit_scaler, clean
from .database import connect_to_mongo, save_model_results

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

async def main():
    print("Démarrage de la comparaison des modèles...")
    
    # 1. Chargement des données (Kaggle dataset enrichi)
    json_path = "data/creditcard_with_users.json"
    if not os.path.exists(json_path):
        print(f"Erreur : {json_path} introuvable.")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data).sample(min(50000, len(data)), random_state=42) # Utiliser un échantillon pour la rapidité
    
    # 2. Prétraitement
    df_clean = clean(df, training=True)
    y = df_clean['Class']
    scaler = fit_scaler(df_clean)
    X = prepare(df_clean, scaler, training=True)
    
    # Sauvegarde du scaler pour l'API
    joblib.dump(scaler, f"{MODELS_DIR}/scaler_json.joblib")
    
    # 3. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. SMOTE (Rééquilibrage)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVM": SVC(kernel='rbf', probability=True, max_iter=1000, random_state=42)
    }
    
    results = []
    
    # Comparaison Classique vs SMOTE
    for name, model in models.items():
        print(f"Entraînement de {name}...")
        
        # Sans SMOTE
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            "model": name,
            "smote": False,
            "f1": f1_score(y_test, y_pred),
            "acc": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        })
        
        # Avec SMOTE
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)
        results.append({
            "model": name,
            "smote": True,
            "f1": f1_score(y_test, y_pred),
            "acc": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        })
        
        # Sauvegarder le meilleur modèle (Random Forest avec SMOTE par défaut)
        if name == "Random Forest":
            joblib.dump(model, f"{MODELS_DIR}/rf_json_model.joblib")

    # 5. Isolation Forest (Non supervisé)
    print("Entraînement de Isolation Forest...")
    iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    y_pred_iso = iso.predict(X_test)
    y_pred_iso = np.where(y_pred_iso == -1, 1, 0)
    results.append({
        "model": "Isolation Forest",
        "smote": False,
        "f1": f1_score(y_test, y_pred_iso),
        "acc": accuracy_score(y_test, y_pred_iso),
        "precision": precision_score(y_test, y_pred_iso),
        "recall": recall_score(y_test, y_pred_iso)
    })

    # 6. Sauvegarde dans MongoDB
    await connect_to_mongo()
    await save_model_results(results)
    print("Comparaison terminée et enregistrée dans MongoDB.")

    # Export CSV pour le dashboard
    pd.DataFrame(results).to_csv(f"{MODELS_DIR}/model_comparison.csv", index=False)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
