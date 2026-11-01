import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def preprocess_data(df):
    """
    Prépare les données pour l'entraînement.
    """
    # Copie du dataframe
    data = df.copy()
    
    # Extraction de caractéristiques temporelles
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    
    # Suppression des colonnes non informatives pour le modèle de base
    data = data.drop(['transaction_id', 'timestamp', 'user_id', 'terminal_id'], axis=1)
    
    # Encodage des variables catégorielles
    le_location = LabelEncoder()
    data['location'] = le_location.fit_transform(data['location'])
    
    le_type = LabelEncoder()
    data['transaction_type'] = le_type.fit_transform(data['transaction_type'])
    
    # Séparation X, y
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, le_location, le_type

def train_and_evaluate():
    # Charger les données
    if not os.path.exists("data/transactions.csv"):
        print("Erreur : data/transactions.csv non trouvé.")
        return
    
    df = pd.read_csv("data/transactions.csv")
    X, y, scaler, le_loc, le_type = preprocess_data(df)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Gérer le déséquilibre avec SMOTE
    print("Application de SMOTE pour équilibrer les classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, random_state=42),
        "Isolation Forest": IsolationForest(contamination=0.02, random_state=42)
    }
    
    results = {}
    os.makedirs("models", exist_ok=True)
    
    print("\n--- Évaluation des modèles ---")
    for name, model in models.items():
        print(f"Entraînement de {name}...")
        
        if name == "Isolation Forest":
            # Isolation Forest est non supervisé, on l'entraîne sur les données normales uniquement ou sur tout
            model.fit(X_train)
            y_pred = model.predict(X_test)
            # Isolation Forest retourne 1 pour normal et -1 pour anomalie
            y_pred = [1 if x == -1 else 0 for x in y_pred]
        else:
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"f1": f1, "acc": acc}
        
        print(f"{name} - F1-Score: {f1:.4f}, Accuracy: {acc:.4f}")
        
        # Sauvegarder le modèle
        joblib.dump(model, f"models/{name.lower().replace(' ', '_')}_model.joblib")

    # Créer un rapport de comparaison
    comparison_df = pd.DataFrame(results).T
    comparison_df.to_csv("models/model_comparison.csv")
    print("\nRapport de comparaison sauvegardé : models/model_comparison.csv")

    # Enregistrer aussi les résultats dans MongoDB (collection model_results)
    save_results_to_mongo(results)

    # Sauvegarder les outils de prétraitement
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(le_loc, "models/le_location.joblib")
    joblib.dump(le_type, "models/le_type.joblib")
    
    print("\nModèles et préprocesseurs sauvegardés dans le dossier 'models/'.")
    return results

def save_results_to_mongo(results):
    """Enregistre les résultats de comparaison dans la collection model_results."""
    try:
        from pymongo import MongoClient
        client = MongoClient(os.getenv("MONGO_URL", "mongodb://localhost:27017"))
        mdb = client["fraud_detection_db"]
        mdb.model_results.delete_many({})
        docs = [
            {"model": name, "f1": float(metrics["f1"]), "acc": float(metrics["acc"]),
             "trained_at": datetime.now().isoformat()}
            for name, metrics in results.items()
        ]
        mdb.model_results.insert_many(docs)
        client.close()
        print("Résultats enregistrés dans MongoDB -> fraud_detection_db.model_results")
    except Exception as e:
        print(f"MongoDB non disponible, résultats conservés uniquement en CSV : {e}")

if __name__ == "__main__":
    train_and_evaluate()
