"""
Initialisation de la base MongoDB "fraud_detection_db".

Crée les collections, les index, et importe les résultats des modèles
depuis models/model_comparison.csv.

Utilisation (depuis la racine du projet) :
    python -m backend.init_db
"""
import os
import csv
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "fraud_detection_db"

COLLECTIONS = ["users", "sessions", "transactions", "model_results", "user_profiles", "cluster_centers"]


def main():
    client = MongoClient(MONGO_URL)
    database = client[DB_NAME]
    print(f"Connexion à MongoDB : {MONGO_URL} / base '{DB_NAME}'")

    existing = database.list_collection_names()

    # 1. Créer les collections manquantes
    for name in COLLECTIONS:
        if name not in existing:
            database.create_collection(name)
            print(f"  + Collection créée : {name}")
        else:
            print(f"  = Collection déjà présente : {name}")

    # 2. Créer les index
    database.users.create_index("email", unique=True)
    database.users.create_index("username", unique=True)
    database.sessions.create_index("token", unique=True)
    database.transactions.create_index("user_id")
    database.transactions.create_index("timestamp")
    print("  + Index créés (users.email, users.username, sessions.token, transactions.user_id, transactions.timestamp)")

    # 3. Importer les résultats des modèles 
    if database.model_results.count_documents({}) == 0:
        csv_path = os.path.join("models", "model_comparison.csv")
        if os.path.exists(csv_path):
            docs = []
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    docs.append({
                        "model": row.get("") or row.get("Model"),
                        "f1": float(row["f1"]),
                        "acc": float(row["acc"])
                    })
            if docs:
                database.model_results.insert_many(docs)
                print(f"  + {len(docs)} résultats de modèles importés depuis {csv_path}")
        else:
            print(f"  ! {csv_path} introuvable : lancez d'abord l'entraînement (backend/train_models.py)")
    else:
        print("  = model_results déjà alimenté")

    # 4. Récapitulatif
    print("\nRécapitulatif de la base :")
    for name in COLLECTIONS:
        print(f"  {name:15s} : {database[name].count_documents({})} document(s)")

    client.close()
    print("\nBase initialisée avec succès.")


if __name__ == "__main__":
    main()
