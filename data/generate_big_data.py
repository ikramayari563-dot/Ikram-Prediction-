"""
Générateur Big Data : produits plusieurs millions de transactions.

- Génération vectorisée par lots (numpy) pour limiter la mémoire.
- Écriture incrémentale dans un CSV.
- Option d'insertion en base MongoDB (collection `transactions`) par lots.

Utilisation (depuis la racine du projet) :
    python -m data.generate_big_data --rows 5000000
    python -m data.generate_big_data --rows 2000000 --output data/transactions_big.csv --to-mongo
"""
import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

LOCATIONS = ["Paris", "Lyon", "Marseille", "Bordeaux", "Lille", "Nantes", "Strasbourg", "Toulouse"]
FRAUD_LOCATIONS = ["Etranger", "Paradis_Fiscal", "Inconnu"]
TRANS_TYPES = ["achat_en_ligne", "retrait_dab", "paiement_magasin", "transfert"]
FIRST_NAMES = ["Jean", "Marie", "Pierre", "Sophie", "Lucas", "Emma", "Thomas", "Julie", "Nicolas", "Sarah"]
LAST_NAMES = ["Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand", "Leroy", "Moreau"]
N_USERS = 10000
N_TERMINALS = 100


def build_chunk(rng, start_index, size, fraud_ratio, base_time):
    """Construit un lot de `size` transactions sous forme de DataFrame avec features V1-V28 et noms."""
    user_idx = rng.integers(1, N_USERS + 1, size=size)
    terminal_idx = rng.integers(1, N_TERMINALS + 1, size=size)
    loc_idx = rng.integers(0, len(LOCATIONS), size=size)
    type_idx = rng.integers(0, len(TRANS_TYPES), size=size)
    fname_idx = rng.integers(0, len(FIRST_NAMES), size=size)
    lname_idx = rng.integers(0, len(LAST_NAMES), size=size)

    is_fraud = (rng.random(size) < fraud_ratio).astype(np.int8)

    amount_normal = np.round(rng.gamma(shape=2.0, scale=60.0, size=size), 2)
    amount_fraud = np.round(rng.uniform(500, 20000, size=size), 2)
    amount = np.where(is_fraud == 1, amount_fraud, amount_normal)

    # Horodatage sur les 30 derniers jours
    offsets = rng.integers(0, 30 * 24 * 3600, size=size)
    timestamps = [base_time - timedelta(seconds=int(s)) for s in offsets]
    times = [float(s) for s in offsets] # Time relative

    locations = np.array(LOCATIONS)[loc_idx]
    fraud_loc = np.array(FRAUD_LOCATIONS)[rng.integers(0, len(FRAUD_LOCATIONS), size=size)]
    locations = np.where(is_fraud == 1, fraud_loc, locations)

    data = {
        "transaction_id": [f"TX_{i:09d}" for i in range(start_index, start_index + size)],
        "user_id": [f"USER_{u:05d}" for u in user_idx],
        "first_name": np.array(FIRST_NAMES)[fname_idx],
        "last_name": np.array(LAST_NAMES)[lname_idx],
        "amount": amount,
        "timestamp": timestamps,
        "time": times,
        "location": locations,
        "terminal_id": [f"TERM_{t:04d}" for t in terminal_idx],
        "transaction_type": np.array(TRANS_TYPES)[type_idx],
        "prediction": is_fraud, # Pour la cohérence UI
        "risk_score": np.where(is_fraud == 1, rng.uniform(0.7, 0.99, size), rng.uniform(0.0, 0.1, size))
    }

    # Ajouter V1-V28
    for i in range(1, 29):
        # Les fraudes ont souvent des distributions différentes sur certains V
        if i in [14, 17, 12, 10]: # Exemples de features discriminantes dans CreditCard PCA
            v_normal = rng.normal(loc=0.0, scale=1.0, size=size)
            v_fraud = rng.normal(loc=-2.0, scale=2.0, size=size)
            data[f"V{i}"] = np.where(is_fraud == 1, v_fraud, v_normal)
        else:
            data[f"V{i}"] = rng.normal(loc=0.0, scale=1.0, size=size)

    return pd.DataFrame(data)


def generate(n_rows, fraud_ratio, output, chunk_size=500000, to_mongo=False, seed=42):
    rng = np.random.default_rng(seed)
    base_time = datetime.now()

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    mongo_col = None
    if to_mongo:
        try:
            from pymongo import MongoClient
            from dotenv import load_dotenv
            load_dotenv()
            client = MongoClient(os.getenv("MONGO_URL", "mongodb://localhost:27017"))
            mongo_col = client["fraud_detection_db"]["transactions"]
            print("Insertion activée -> MongoDB (fraud_detection_db.transactions)")
        except Exception as e:
            print(f"MongoDB indisponible, génération CSV seulement : {e}")
            mongo_col = None

    total_frauds = 0
    first = True
    for start in range(0, n_rows, chunk_size):
        size = min(chunk_size, n_rows - start)
        df = build_chunk(rng, start, size, fraud_ratio, base_time)
        total_frauds += int(df["prediction"].sum())

        df.to_csv(output, mode="w" if first else "a", header=first, index=False)
        first = False

        if mongo_col is not None:
            records = df.to_dict(orient="records")
            for r in records:
                r["timestamp"] = str(r["timestamp"])
            mongo_col.insert_many(records)

        print(f"  {min(start + size, n_rows):>10,} / {n_rows:,} transactions...")

    print(f"\nTerminé : {n_rows:,} transactions écrites dans {output}")
    print(f"Fraudes : {total_frauds:,} ({total_frauds / n_rows * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère un gros volume de transactions.")
    parser.add_argument("--rows", type=int, default=5_000_000, help="Nombre de transactions à générer")
    parser.add_argument("--fraud-ratio", type=float, default=0.005, help="Proportion de fraudes (ex: 0.005 = 0.5%)")
    parser.add_argument("--output", type=str, default="data/transactions_big.csv", help="Fichier CSV de sortie")
    parser.add_argument("--chunk", type=int, default=500000, help="Taille des lots")
    parser.add_argument("--to-mongo", action="store_true", help="Insérer aussi dans MongoDB")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(args.rows, args.fraud_ratio, args.output, args.chunk, args.to_mongo, args.seed)
