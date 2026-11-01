import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_transactions(n_samples=10000, fraud_ratio=0.01):
    """
    Génère un dataset de transactions réalistes avec une proportion de fraudes.
    """
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # Configuration des utilisateurs et terminaux
    user_ids = [f"USER_{i:04d}" for i in range(1, 501)]
    terminal_ids = [f"TERM_{i:04d}" for i in range(1, 101)]
    locations = ["Paris", "Lyon", "Marseille", "Bordeaux", "Lille", "Nantes", "Strasbourg", "Toulouse"]
    trans_types = ["achat_en_ligne", "retrait_dab", "paiement_magasin", "transfert"]

    data = []

    # Génération des transactions normales
    for _ in range(n_normal):
        user = random.choice(user_ids)
        amount = round(random.uniform(5, 200), 2)
        timestamp = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        location = random.choice(locations)
        terminal = random.choice(terminal_ids)
        trans_type = random.choice(trans_types)
        
        data.append({
            "user_id": user,
            "amount": amount,
            "timestamp": timestamp,
            "location": location,
            "terminal_id": terminal,
            "transaction_type": trans_type,
            "is_fraud": 0
        })

    # Génération des transactions frauduleuses (caractéristiques inhabituelles)
    for _ in range(n_fraud):
        user = random.choice(user_ids)
        # Montants plus élevés ou inhabituels
        amount = round(random.uniform(500, 5000), 2)
        # Souvent la nuit
        timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
        timestamp = timestamp.replace(hour=random.randint(0, 4), minute=random.randint(0, 59))
        
        # Localisation inhabituelle ou lointaine (simulée ici par une liste restreinte)
        location = random.choice(["Etranger", "Paradis_Fiscal", "Inconnu"])
        terminal = f"TERM_FRAUD_{random.randint(1, 10)}"
        trans_type = "achat_en_ligne" # La fraude est plus fréquente en ligne
        
        data.append({
            "user_id": user,
            "amount": amount,
            "timestamp": timestamp,
            "location": location,
            "terminal_id": terminal,
            "transaction_type": trans_type,
            "is_fraud": 1
        })

    df = pd.DataFrame(data)
    # Mélanger les données
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Ajouter un ID de transaction
    df.insert(0, 'transaction_id', [f"TX_{i:06d}" for i in range(len(df))])
    
    return df

if __name__ == "__main__":
    print("Génération des données...")
    df = generate_transactions(n_samples=20000, fraud_ratio=0.02)
    
    # Créer le dossier data s'il n'existe pas
    os.makedirs("data", exist_ok=True)
    
    output_path = "data/transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset généré avec succès : {output_path}")
    print(f"Total transactions : {len(df)}")
    print(f"Fraudes : {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
