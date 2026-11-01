import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from dotenv import load_dotenv

load_dotenv()


def save_profiles_to_mongo(user_profiles, centers):
    """Enregistre les profils utilisateurs et les centres de clusters dans MongoDB."""
    try:
        from pymongo import MongoClient
        client = MongoClient(os.getenv("MONGO_URL", "mongodb://localhost:27017"))
        mdb = client["fraud_detection_db"]

        def to_native(row):
            return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}

        mdb.user_profiles.delete_many({})
        mdb.user_profiles.insert_many([to_native(r) for r in user_profiles.to_dict(orient="records")])

        mdb.cluster_centers.delete_many({})
        mdb.cluster_centers.insert_many(
            [{k: float(v) for k, v in row.items()} for row in centers.to_dict(orient="records")]
        )
        client.close()
        print("Profils et centres enregistrés dans MongoDB (user_profiles / cluster_centers).")
    except Exception as e:
        print(f"MongoDB non disponible : {e}")


def update_clustering():
    json_path = "data/creditcard_with_users.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Profils utilisateurs basés sur le comportement de transaction
    user_profiles = df.groupby('user_id').agg({
        'Amount': ['mean', 'max', 'sum', 'count'],
        'Class': 'sum',
        'Time': 'std'
    }).reset_index()
    
    user_profiles.columns = ['user_id', 'avg_amount', 'max_amount', 'total_amount', 'trans_count', 'fraud_count', 'time_std']
    user_profiles = user_profiles.fillna(0)
    
    # Clustering sur les montants et la fréquence
    features = ['avg_amount', 'total_amount', 'trans_count']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_profiles[features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    user_profiles['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Identifier les anomalies par distance au centre du cluster
    distances = kmeans.transform(X_scaled)
    min_distances = np.min(distances, axis=1)
    threshold = np.percentile(min_distances, 95)
    user_profiles['is_anomaly'] = (min_distances > threshold).astype(int)
    
    user_profiles.to_csv("data/user_profiles_json.csv", index=False)
    print("Nouveaux profils utilisateurs générés.")
    
    # Sauvegarder les centres pour le radar chart
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
    centers.to_json("data/cluster_centers.json", orient="records")

    # Persistance MongoDB (source de vérité utilisée par l'API)
    save_profiles_to_mongo(user_profiles, centers)

if __name__ == "__main__":
    update_clustering()
