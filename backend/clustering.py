import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import os

def perform_user_clustering(csv_path="data/transactions.csv"):
    if not os.path.exists(csv_path):
        print(f"Erreur : {csv_path} non trouvé.")
        return
    
    df = pd.read_csv(csv_path)
    
    # Agrégation par utilisateur pour obtenir le profil comportemental
    user_profiles = df.groupby('user_id').agg({
        'amount': ['mean', 'max', 'count'],
        'is_fraud': 'sum'
    }).reset_index()
    
    user_profiles.columns = ['user_id', 'avg_amount', 'max_amount', 'trans_count', 'fraud_count']
    
    # Préparation des données pour le clustering
    features = ['avg_amount', 'max_amount', 'trans_count']
    X = user_profiles[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means pour regrouper les comportements similaires
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    user_profiles['cluster'] = kmeans.fit_transform(X_scaled).argmin(axis=1) # Just assign a cluster
    user_profiles['cluster'] = kmeans.predict(X_scaled)
    
    # DBSCAN pour détecter les anomalies (points qui n'appartiennent à aucun cluster dense)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    user_profiles['dbscan_label'] = dbscan.fit_predict(X_scaled)
    
    # Les anomalies sont marquées -1 par DBSCAN
    user_profiles['is_anomaly'] = user_profiles['dbscan_label'].apply(lambda x: 1 if x == -1 else 0)
    
    # Sauvegarder les profils pour le dashboard
    user_profiles.to_csv("data/user_profiles.csv", index=False)
    
    print("Clustering terminé. Profils utilisateurs sauvegardés.")
    print(f"Nombre d'anomalies détectées par Data Mining : {user_profiles['is_anomaly'].sum()}")
    
    # Sauvegarder le modèle de clustering
    joblib.dump(kmeans, "models/kmeans_model.joblib")
    joblib.dump(scaler, "models/cluster_scaler.joblib")
    
    return user_profiles

if __name__ == "__main__":
    perform_user_clustering()
