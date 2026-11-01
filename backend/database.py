from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "fraud_detection_db"

class Database:
    client: AsyncIOMotorClient = None
    db = None

db = Database()

async def connect_to_mongo():
    try:
        db.client = AsyncIOMotorClient(MONGO_URL)
        db.db = db.client[DB_NAME]
        # Vérifier la connexion
        await db.client.admin.command('ping')
        print("Connecté à MongoDB avec succès.")
    except Exception as e:
        print(f"Erreur de connexion à MongoDB : {e}")
        print("Le système fonctionnera en mode dégradé (sans base de données persistante).")
        db.db = None

async def close_mongo_connection():
    if db.client:
        db.client.close()
        print("Connexion MongoDB fermée.")

async def save_transaction(transaction_data):
    if db.db is not None:
        # On insère une copie : insert_one ajoute un _id (ObjectId) au document
        await db.db.transactions.insert_one(dict(transaction_data))
        return True
    # Fallback : log ou stockage local simple (optionnel)
    return False

async def get_all_transactions(search_query=None):
    if db.db is not None:
        query = {}
        if search_query and search_query.strip():
            term = search_query.strip()
            print(f"Recherche MongoDB avec le terme : '{term}'")
            # On cherche si le terme est contenu dans l'un des champs clés
            # Utilisation de $or avec vérification du type string pour éviter les erreurs sur les champs null
            query = {
                "$or": [
                    {"transaction_id": {"$regex": term, "$options": "i"}},
                    {"user_id": {"$regex": term, "$options": "i"}},
                    {"first_name": {"$regex": term, "$options": "i"}},
                    {"last_name": {"$regex": term, "$options": "i"}},
                    {"risk_level": {"$regex": term, "$options": "i"}}
                ]
            }
        
        # On trie par _id décroissant pour avoir les derniers ajouts en haut
        cursor = db.db.transactions.find(query).sort([("_id", -1)])
        results = await cursor.to_list(length=1000)
        print(f"Nombre de résultats trouvés : {len(results)}")
        return results
    return []

async def get_stats():
    if db.db is not None:
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "total_amount": {"$sum": "$Amount"},
                    "avg_amount": {"$avg": "$Amount"},
                    "frauds": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$or": [
                                        {"$eq": ["$prediction", 1]},
                                        {"$eq": ["$is_fraud", 1]},
                                        {"$eq": ["$Class", 1]},
                                        {"$eq": ["$fraud", 1]}
                                    ]
                                },
                                1,
                                0
                            ]
                        }
                    }
                }
            }
        ]
        cursor = db.db.transactions.aggregate(pipeline)
        results = await cursor.to_list(length=1)
        if results:
            res = results[0]
            return {
                "total": res["total"],
                "frauds": res["frauds"],
                "total_amount": res.get("total_amount", 0),
                "avg_amount": res.get("avg_amount", 0),
                "fraud_rate": (res["frauds"] / res["total"] * 100) if res["total"] > 0 else 0
            }
    return {"total": 0, "frauds": 0, "total_amount": 0, "avg_amount": 0, "fraud_rate": 0}

async def get_chart_data_from_db():
    if db.db is None:
        return {}
    
    # 1. Distribution des montants
    amount_pipeline = [
        {
            "$bucket": {
                "groupBy": "$Amount",
                "boundaries": [0, 10, 50, 100, 500, 1000, 5000, 100000],
                "default": "5000+",
                "output": {"count": {"$sum": 1}}
            }
        }
    ]
    amount_cursor = db.db.transactions.aggregate(amount_pipeline)
    amount_results = await amount_cursor.to_list(length=10)
    
    labels = {0: '0-10', 10: '10-50', 50: '50-100', 100: '100-500', 500: '500-1000', 1000: '1000-5000', 5000: '5000+'}
    amount_dist = {labels.get(r["_id"], str(r["_id"])): r["count"] for r in amount_results}

    # 2. Fraudes par heure
    hour_pipeline = [
        {
            "$project": {
                "hour": {"$mod": [{"$floor": {"$divide": ["$Time", 3600]}}, 24]},
                "is_fraud": {
                    "$or": [
                        {"$eq": ["$prediction", 1]},
                        {"$eq": ["$is_fraud", 1]},
                        {"$eq": ["$Class", 1]},
                        {"$eq": ["$fraud", 1]}
                    ]
                }
            }
        },
        {"$match": {"is_fraud": True}},
        {"$group": {"_id": "$hour", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    hour_cursor = db.db.transactions.aggregate(hour_pipeline)
    hour_results = await hour_cursor.to_list(length=24)
    hourly_fraud = {str(int(r["_id"])): r["count"] for r in hour_results}

    return {
        "amount_dist": amount_dist,
        "hourly_fraud": hourly_fraud
    }

async def get_user_info(user_id):
    """Recherche les informations d'un utilisateur dans la collection transactions."""
    if db.db is None:
        return None
    # On cherche la transaction la plus récente avec un nom pour cet utilisateur
    user = await db.db.transactions.find_one(
        {"user_id": user_id, "first_name": {"$ne": None}},
        sort=[("_id", -1)]
    )
    if user:
        return {
            "first_name": user.get("first_name"),
            "last_name": user.get("last_name")
        }
    return None

# ---------- Utilisateurs (comptes) ----------

async def create_user(user_doc):
    if db.db is None:
        return False
    await db.db.users.insert_one(dict(user_doc))
    return True

async def get_user_by_email(email):
    if db.db is None:
        return None
    return await db.db.users.find_one({"email": email})

async def get_user_by_username(username):
    if db.db is None:
        return None
    return await db.db.users.find_one({"username": username})

# ---------- Sessions (jetons de connexion) ----------

async def save_session(session_doc):
    if db.db is None:
        return False
    await db.db.sessions.insert_one(dict(session_doc))
    return True

async def get_session(token):
    if db.db is None:
        return None
    return await db.db.sessions.find_one({"token": token})

async def delete_session(token):
    if db.db is None:
        return False
    await db.db.sessions.delete_one({"token": token})
    return True

# ---------- Résultats des modèles ----------

async def save_model_results(results):
    """Remplace les résultats de comparaison des modèles stockés en base."""
    if db.db is None:
        return False
    await db.db.model_results.delete_many({})
    if results:
        await db.db.model_results.insert_many([dict(r) for r in results])
    return True

async def get_model_results():
    if db.db is None:
        return []
    cursor = db.db.model_results.find({}, {"_id": 0})
    return await cursor.to_list(length=100)

# ---------- Profils utilisateurs (Data Mining) ----------

async def save_user_profiles(profiles):
    """Remplace les profils utilisateurs (clustering) stockés en base."""
    if db.db is None:
        return False
    await db.db.user_profiles.delete_many({})
    if profiles:
        await db.db.user_profiles.insert_many([dict(p) for p in profiles])
    return True

async def get_cluster_distribution():
    """Répartition des utilisateurs par cluster (agrégation MongoDB)."""
    if db.db is None:
        return {}
    pipeline = [
        {"$match": {"cluster": {"$ne": None}}},
        {"$group": {"_id": "$cluster", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    cursor = db.db.user_profiles.aggregate(pipeline)
    results = await cursor.to_list(length=100)
    return {str(int(r["_id"])): r["count"] for r in results if r["_id"] is not None}

async def get_anomalous_profiles(limit=10):
    """Utilisateurs atypiques détectés par le Data Mining."""
    if db.db is None:
        return []
    cursor = db.db.user_profiles.find({"is_anomaly": 1}, {"_id": 0}).limit(limit)
    return await cursor.to_list(length=limit)

async def save_cluster_centers(centers):
    """Remplace les centres de clusters stockés en base."""
    if db.db is None:
        return False
    await db.db.cluster_centers.delete_many({})
    if centers:
        await db.db.cluster_centers.insert_many([dict(c) for c in centers])
    return True

async def get_cluster_centers_from_db():
    if db.db is None:
        return []
    cursor = db.db.cluster_centers.find({}, {"_id": 0})
    return await cursor.to_list(length=100)
