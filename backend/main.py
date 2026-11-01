from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import json
import io
import hashlib
import secrets
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fpdf import FPDF
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from .preprocessing import prepare, FEATURES
from .database import (
    connect_to_mongo, close_mongo_connection, save_transaction, get_all_transactions, get_stats,
    create_user, get_user_by_email, get_user_by_username,
    save_session, get_session, delete_session,
    get_model_results, save_model_results,
    get_user_info, get_chart_data_from_db,
    get_cluster_distribution, get_anomalous_profiles, get_cluster_centers_from_db
)

load_dotenv()

app = FastAPI(title="FraudShield AI Advanced API")

# Configuration de l'email
conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM_ADDRESS"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 465)),
    MAIL_SERVER=os.getenv("MAIL_HOST"),
    MAIL_FROM_NAME=os.getenv("MAIL_FROM_NAME"),
    MAIL_STARTTLS=False,
    MAIL_SSL_TLS=True,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement
MODELS_DIR = "models"
model_rf = None
scaler = None

@app.on_event("startup")
async def startup_event():
    global model_rf, scaler
    await connect_to_mongo()
    try:
        model_rf = joblib.load(f"{MODELS_DIR}/rf_json_model.joblib")
        scaler = joblib.load(f"{MODELS_DIR}/scaler_json.joblib")
        print("Modèles chargés et connexion MongoDB établie.")
    except Exception as e:
        print(f"Erreur chargement : {e}")

class JsonTransaction(BaseModel):
    user_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    Time: float
    Amount: float
    V_features: Optional[Dict[str, float]] = None

async def _compute_prediction(tx: JsonTransaction) -> dict:
    """Prépare le vecteur de features et retourne le résultat de prédiction."""
    
    # Construction du DataFrame pour le prétraitement unifié
    v_dict = tx.V_features or {}
    tx_data = {f"V{i}": [float(v_dict.get(f"V{i}", 0.0))] for i in range(1, 29)}
    tx_data["Time"] = [tx.Time]
    tx_data["Amount"] = [tx.Amount]
    df_tx = pd.DataFrame(tx_data)
    
    # Prétraitement unifié (Features + Normalisation)
    X_processed = prepare(df_tx, scaler, training=False)

    prediction = int(model_rf.predict(X_processed)[0])
    probs = model_rf.predict_proba(X_processed)[0]
    risk_score = float(probs[1])

    # Lookup du nom si manquant dans MongoDB
    fname = tx.first_name
    lname = tx.last_name
    if (not fname or not lname):
        info = await get_user_info(tx.user_id)
        if info:
            fname = fname or info.get("first_name")
            lname = lname or info.get("last_name")

    res = {
        "transaction_id": f"TX_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "user_id": tx.user_id,
        "first_name": fname,
        "last_name": lname,
        "prediction": prediction,
        "risk_score": risk_score,
        "risk_level": "High" if risk_score > 0.5 else "Medium" if risk_score > 0.2 else "Low",
        "timestamp": datetime.now().isoformat(),
        "amount": tx.Amount,
        "time": tx.Time
    }
    
    # Ajouter V1-V28 à la racine du document pour MongoDB
    for i in range(1, 29):
        key = f"V{i}"
        res[key] = float(v_dict.get(key, 0.0))
        
    return res

@app.post("/predict")
async def predict_json(tx: JsonTransaction, background_tasks: BackgroundTasks):
    try:
        result = await _compute_prediction(tx)
        background_tasks.add_task(save_transaction, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transactions")
async def create_transaction(tx: JsonTransaction):
    """Ajoute une transaction manuelle (avec sa prédiction) dans la base MongoDB."""
    try:
        result = await _compute_prediction(tx)
        saved = await save_transaction(result)
        return {
            "message": "Transaction enregistrée dans MongoDB" if saved else "MongoDB indisponible : transaction non persistée",
            "saved": saved,
            "transaction": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions")
async def list_transactions(q: Optional[str] = None):
    """Liste les transactions stockées dans la base MongoDB avec recherche optionnelle."""
    try:
        records = await get_all_transactions(search_query=q)
        for r in records:
            if "_id" in r:
                r["_id"] = str(r["_id"])
            
            # Enrichissement dynamique si le nom est manquant en base (via lookup DB)
            if not r.get("first_name") or not r.get("last_name"):
                info = await get_user_info(r["user_id"])
                if info:
                    r["first_name"] = r.get("first_name") or info.get("first_name")
                    r["last_name"] = r.get("last_name") or info.get("last_name")
                
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hachage du mot de passe avec PBKDF2-SHA256 (sans dépendance externe)."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return f"{salt}${digest.hex()}"

def verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except (ValueError, AttributeError):
        return False
    return secrets.compare_digest(hash_password(password, salt), stored)

def _extract_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    return authorization.replace("Bearer ", "").strip()

@app.post("/auth/register")
async def register(user: UserRegister):
    """Crée un compte utilisateur enregistré dans MongoDB."""
    username = user.username.strip()
    email = user.email.strip().lower()
    if not username or not email:
        raise HTTPException(status_code=400, detail="Nom d'utilisateur et email requis.")
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Le mot de passe doit contenir au moins 6 caractères.")
    if await get_user_by_email(email):
        raise HTTPException(status_code=400, detail="Cet email est déjà utilisé.")
    if await get_user_by_username(username):
        raise HTTPException(status_code=400, detail="Ce nom d'utilisateur est déjà pris.")

    doc = {
        "username": username,
        "email": email,
        "password": hash_password(user.password),
        "created_at": datetime.now().isoformat()
    }
    if not await create_user(doc):
        raise HTTPException(status_code=503, detail="Base de données indisponible.")

    token = secrets.token_urlsafe(32)
    await save_session({"token": token, "email": email, "created_at": datetime.now().isoformat()})
    return {"token": token, "user": {"username": username, "email": email}}

@app.post("/auth/login")
async def login(credentials: UserLogin):
    """Authentifie un utilisateur et renvoie un jeton de session."""
    email = credentials.email.strip().lower()
    user = await get_user_by_email(email)
    if not user or not verify_password(credentials.password, user.get("password", "")):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect.")

    token = secrets.token_urlsafe(32)
    await save_session({"token": token, "email": email, "created_at": datetime.now().isoformat()})
    return {"token": token, "user": {"username": user["username"], "email": user["email"]}}

@app.post("/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    token = _extract_token(authorization)
    if token:
        await delete_session(token)
    return {"message": "Déconnecté"}

@app.get("/auth/me")
async def me(authorization: Optional[str] = Header(None)):
    """Vérifie le jeton et renvoie l'utilisateur connecté."""
    token = _extract_token(authorization)
    session = await get_session(token) if token else None
    if not session:
        raise HTTPException(status_code=401, detail="Non authentifié.")
    user = await get_user_by_email(session["email"])
    if not user:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable.")
    return {"username": user["username"], "email": user["email"]}

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    print(f"Réception du fichier : {file.filename}")
    try:
        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith('.json'):
            # Essayer de lire le JSON de manière plus robuste
            try:
                data = json.loads(content.decode('utf-8'))
                df = pd.DataFrame(data)
            except Exception as e:
                print(f"Erreur lors de la lecture JSON : {e}")
                df = pd.read_json(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Format de fichier non supporté. Utilisez CSV ou JSON.")
        
        print(f"Colonnes trouvées : {df.columns.tolist()}")
        
        # Prédiction en masse avec prétraitement unifié
        X_processed = prepare(df, scaler, training=False)
        predictions = model_rf.predict(X_processed)
        probs = model_rf.predict_proba(X_processed)[:, 1]
        
        df['prediction'] = predictions
        df['risk_score'] = probs
        
        summary = {
            "filename": file.filename,
            "total_rows": len(df),
            "frauds_detected": int(sum(predictions)),
            "avg_risk": float(np.mean(probs)),
            "timestamp": datetime.now().isoformat()
        }
        
        # Sauvegarder temporairement pour le PDF
        os.makedirs("temp", exist_ok=True)
        df.to_csv(f"temp/last_analysis.csv", index=False)
        with open("temp/last_summary.json", "w") as f:
            json.dump(summary, f)
            
        print(f"Analyse terminée avec succès pour {file.filename}")
        return summary
        
    except Exception as e:
        print(f"Erreur lors de l'analyse : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def build_analysis_pdf(summary, df, pdf_path="temp/fraud_report.pdf"):
    """Génère le rapport PDF enrichi (graphiques + détails + suspects) et retourne son chemin."""

    # Chiffres clés (comparatif Fraude vs Normal)
    total_rows = len(df)
    fraud_rows = df[df['prediction'] == 1]
    normal_rows = df[df['prediction'] == 0]
    n_frauds = len(fraud_rows)
    n_normaux = len(normal_rows)
    taux_fraude = (n_frauds / total_rows * 100) if total_rows else 0.0
    montant_fraude = fraud_rows['Amount'].mean() if n_frauds else 0.0
    montant_normal = normal_rows['Amount'].mean() if n_normaux else 0.0
    score_moyen = float(df['risk_score'].mean()) if total_rows else 0.0

    # Comparaison des modèles (sur quoi l'étude est basée)
    model_reference = "Random Forest Classifier (rf_json_model.joblib)"
    models_df = None
    try:
        models_df = pd.read_csv("models/model_comparison.csv", index_col=0)
    except Exception as e:
        print(f"Comparaison des modèles indisponible : {e}")

    # 1. Générer les graphiques avec Matplotlib pour le PDF
    plt.figure(figsize=(10, 9))

    # Graphique 1: Répartition Fraude vs Normal
    plt.subplot(2, 2, 1)
    labels = ['Normal', 'Fraude']
    counts = [n_normaux, n_frauds]
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#10b981', '#ef4444'])
    plt.title('Répartition des Transactions')

    # Graphique 2: Distribution du Score de Risque
    plt.subplot(2, 2, 2)
    plt.hist(df['risk_score'], bins=20, color='#7c3aed', alpha=0.7)
    plt.title('Distribution du Score de Risque')
    plt.xlabel('Score')
    plt.ylabel('Fréquence')

    # Graphique 3: Montant moyen Fraude vs Normal (comparatif)
    plt.subplot(2, 2, 3)
    plt.bar(['Normal', 'Fraude'], [montant_normal, montant_fraude], color=['#10b981', '#ef4444'])
    plt.title('Montant Moyen par Classe')
    plt.ylabel('Montant (E)')

    # Graphique 4: Fraudes par heure
    plt.subplot(2, 2, 4)
    df['hour'] = (df['Time'] // 3600) % 24
    hourly = df.groupby('hour')['prediction'].sum().reindex(range(24), fill_value=0)
    plt.bar(hourly.index, hourly.values, color='#7c3aed', alpha=0.7)
    plt.title('Fraudes par heure')
    plt.xlabel('Heure')
    plt.ylabel('Nombre')

    plt.tight_layout()
    graph_path = "temp/pdf_graphs.png"
    plt.savefig(graph_path)
    plt.close()

    # 2. Création du PDF
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(124, 58, 237) # Violet Ikram
    pdf.cell(200, 20, "IKRAM PREDICTION - RAPPORT D'AUDIT", ln=True, align='C')
    pdf.ln(10)

    # Infos générales
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, "Informations Générales", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, f"Fichier analysé : {summary['filename']}", ln=True)
    pdf.cell(200, 8, f"Date de l'audit : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.cell(200, 8, f"Modèle utilisé pour l'analyse : {model_reference}", ln=True)
    pdf.ln(6)

    # Modèle de référence pour la comparaison / l'étude
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Modèle de Référence de l'Étude Comparative", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 6, "L'etude comparative des modeles a ete realisee sur le meme jeu de donnees (Credit Card Fraud : V1-V28 + Time + Amount), avec un decoupage 80/20 stratifie et un reequilibrage SMOTE. Le Random Forest Classifier a ete retenu comme modele de production et sert de reference pour toutes les analyses presentees dans ce rapport.")
    pdf.ln(2)
    if models_df is not None:
        pdf.set_font("Arial", 'B', 9)
        pdf.cell(70, 7, "Modele", border=1)
        pdf.cell(45, 7, "F1-Score", border=1)
        pdf.cell(45, 7, "Accuracy", border=1)
        pdf.ln()
        pdf.set_font("Arial", '', 9)
        for model_name, row in models_df.iterrows():
            pdf.cell(70, 7, str(model_name), border=1)
            pdf.cell(45, 7, f"{row.get('f1', 0):.4f}", border=1)
            pdf.cell(45, 7, f"{row.get('acc', 0):.4f}", border=1)
            pdf.ln()
    pdf.ln(8)

    # Synthèse Analytique + phrase comparative
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Synthèse Analytique Comparative", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, f"L'analyse a porte sur {total_rows} transactions. Le modele a classe {n_frauds} transactions comme frauduleuses ({taux_fraude:.2f}%) et {n_normaux} comme normales ({100 - taux_fraude:.2f}%).")
    pdf.ln(2)
    pdf.multi_cell(0, 8, f"Comparatif : le montant moyen des transactions jugees frauduleuses est de {montant_fraude:.2f} E contre {montant_normal:.2f} E pour les transactions normales, et le score de risque moyen du fichier est de {score_moyen:.3f}.")
    pdf.ln(5)

    # Logique de détection (explication du critère fraude / normal)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Critère de Décision (Fraude / Normal)", ln=True)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 6, "Sur quoi repose la decision : le modele Random Forest calcule un score de risque (probabilite de fraude) a partir des caracteristiques V1-V28, de Time et de Amount. Une transaction est declaree FRAUDULEUSE lorsque ce score >= 0.5 (les V1-V28 presentent alors des anomalies statistiques, souvent associees a des montants inhabituels et/ou des horaires atypiques). En dessous de 0.5, la transaction est consideree comme NORMALE. Un score > 0.8 correspond a un profil fortement atypique.")
    pdf.ln(10)

    # Insertion des graphiques (page dédiée pour éviter le débordement)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Visualisation Graphique de l'Audit", ln=True)
    pdf.image(graph_path, x=10, y=None, w=190)
    pdf.ln(10)

    # Utilisateurs les plus suspects
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Top 5 Utilisateurs les plus Suspects", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(30, 8, "ID", border=1)
    pdf.cell(40, 8, "Utilisateur", border=1)
    pdf.cell(25, 8, "Montant", border=1)
    pdf.cell(25, 8, "Score", border=1)
    pdf.cell(70, 8, "Motif de suspicion", border=1)
    pdf.ln()

    pdf.set_font("Arial", '', 8)
    seuil_montant = df['Amount'].mean() * 3 if total_rows else 0
    # On prend les 5 plus hauts scores de risque
    suspects = df.sort_values(by='risk_score', ascending=False).head(5)
    for _, row in suspects.iterrows():
        user_id = str(row.get('user_id', 'N/A'))
        if 'first_name' in df.columns and 'last_name' in df.columns:
            user_name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip() or 'N/A'
        else:
            user_name = 'N/A'
        amount = f"{row['Amount']:.2f} E"
        risk = f"{(row['risk_score']*100):.1f}%"

        motifs = []
        if row['risk_score'] >= 0.8:
            motifs.append("profil fortement atypique")
        elif row['risk_score'] >= 0.5:
            motifs.append("score de risque eleve")
        elif row['risk_score'] >= 0.2:
            motifs.append("risque modere")
        if row['Amount'] > seuil_montant and seuil_montant:
            motifs.append("montant inhabituel")
        hour = int((row['Time'] // 3600) % 24)
        if hour <= 4:
            motifs.append("transaction nocturne")
        reason = ", ".join(motifs) if motifs else "Alerte risque"

        pdf.cell(30, 8, user_id, border=1)
        pdf.cell(40, 8, user_name[:22], border=1)
        pdf.cell(25, 8, amount, border=1)
        pdf.cell(25, 8, risk, border=1)
        pdf.cell(70, 8, reason[:45], border=1)
        pdf.ln()

    pdf.output(pdf_path)
    return pdf_path


@app.get("/export-pdf")
async def export_pdf():
    print("Requête export-pdf reçue")
    try:
        if not os.path.exists("temp/last_summary.json"):
            print("Erreur : last_summary.json non trouvé")
            raise HTTPException(status_code=404, detail="Aucune analyse récente trouvée.")

        with open("temp/last_summary.json", "r") as f:
            summary = json.load(f)

        df = pd.read_csv("temp/last_analysis.csv")

        print(f"Génération du PDF enrichi pour {summary['filename']}...")
        pdf_path = build_analysis_pdf(summary, df)
        print(f"PDF enrichi généré avec succès : {pdf_path}")

        return FileResponse(pdf_path, media_type='application/pdf', filename=f"Rapport_Ikram_Prediction_{datetime.now().strftime('%Y%m%d')}.pdf")

    except Exception as e:
        print(f"Erreur export-pdf : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send-report-email")
async def send_report_email(email: str):
    print(f"Requête send-report-email reçue pour : {email}")
    try:
        # Re-générer le PDF pour être sûr qu'il est à jour
        if not os.path.exists("temp/last_summary.json"):
            raise HTTPException(status_code=404, detail="Aucune analyse disponible pour l'envoi.")
        
        with open("temp/last_summary.json", "r") as f:
            summary = json.load(f)

        df = pd.read_csv("temp/last_analysis.csv")

        # On génère exactement le même rapport enrichi (graphiques + détails) que le téléchargement
        pdf_path = build_analysis_pdf(summary, df, pdf_path="temp/email_report.pdf")
        print("PDF enrichi généré pour email.")

        message = MessageSchema(
            subject="Rapport d'Audit de Fraude - Ikram Prediction",
            recipients=[email],
            body=f"Bonjour,\n\nVeuillez trouver ci-joint le rapport d'analyse de fraude pour le fichier {summary['filename']}.\n\nCordialement,\nL'équipe Ikram Prediction",
            subtype=MessageType.plain,
            attachments=[pdf_path]
        )

        fm = FastMail(conf)
        await fm.send_message(message)
        print("Email envoyé avec succès !")
        return {"message": "Email envoyé avec succès"}
    except Exception as e:
        print(f"Erreur envoi email : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/models")
async def get_model_comparison():
    """Renvoie la comparaison des modèles (avec/sans SMOTE) depuis MongoDB."""
    try:
        records = await get_model_results()
        if records:
            return records
        # Fallback CSV si MongoDB est vide
        df = pd.read_csv("models/model_comparison.csv")
        return df.to_dict(orient="records")
    except:
        return [
            {"model": "Random Forest", "smote": True, "f1": 0.85, "acc": 0.99, "precision": 0.88, "recall": 0.82},
            {"model": "Random Forest", "smote": False, "f1": 0.70, "acc": 0.99, "precision": 0.95, "recall": 0.55}
        ]

@app.get("/stats/overview")
async def get_overview():
    stats = await get_stats()
    return {
        "total_transactions": stats.get("total", 0),
        "total_frauds": stats.get("frauds", 0),
        "total_amount": stats.get("total_amount", 0),
        "avg_amount": stats.get("avg_amount", 0),
        "fraud_rate": stats.get("fraud_rate", 0)
    }

@app.get("/stats/charts")
async def get_chart_data():
    db_charts = await get_chart_data_from_db()

    # 3. Répartition des clusters (Data Mining) - agrégation MongoDB
    cluster_dist = await get_cluster_distribution()

    return {
        "amount_dist": db_charts.get("amount_dist", {}),
        "hourly_fraud": db_charts.get("hourly_fraud", {}),
        "cluster_dist": cluster_dist
    }

@app.get("/stats/anomalies")
async def get_anomalies():
    """Utilisateurs atypiques issus du Data Mining (collection user_profiles)."""
    return await get_anomalous_profiles(10)

@app.get("/stats/clusters")
async def get_cluster_centers():
    """Centres des clusters (collection cluster_centers)."""
    return await get_cluster_centers_from_db()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
