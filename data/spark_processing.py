"""
Traitement distribué (Big Data) des transactions avec Apache Spark.

Réalise : nettoyage, normalisation (min-max du montant), extraction de
caractéristiques (heure, nuit), agrégations (fraudes par localisation,
utilisateurs suspects) puis écriture des résultats.

Prérequis :
    pip install pyspark

Utilisation (depuis la racine du projet) :
    python -m data.spark_processing --input data/transactions_big.csv --output data/bigdata_output
"""
import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    parser = argparse.ArgumentParser(description="Traitement distribué des transactions avec Spark.")
    parser.add_argument("--input", default="data/transactions_big.csv", help="CSV généré par generate_big_data.py")
    parser.add_argument("--output", default="data/bigdata_output", help="Dossier de sortie des résultats")
    parser.add_argument("--master", default="local[*]", help="Master Spark (ex: spark://host:7077)")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("FraudDetectionBigData").master(args.master).getOrCreate()

    print(f"Lecture du fichier : {args.input}")
    df = spark.read.csv(args.input, header=True, inferSchema=True)

    total_initial = df.count()

    # 1. Nettoyage : suppression des valeurs manquantes et des doublons
    df = df.dropna().dropDuplicates(["transaction_id"])
    total_clean = df.count()
    print(f"Nettoyage : {total_initial:,} -> {total_clean:,} lignes ({(total_initial - total_clean):,} supprimées)")

    # 2. Normalisation min-max du montant
    bounds = df.select(F.min("amount").alias("mn"), F.max("amount").alias("mx")).first()
    mn, mx = bounds["mn"], bounds["mx"]
    if mx is not None and mx != mn:
        df = df.withColumn("amount_norm", (F.col("amount") - mn) / (mx - mn))
    else:
        df = df.withColumn("amount_norm", F.lit(0.0))

    # 3. Extraction de caractéristiques temporelles
    df = df.withColumn("hour", F.hour(F.to_timestamp("timestamp")))
    df = df.withColumn("is_night", F.when(F.col("hour") <= 4, 1).otherwise(0))

    # 4. Statistiques de fraude par localisation
    # On gère le nom de colonne 'prediction' (nouveau) ou 'is_fraud' (ancien)
    fraud_col = "prediction" if "prediction" in df.columns else "is_fraud"
    
    by_location = df.groupBy("location").agg(
        F.count("*").alias("total"),
        F.sum(fraud_col).alias("fraudes")
    ).withColumn("taux_fraude_pct", F.round(F.col("fraudes") / F.col("total") * 100, 3)) \
     .orderBy(F.col("taux_fraude_pct").desc())
    print("\n--- Fraudes par localisation ---")
    by_location.show(20, truncate=False)

    # 5. Utilisateurs suspects (comportement éloigné de la normale)
    suspects = df.groupBy("user_id").agg(
        F.count("*").alias("nb_transactions"),
        F.round(F.sum("amount"), 2).alias("montant_total"),
        F.sum(fraud_col).alias("nb_fraudes"),
        F.round(F.sum("amount_norm"), 3).alias("score_risque")
    ).orderBy(F.col("score_risque").desc()).limit(20)
    print("\n--- Top 20 utilisateurs suspects ---")
    suspects.show(20, truncate=False)

    # 6. Écriture des résultats
    os.makedirs(args.output, exist_ok=True)
    by_location.toPandas().to_csv(os.path.join(args.output, "fraudes_par_localisation.csv"), index=False)
    suspects.toPandas().to_csv(os.path.join(args.output, "utilisateurs_suspects.csv"), index=False)

    print(f"\nRésultats écrits dans {args.output}/")
    print("  - fraudes_par_localisation.csv")
    print("  - utilisateurs_suspects.csv")

    spark.stop()


if __name__ == "__main__":
    main()
