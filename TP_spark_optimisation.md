# TRAVAUX PRATIQUES - Optimisation Spark
## Master 2 Data Engineering

---

## Objectifs des TP

À l'issue de ces travaux pratiques, vous serez capable de :
- Diagnostiquer les problèmes de performance dans un job Spark
- Optimiser les partitions et gérer le data skew
- Configurer efficacement la mémoire et le cache
- Optimiser les jointures et les shuffles
- Appliquer les meilleures pratiques d'optimisation

---

##  Configuration de l'environnement

### Prérequis
```bash
# Installation PySpark
pip install pyspark==3.5.0 pandas numpy faker

# Variables d'environnement
export SPARK_HOME=/path/to/spark
export PYSPARK_PYTHON=python3
```

### Initialisation Spark
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Session Spark pour les TP
spark = SparkSession.builder \
    .appName("TP-Optimisation-Spark") \
    .master("local[4]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.default.parallelism", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
```

---

## TP1 : Génération de données et analyse initiale

### Objectif
Créer des datasets de test et comprendre leur structure avant optimisation.

### Exercice 1.1 : Génération de données

```python
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker('fr_FR')

def generate_customers(num_records):
    """Génère des données clients"""
    customers = []
    for i in range(num_records):
        customers.append({
            'customer_id': i,
            'name': fake.name(),
            'email': fake.email(),
            'city': fake.city(),
            'registration_date': fake.date_between(start_date='-2y', end_date='today'),
            'customer_segment': random.choice(['Premium', 'Standard', 'Basic', 'VIP'])
        })
    return customers

def generate_transactions(num_records, num_customers):
    """Génère des transactions avec skew intentionnel"""
    transactions = []
    
    # 20% des clients génèrent 80% des transactions (Pareto)
    popular_customers = list(range(int(num_customers * 0.2)))
    
    for i in range(num_records):
        # Créer du skew : 80% des transactions sur 20% des clients
        if random.random() < 0.8:
            customer_id = random.choice(popular_customers)
        else:
            customer_id = random.randint(0, num_customers - 1)
        
        transactions.append({
            'transaction_id': i,
            'customer_id': customer_id,
            'product_id': random.randint(1, 1000),
            'amount': round(random.uniform(10, 1000), 2),
            'timestamp': fake.date_time_between(start_date='-1y', end_date='now'),
            'status': random.choice(['completed', 'pending', 'cancelled']),
            'payment_method': random.choice(['card', 'paypal', 'bank_transfer'])
        })
    return transactions

def generate_products(num_records):
    """Génère un catalogue produits"""
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Food']
    products = []
    
    for i in range(1, num_records + 1):
        products.append({
            'product_id': i,
            'product_name': fake.catch_phrase(),
            'category': random.choice(categories),
            'price': round(random.uniform(5, 500), 2),
            'stock': random.randint(0, 1000)
        })
    return products

# Génération des datasets
print("Génération des données...")
customers_data = generate_customers(100000)
transactions_data = generate_transactions(1000000, 100000)
products_data = generate_products(1000)

# Création des DataFrames
df_customers = spark.createDataFrame(customers_data)
df_transactions = spark.createDataFrame(transactions_data)
df_products = spark.createDataFrame(products_data)

print(f"Customers: {df_customers.count()} lignes")
print(f"Transactions: {df_transactions.count()} lignes")
print(f"Products: {df_products.count()} lignes")
```

### Exercice 1.2 : Analyse exploratoire

```python
# 1. Schema et aperçu
print("=" * 50)
print("SCHEMA CUSTOMERS")
df_customers.printSchema()
df_customers.show(5)

print("\nSCHEMA TRANSACTIONS")
df_transactions.printSchema()
df_transactions.show(5)

# 2. Statistiques de base
print("\nSTATISTIQUES TRANSACTIONS")
df_transactions.describe(['amount']).show()

# 3. Distribution par segment client
print("\nDISTRIBUTION CLIENTS PAR SEGMENT")
df_customers.groupBy('customer_segment').count().orderBy('count', ascending=False).show()

# 4. Transactions par statut
print("\nTRANSACTIONS PAR STATUT")
df_transactions.groupBy('status').count().show()
```

### Exercice 1.3 : Détection du partitionnement initial

```python
# Vérifier le nombre de partitions
print(f"Partitions customers: {df_customers.rdd.getNumPartitions()}")
print(f"Partitions transactions: {df_transactions.rdd.getNumPartitions()}")
print(f"Partitions products: {df_products.rdd.getNumPartitions()}")

# Analyser la distribution des données par partition
from pyspark.sql.functions import spark_partition_id

partition_dist = df_transactions \
    .withColumn("partition_id", spark_partition_id()) \
    .groupBy("partition_id") \
    .count() \
    .orderBy("count", ascending=False)

print("\nDISTRIBUTION DES TRANSACTIONS PAR PARTITION")
partition_dist.show()

# Calculer le coefficient de variation (skew indicator)
stats = partition_dist.select(
    mean('count').alias('mean'),
    stddev('count').alias('stddev')
).collect()[0]

cv = stats['stddev'] / stats['mean'] if stats['mean'] > 0 else 0
print(f"\nCoefficient de variation: {cv:.2f}")
print("CV > 0.5 indique un déséquilibre significatif" if cv > 0.5 else "Distribution équilibrée")
```

### Questions TP1

1. Quel est le nombre de partitions par défaut ? Est-ce optimal pour vos données ?
2. Observez-vous du data skew dans les transactions ? Comment le quantifiez-vous ?
3. Quelle taille moyenne ont vos partitions ?

---

## 🔧 TP2 : Optimisation des partitions

### Objectif
Maîtriser les techniques de repartitionnement et résoudre le data skew.

### Exercice 2.1 : Repartitionnement basique

```python
import time

def measure_time(func, description):
    """Mesure le temps d'exécution"""
    start = time.time()
    result = func()
    end = time.time()
    print(f"{description}: {end - start:.2f} secondes")
    return result, end - start

# Scénario : Agrégation par customer_id
print("=" * 60)
print("BENCHMARK: Agrégation par customer_id")
print("=" * 60)

# Version 1 : Sans optimisation
def agg_v1():
    return df_transactions.groupBy("customer_id").agg(
        count("*").alias("nb_transactions"),
        sum("amount").alias("total_amount")
    ).count()

result1, time1 = measure_time(agg_v1, "V1 - Sans repartition")

# Version 2 : Avec repartition
def agg_v2():
    return df_transactions.repartition(16, "customer_id").groupBy("customer_id").agg(
        count("*").alias("nb_transactions"),
        sum("amount").alias("total_amount")
    ).count()

result2, time2 = measure_time(agg_v2, "V2 - Avec repartition(16)")

# Version 3 : Avec plus de partitions
def agg_v3():
    return df_transactions.repartition(32, "customer_id").groupBy("customer_id").agg(
        count("*").alias("nb_transactions"),
        sum("amount").alias("total_amount")
    ).count()

result3, time3 = measure_time(agg_v3, "V3 - Avec repartition(32)")

print(f"\nAmélioration V2 vs V1: {((time1 - time2) / time1) * 100:.1f}%")
print(f"Amélioration V3 vs V1: {((time1 - time3) / time1) * 100:.1f}%")
```

### Exercice 2.2 : Gestion du data skew avec salting

```python
# Identifier les clients "hot" (top 1% avec le plus de transactions)
hot_customers = df_transactions.groupBy("customer_id").count() \
    .orderBy(desc("count")) \
    .limit(1000) \
    .select("customer_id") \
    .rdd.flatMap(lambda x: x).collect()

print(f"Nombre de hot customers: {len(hot_customers)}")

# Technique 1 : Salting pour les hot customers
from pyspark.sql.functions import rand, when, concat, lit

df_transactions_salted = df_transactions.withColumn(
    "salt",
    when(col("customer_id").isin(hot_customers), 
         (rand() * 10).cast("int"))
    .otherwise(lit(0))
).withColumn(
    "customer_id_salted",
    concat(col("customer_id").cast("string"), lit("_"), col("salt").cast("string"))
)

# Agrégation avec salting
def agg_with_salting():
    # Phase 1 : Agrégation sur clé saltée
    salted_agg = df_transactions_salted.groupBy("customer_id_salted", "customer_id").agg(
        count("*").alias("nb_trans_partial"),
        sum("amount").alias("amount_partial")
    )
    
    # Phase 2 : Agrégation finale sur customer_id réel
    final_agg = salted_agg.groupBy("customer_id").agg(
        sum("nb_trans_partial").alias("nb_transactions"),
        sum("amount_partial").alias("total_amount")
    )
    
    return final_agg.count()

result_salt, time_salt = measure_time(agg_with_salting, "Avec salting")
print(f"Amélioration salting vs baseline: {((time1 - time_salt) / time1) * 100:.1f}%")

# Vérifier la distribution après salting
df_transactions_salted.groupBy("customer_id_salted") \
    .count() \
    .select(
        mean("count").alias("mean"),
        stddev("count").alias("stddev"),
        max("count").alias("max"),
        min("count").alias("min")
    ).show()
```

### Exercice 2.3 : Coalesce vs Repartition

```python
# Créer un DataFrame avec beaucoup de partitions
df_many_partitions = df_transactions.repartition(100)
print(f"Partitions initiales: {df_many_partitions.rdd.getNumPartitions()}")

# Test 1 : Coalesce (sans shuffle)
def test_coalesce():
    df_coalesced = df_many_partitions.coalesce(10)
    return df_coalesced.write.mode("overwrite").parquet("/tmp/coalesced")

time_coalesce = measure_time(test_coalesce, "Coalesce (10 partitions)")[1]

# Test 2 : Repartition (avec shuffle)
def test_repartition():
    df_repartitioned = df_many_partitions.repartition(10)
    return df_repartitioned.write.mode("overwrite").parquet("/tmp/repartitioned")

time_repartition = measure_time(test_repartition, "Repartition (10 partitions)")[1]

print(f"\nCoalesce est {time_repartition / time_coalesce:.1f}x plus rapide")
print("Mais attention : coalesce peut créer du déséquilibre !")
```

### Questions TP2

1. Quel est le nombre optimal de partitions pour vos données ?
2. Le salting améliore-t-il les performances ? De combien ?
3. Quand utiliser `coalesce()` vs `repartition()` ?

---

## TP3 : Cache et gestion mémoire

### Objectif
Utiliser efficacement le cache pour optimiser les pipelines avec réutilisation de données.

### Exercice 3.1 : Impact du cache

```python
# Pipeline avec multiples utilisations du même DataFrame
df_active_customers = df_customers.filter(col("customer_segment").isin(["Premium", "VIP"]))

# Scénario 1 : Sans cache
print("=" * 60)
print("SCÉNARIO 1 : SANS CACHE")
print("=" * 60)

def pipeline_no_cache():
    # Opération 1
    count1 = df_active_customers.count()
    
    # Opération 2
    by_city = df_active_customers.groupBy("city").count().count()
    
    # Opération 3
    by_segment = df_active_customers.groupBy("customer_segment").count().count()
    
    return count1, by_city, by_segment

result_no_cache, time_no_cache = measure_time(pipeline_no_cache, "Sans cache")

# Scénario 2 : Avec cache
print("\n" + "=" * 60)
print("SCÉNARIO 2 : AVEC CACHE")
print("=" * 60)

def pipeline_with_cache():
    df_cached = df_active_customers.cache()
    
    # Opération 1 (charge le cache)
    count1 = df_cached.count()
    
    # Opération 2 (utilise le cache)
    by_city = df_cached.groupBy("city").count().count()
    
    # Opération 3 (utilise le cache)
    by_segment = df_cached.groupBy("customer_segment").count().count()
    
    df_cached.unpersist()
    return count1, by_city, by_segment

result_with_cache, time_with_cache = measure_time(pipeline_with_cache, "Avec cache")

print(f"\nGain de performance: {((time_no_cache - time_with_cache) / time_no_cache) * 100:.1f}%")
```

### Exercice 3.2 : Niveaux de stockage

```python
from pyspark import StorageLevel

# Test différents niveaux de stockage
df_test = df_transactions.filter(col("amount") > 100)

# Niveau 1 : MEMORY_ONLY
df_test.persist(StorageLevel.MEMORY_ONLY)
time_mem_only = measure_time(
    lambda: df_test.groupBy("customer_id").count().count(),
    "MEMORY_ONLY"
)[1]
df_test.unpersist()

# Niveau 2 : MEMORY_AND_DISK
df_test.persist(StorageLevel.MEMORY_AND_DISK)
time_mem_disk = measure_time(
    lambda: df_test.groupBy("customer_id").count().count(),
    "MEMORY_AND_DISK"
)[1]
df_test.unpersist()

# Niveau 3 : MEMORY_ONLY_SER
df_test.persist(StorageLevel.MEMORY_ONLY_SER)
time_mem_ser = measure_time(
    lambda: df_test.groupBy("customer_id").count().count(),
    "MEMORY_ONLY_SER"
)[1]
df_test.unpersist()

print("\nComparaison des niveaux de stockage:")
print(f"MEMORY_ONLY: {time_mem_only:.2f}s")
print(f"MEMORY_AND_DISK: {time_mem_disk:.2f}s")
print(f"MEMORY_ONLY_SER: {time_mem_ser:.2f}s")
```

### Exercice 3.3 : Monitoring du cache

```python
# Fonction pour afficher l'utilisation du cache
def show_cache_stats():
    """Affiche les statistiques de cache via Spark UI"""
    # Note: En production, consulter http://localhost:4040/storage/
    print("\nPour voir les stats détaillées du cache:")
    print("1. Ouvrir http://localhost:4040 (Spark UI)")
    print("2. Aller dans l'onglet 'Storage'")
    print("3. Observer: Size in Memory, Size on Disk, Fraction Cached")

# Cache avec observation
df_to_monitor = df_transactions.filter(col("status") == "completed").cache()

# Forcer le cache
df_to_monitor.count()

show_cache_stats()

# Libérer
df_to_monitor.unpersist()
```

### Questions TP3

1. Quand le cache est-il vraiment bénéfique ?
2. Quel niveau de stockage choisir selon le use case ?
3. Comment surveiller l'utilisation de la mémoire ?

---

## 🔗 TP4 : Optimisation des jointures

### Objectif
Maîtriser les différents types de jointures et leurs optimisations.

### Exercice 4.1 : Broadcast join vs shuffle join

```python
# Créer une petite table de référence
df_products_small = df_products  # 1000 produits

# Test 1 : Join normal (shuffle join)
def normal_join():
    return df_transactions.join(df_products_small, "product_id").count()

time_normal, _ = measure_time(normal_join, "Normal Join (Shuffle)")

# Test 2 : Broadcast join explicite
from pyspark.sql.functions import broadcast

def broadcast_join():
    return df_transactions.join(broadcast(df_products_small), "product_id").count()

time_broadcast, _ = measure_time(broadcast_join, "Broadcast Join")

print(f"\nBroadcast join est {time_normal / time_broadcast:.1f}x plus rapide")

# Vérifier le plan d'exécution
print("\n" + "=" * 60)
print("PLAN D'EXÉCUTION - BROADCAST JOIN")
print("=" * 60)
df_transactions.join(broadcast(df_products_small), "product_id").explain()
```

### Exercice 4.2 : Configuration auto-broadcast

```python
# Tester différents seuils de broadcast
thresholds = [1024 * 1024, 10 * 1024 * 1024, 50 * 1024 * 1024]  # 1MB, 10MB, 50MB

for threshold in thresholds:
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", threshold)
    
    def join_with_threshold():
        return df_transactions.join(df_products, "product_id").count()
    
    time_taken = measure_time(
        join_with_threshold,
        f"Auto-broadcast threshold: {threshold / (1024*1024):.0f}MB"
    )[1]
```

### Exercice 4.3 : Bucketing pour optimiser les joins récurrents

```python
# Écrire les tables avec bucketing
print("Création des tables bucketées...")

# Bucketing transactions
df_transactions.write \
    .bucketBy(20, "customer_id") \
    .sortBy("timestamp") \
    .mode("overwrite") \
    .saveAsTable("transactions_bucketed")

# Bucketing customers
df_customers.write \
    .bucketBy(20, "customer_id") \
    .sortBy("customer_id") \
    .mode("overwrite") \
    .saveAsTable("customers_bucketed")

# Lecture et join sans shuffle
df_trans_bucketed = spark.table("transactions_bucketed")
df_cust_bucketed = spark.table("customers_bucketed")

def bucketed_join():
    return df_trans_bucketed.join(df_cust_bucketed, "customer_id").count()

time_bucketed = measure_time(bucketed_join, "Join avec bucketing")[1]

# Comparer avec join normal
def non_bucketed_join():
    return df_transactions.join(df_customers, "customer_id").count()

time_non_bucketed = measure_time(non_bucketed_join, "Join sans bucketing")[1]

print(f"\nGain avec bucketing: {((time_non_bucketed - time_bucketed) / time_non_bucketed) * 100:.1f}%")

# Vérifier l'absence de shuffle dans le plan
print("\nPlan d'exécution (doit montrer 'buckets' et pas de 'Exchange'):")
df_trans_bucketed.join(df_cust_bucketed, "customer_id").explain()
```

### Exercice 4.4 : Skew join handling

```python
# Activer l'optimisation AQE pour skew joins
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")

# Join avec skew (beaucoup de transactions pour peu de clients)
def skewed_join():
    return df_transactions.join(df_customers, "customer_id") \
        .groupBy("customer_segment") \
        .agg(sum("amount").alias("total_revenue")) \
        .count()

time_aqe = measure_time(skewed_join, "Join avec AQE")[1]

# Désactiver AQE pour comparer
spark.conf.set("spark.sql.adaptive.enabled", "false")
time_no_aqe = measure_time(skewed_join, "Join sans AQE")[1]

print(f"\nAmélioration avec AQE: {((time_no_aqe - time_aqe) / time_no_aqe) * 100:.1f}%")

# Réactiver AQE
spark.conf.set("spark.sql.adaptive.enabled", "true")
```

### Questions TP4

1. Quand utiliser broadcast join vs shuffle join ?
2. Comment le bucketing améliore-t-il les performances ?
3. Quel impact a l'Adaptive Query Execution (AQE) ?

---

## TP5 : Formats de fichiers et I/O

### Objectif
Comparer les formats et optimiser les opérations de lecture/écriture.

### Exercice 5.1 : Comparaison CSV vs Parquet

```python
import os

# Écriture CSV
def write_csv():
    df_transactions.write.mode("overwrite").csv("/tmp/transactions.csv")

time_write_csv = measure_time(write_csv, "Écriture CSV")[1]

# Écriture Parquet
def write_parquet():
    df_transactions.write.mode("overwrite").parquet("/tmp/transactions.parquet")

time_write_parquet = measure_time(write_parquet, "Écriture Parquet")[1]

# Taille des fichiers
def get_folder_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)  # MB

size_csv = get_folder_size("/tmp/transactions.csv")
size_parquet = get_folder_size("/tmp/transactions.parquet")

print(f"\nTaille CSV: {size_csv:.2f} MB")
print(f"Taille Parquet: {size_parquet:.2f} MB")
print(f"Compression Parquet: {((size_csv - size_parquet) / size_csv) * 100:.1f}%")

# Lecture avec projection
def read_csv_projection():
    return spark.read.csv("/tmp/transactions.csv", header=True) \
        .select("transaction_id", "amount") \
        .filter(col("amount") > 100) \
        .count()

def read_parquet_projection():
    return spark.read.parquet("/tmp/transactions.parquet") \
        .select("transaction_id", "amount") \
        .filter(col("amount") > 100) \
        .count()

time_read_csv = measure_time(read_csv_projection, "Lecture CSV + projection")[1]
time_read_parquet = measure_time(read_parquet_projection, "Lecture Parquet + projection")[1]

print(f"\nParquet est {time_read_csv / time_read_parquet:.1f}x plus rapide en lecture")
```

### Exercice 5.2 : Partitionnement à l'écriture

```python
# Ajouter une colonne date pour partitionnement
df_trans_with_date = df_transactions.withColumn(
    "year", year(col("timestamp"))
).withColumn(
    "month", month(col("timestamp"))
)

# Écriture partitionnée
def write_partitioned():
    df_trans_with_date.write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet("/tmp/transactions_partitioned")

time_write_part = measure_time(write_partitioned, "Écriture partitionnée")[1]

# Lecture avec predicate pushdown
def read_partitioned():
    return spark.read.parquet("/tmp/transactions_partitioned") \
        .filter((col("year") == 2024) & (col("month") == 1)) \
        .count()

def read_non_partitioned():
    return spark.read.parquet("/tmp/transactions.parquet") \
        .filter((year(col("timestamp")) == 2024) & (month(col("timestamp")) == 1)) \
        .count()

time_part = measure_time(read_partitioned, "Lecture partitionnée")[1]
time_non_part = measure_time(read_non_partitioned, "Lecture non-partitionnée")[1]

print(f"\nPartitioning améliore la lecture de {((time_non_part - time_part) / time_non_part) * 100:.1f}%")
```

### Exercice 5.3 : Optimisation de la taille des fichiers

```python
# Problème : Trop de petits fichiers
df_trans_with_date.write \
    .mode("overwrite") \
    .partitionBy("year", "month", "status") \
    .parquet("/tmp/transactions_many_small_files")

# Compter les fichiers
import glob
num_files = len(glob.glob("/tmp/transactions_many_small_files/**/*.parquet", recursive=True))
print(f"Nombre de fichiers créés: {num_files}")

# Solution : Repartition avant écriture
def write_optimized():
    df_trans_with_date \
        .repartition(10, "year", "month") \
        .write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet("/tmp/transactions_optimized")

measure_time(write_optimized, "Écriture avec repartition")[1]

num_files_opt = len(glob.glob("/tmp/transactions_optimized/**/*.parquet", recursive=True))
print(f"Nombre de fichiers optimisés: {num_files_opt}")
print(f"Réduction: {((num_files - num_files_opt) / num_files) * 100:.1f}%")
```

### Questions TP5

1. Pourquoi Parquet est-il plus performant que CSV ?
2. Comment le partitionnement améliore-t-il les performances ?
3. Quel est l'impact des petits fichiers ?

---

## TP6 : UDFs et optimisations Catalyst

### Objectif
Comprendre l'impact des UDFs et utiliser les fonctions natives Spark.

### Exercice 6.1 : UDF Python vs fonctions natives

```python
from pyspark.sql.types import StringType

# Créer une UDF Python classique
@udf(returnType=StringType())
def categorize_amount_udf(amount):
    if amount < 50:
        return "Low"
    elif amount < 200:
        return "Medium"
    else:
        return "High"

# Version native Spark
def categorize_amount_native(df):
    return df.withColumn(
        "category",
        when(col("amount") < 50, "Low")
        .when(col("amount") < 200, "Medium")
        .otherwise("High")
    )

# Benchmark UDF
def test_udf():
    return df_transactions.withColumn(
        "category", categorize_amount_udf(col("amount"))
    ).count()

time_udf = measure_time(test_udf, "UDF Python")[1]

# Benchmark fonction native
def test_native():
    return categorize_amount_native(df_transactions).count()

time_native = measure_time(test_native, "Fonction native")[1]

print(f"\nFonction native est {time_udf / time_native:.1f}x plus rapide")

# Analyser le plan d'exécution
print("\n" + "=" * 60)
print("Plan avec UDF (pas d'optimisation Catalyst):")
df_transactions.withColumn("category", categorize_amount_udf(col("amount"))).explain()

print("\n" + "=" * 60)
print("Plan avec fonction native (optimisé par Catalyst):")
categorize_amount_native(df_transactions).explain()
```

### Exercice 6.2 : Pandas UDF (Vectorized UDF)

```python
from pyspark.sql.functions import pandas_udf
import pandas as pd

# Pandas UDF - Vectorisé
@pandas_udf(StringType())
def categorize_amount_pandas(amounts: pd.Series) -> pd.Series:
    return amounts.apply(lambda x: "Low" if x < 50 else ("Medium" if x < 200 else "High"))

# Benchmark Pandas UDF
def test_pandas_udf():
    return df_transactions.withColumn(
        "category", categorize_amount_pandas(col("amount"))
    ).count()

time_pandas_udf = measure_time(test_pandas_udf, "Pandas UDF")[1]

print("\nComparaison des performances:")
print(f"UDF Python: {time_udf:.2f}s (baseline)")
print(f"Pandas UDF: {time_pandas_udf:.2f}s ({time_udf / time_pandas_udf:.1f}x plus rapide)")
print(f"Fonction native: {time_native:.2f}s ({time_udf / time_native:.1f}x plus rapide)")
```

### Exercice 6.3 : Predicate pushdown et projection

```python
# Lire un Parquet avec explain pour voir les optimisations
df_parquet = spark.read.parquet("/tmp/transactions.parquet")

# Sans optimisations
print("=" * 60)
print("SANS OPTIMISATIONS (Select * puis filter)")
print("=" * 60)
df_parquet.select("*").filter(col("amount") > 500).explain()

# Avec optimisations
print("\n" + "=" * 60)
print("AVEC OPTIMISATIONS (Projection + Filter)")
print("=" * 60)
df_parquet.select("transaction_id", "customer_id", "amount") \
    .filter(col("amount") > 500) \
    .explain()

# Mesurer l'impact
def without_optimization():
    return df_parquet.select("*").filter(col("amount") > 500).count()

def with_optimization():
    return df_parquet.select("transaction_id", "customer_id", "amount") \
        .filter(col("amount") > 500).count()

time_without = measure_time(without_optimization, "Sans projection")[1]
time_with = measure_time(with_optimization, "Avec projection")[1]

print(f"\nAmélioration: {((time_without - time_with) / time_without) * 100:.1f}%")
```

### Questions TP6

1. Pourquoi les UDFs Python sont-elles lentes ?
2. Quand utiliser des Pandas UDFs ?
3. Comment fonctionne le predicate pushdown ?

---

## TP7 : Projet final - Pipeline complet optimisé

### Objectif
Construire un pipeline ETL complet en appliquant toutes les optimisations apprises.

### Contexte
Vous devez créer un rapport d'analyse des ventes qui :
1. Joint les transactions, clients et produits
2. Calcule des KPIs par segment client et catégorie produit
3. Identifie les clients VIP (top 10% revenue)
4. Sauvegarde les résultats en Parquet partitionné

### Exercice 7.1 : Pipeline non optimisé (baseline)

```python
def pipeline_baseline():
    """Pipeline sans optimisations - BASELINE"""
    
    # Join complet
    df_full = df_transactions \
        .join(df_customers, "customer_id") \
        .join(df_products, "product_id")
    
    # KPIs par segment et catégorie
    kpis = df_full.groupBy("customer_segment", "category").agg(
        count("*").alias("nb_transactions"),
        sum("amount").alias("total_revenue"),
        avg("amount").alias("avg_transaction"),
        countDistinct("customer_id").alias("unique_customers")
    )
    
    # Top clients
    top_customers = df_full.groupBy("customer_id", "name").agg(
        sum("amount").alias("total_revenue")
    ).orderBy(desc("total_revenue")).limit(1000)
    
    # Sauvegardes
    kpis.write.mode("overwrite").parquet("/tmp/kpis_baseline")
    top_customers.write.mode("overwrite").parquet("/tmp/top_customers_baseline")
    
    return kpis.count(), top_customers.count()

time_baseline = measure_time(pipeline_baseline, "Pipeline BASELINE")[1]
```

### Exercice 7.2 : Pipeline optimisé

```python
def pipeline_optimized():
    """Pipeline avec toutes les optimisations"""
    
    # 1. OPTIMISATION: Broadcast join pour products (petite table)
    df_trans_products = df_transactions.join(
        broadcast(df_products), "product_id"
    )
    
    # 2. OPTIMISATION: Repartition avant join principal
    df_trans_products_repart = df_trans_products.repartition(32, "customer_id")
    
    # 3. OPTIMISATION: Join avec repartitioning
    df_full = df_trans_products_repart.join(
        df_customers.repartition(32, "customer_id"),
        "customer_id"
    )
    
    # 4. OPTIMISATION: Cache du DataFrame principal (utilisé 2 fois)
    df_full_cached = df_full.select(
        "customer_id", "name", "customer_segment", 
        "category", "amount", "transaction_id"
    ).cache()
    
    # Force le cache
    df_full_cached.count()
    
    # 5. KPIs avec DataFrame caché
    kpis = df_full_cached.groupBy("customer_segment", "category").agg(
        count("*").alias("nb_transactions"),
        sum("amount").alias("total_revenue"),
        avg("amount").alias("avg_transaction"),
        countDistinct("customer_id").alias("unique_customers")
    )
    
    # 6. Top clients avec DataFrame caché
    top_customers = df_full_cached.groupBy("customer_id", "name").agg(
        sum("amount").alias("total_revenue")
    ).orderBy(desc("total_revenue")).limit(1000)
    
    # 7. OPTIMISATION: Sauvegarde avec coalesce
    kpis.coalesce(1).write.mode("overwrite") \
        .option("compression", "snappy") \
        .parquet("/tmp/kpis_optimized")
    
    top_customers.coalesce(1).write.mode("overwrite") \
        .option("compression", "snappy") \
        .parquet("/tmp/top_customers_optimized")
    
    # Libérer le cache
    df_full_cached.unpersist()
    
    return kpis.count(), top_customers.count()

time_optimized = measure_time(pipeline_optimized, "Pipeline OPTIMISÉ")[1]

print(f"\n{'=' * 60}")
print(f"RÉSULTATS FINAUX")
print(f"{'=' * 60}")
print(f"Temps baseline: {time_baseline:.2f}s")
print(f"Temps optimisé: {time_optimized:.2f}s")
print(f"Amélioration: {((time_baseline - time_optimized) / time_baseline) * 100:.1f}%")
print(f"Speedup: {time_baseline / time_optimized:.2f}x")
```

### Exercice 7.3 : Analyse comparative

```python
# Comparer la taille des fichiers produits
def analyze_output():
    print("\n" + "=" * 60)
    print("ANALYSE DES SORTIES")
    print("=" * 60)
    
    # Taille baseline
    size_kpis_baseline = get_folder_size("/tmp/kpis_baseline")
    size_top_baseline = get_folder_size("/tmp/top_customers_baseline")
    
    # Taille optimisée
    size_kpis_opt = get_folder_size("/tmp/kpis_optimized")
    size_top_opt = get_folder_size("/tmp/top_customers_optimized")
    
    # Nombre de fichiers
    files_kpis_baseline = len(glob.glob("/tmp/kpis_baseline/*.parquet"))
    files_kpis_opt = len(glob.glob("/tmp/kpis_optimized/*.parquet"))
    
    print(f"\nKPIs - Baseline:")
    print(f"  Taille: {size_kpis_baseline:.2f} MB")
    print(f"  Fichiers: {files_kpis_baseline}")
    
    print(f"\nKPIs - Optimisé:")
    print(f"  Taille: {size_kpis_opt:.2f} MB")
    print(f"  Fichiers: {files_kpis_opt}")
    print(f"  Réduction fichiers: {((files_kpis_baseline - files_kpis_opt) / files_kpis_baseline) * 100:.1f}%")
    
    print(f"\nTop Customers - Baseline: {size_top_baseline:.2f} MB")
    print(f"Top Customers - Optimisé: {size_top_opt:.2f} MB")

analyze_output()
```

### Questions TP7

1. Quelles optimisations ont eu le plus d'impact ?
2. Quelle est l'amélioration globale de performance ?
3. Y a-t-il des compromis (trade-offs) à considérer ?

---

## TP8 : Monitoring et debugging

### Objectif
Utiliser les outils de monitoring pour identifier et résoudre les problèmes de performance.

### Exercice 8.1 : Analyse des plans d'exécution

```python
# Créer une requête complexe
complex_query = df_transactions \
    .join(df_customers, "customer_id") \
    .join(df_products, "product_id") \
    .filter(col("amount") > 100) \
    .groupBy("customer_segment", "category") \
    .agg(
        sum("amount").alias("revenue"),
        count("*").alias("transactions")
    ) \
    .filter(col("revenue") > 1000)

# Analyser les différents niveaux de explain
print("=" * 60)
print("SIMPLE EXPLAIN")
print("=" * 60)
complex_query.explain()

print("\n" + "=" * 60)
print("EXTENDED EXPLAIN")
print("=" * 60)
complex_query.explain(extended=True)

print("\n" + "=" * 60)
print("FORMATTED EXPLAIN")
print("=" * 60)
complex_query.explain(mode="formatted")
```

### Exercice 8.2 : Métriques custom avec accumulateurs

```python
from pyspark import AccumulatorParam

# Créer des accumulateurs pour tracker les métriques
high_value_transactions = spark.sparkContext.accumulator(0)
low_value_transactions = spark.sparkContext.accumulator(0)
total_revenue_processed = spark.sparkContext.accumulator(0.0)

def process_with_metrics(row):
    """Fonction avec tracking de métriques"""
    amount = row['amount']
    
    # Incrémenter les compteurs
    if amount > 500:
        high_value_transactions.add(1)
    else:
        low_value_transactions.add(1)
    
    total_revenue_processed.add(float(amount))
    
    return row

# Appliquer avec tracking
df_processed = df_transactions.rdd.map(process_with_metrics).toDF()
df_processed.count()

# Afficher les métriques
print("\n" + "=" * 60)
print("MÉTRIQUES CUSTOM")
print("=" * 60)
print(f"Transactions haute valeur (>500): {high_value_transactions.value:,}")
print(f"Transactions basse valeur (<=500): {low_value_transactions.value:,}")
print(f"Revenue total traité: {total_revenue_processed.value:,.2f}€")
```

### Exercice 8.3 : Détection automatique de problèmes

```python
def diagnose_dataframe(df, name="DataFrame"):
    """Diagnostic automatique de problèmes potentiels"""
    print(f"\n{'=' * 60}")
    print(f"DIAGNOSTIC: {name}")
    print(f"{'=' * 60}")
    
    # 1. Nombre de partitions
    num_partitions = df.rdd.getNumPartitions()
    print(f"\n1. Partitions: {num_partitions}")
    
    # 2. Distribution des données par partition
    partition_counts = df.withColumn("partition_id", spark_partition_id()) \
        .groupBy("partition_id").count() \
        .select(
            mean("count").alias("mean"),
            stddev("count").alias("stddev"),
            min("count").alias("min"),
            max("count").alias("max")
        ).collect()[0]
    
    cv = partition_counts['stddev'] / partition_counts['mean'] if partition_counts['mean'] > 0 else 0
    
    print(f"\n2. Distribution par partition:")
    print(f"   Moyenne: {partition_counts['mean']:,.0f} lignes")
    print(f"   Écart-type: {partition_counts['stddev']:,.0f}")
    print(f"   Min: {partition_counts['min']:,.0f}")
    print(f"   Max: {partition_counts['max']:,.0f}")
    print(f"   Coefficient de variation: {cv:.2f}")
    
    if cv > 0.5:
        print("   ⚠️  ALERTE: Data skew détecté (CV > 0.5)")
        print("   → Recommandation: Envisager du salting ou repartitioning")
    
    # 3. Nombre de colonnes
    num_cols = len(df.columns)
    print(f"\n3. Colonnes: {num_cols}")
    
    if num_cols > 50:
        print("   ⚠️  ALERTE: Beaucoup de colonnes")
        print("   → Recommandation: Utiliser select() pour la projection")
    
    # 4. Taille estimée
    count = df.count()
    print(f"\n4. Nombre de lignes: {count:,}")
    
    # Recommendations générales
    print(f"\n5. Recommandations:")
    
    ideal_partitions = max(8, count // 100000)
    if num_partitions < ideal_partitions // 2:
        print(f"   • Augmenter le nombre de partitions à ~{ideal_partitions}")
    elif num_partitions > ideal_partitions * 2:
        print(f"   • Réduire le nombre de partitions à ~{ideal_partitions}")
    
    print(f"   • Partitions idéales estimées: {ideal_partitions}")

# Diagnostiquer nos DataFrames
diagnose_dataframe(df_transactions, "Transactions")
diagnose_dataframe(df_customers, "Customers")
```

### Questions TP8

1. Comment interpréter un plan d'exécution Spark ?
2. Quels sont les signaux d'un problème de performance ?
3. Comment utiliser les métriques pour l'optimisation ?

---

## TP9 : Challenge final - Optimisation libre

### Objectif
Optimiser un pipeline complexe en autonomie complète.

### Énoncé du challenge

Vous recevez un pipeline de production qui présente des problèmes de performance. Votre mission :

1. Analyser le pipeline actuel
2. Identifier les goulots d'étranglement
3. Proposer et implémenter des optimisations
4. Mesurer et documenter les améliorations

### Pipeline à optimiser

```python
def challenge_pipeline():
    """
    Pipeline problématique - À VOUS D'OPTIMISER !
    
    Objectif: Créer un dashboard de ventes avec:
    - Revenue par région et mois
    - Top 100 produits par catégorie
    - Analyse de cohorte clients (mois de première transaction)
    - Détection de fraude (montants suspects)
    """
    
    # Lecture des données (pas de cache, pas de projection)
    trans = spark.read.parquet("/tmp/transactions.parquet")
    cust = spark.read.parquet("/tmp/customers_large.parquet")  # Simuler une grande table
    prod = df_products
    
    # Join massif sans optimisation
    full_data = trans.join(cust, "customer_id").join(prod, "product_id")
    
    # KPI 1: Revenue par région et mois (UDF Python lent)
    @udf(returnType=StringType())
    def extract_month(timestamp):
        return str(timestamp)[:7] if timestamp else None
    
    revenue_by_region = full_data \
        .withColumn("month", extract_month(col("timestamp"))) \
        .groupBy("city", "month") \
        .agg(sum("amount").alias("revenue")) \
        .collect()  # COLLECT SUR GRANDE TABLE !
    
    # KPI 2: Top produits (tri sans limite précoce)
    top_products = full_data \
        .groupBy("category", "product_name") \
        .agg(sum("amount").alias("sales")) \
        .orderBy(desc("sales")) \
        .collect()[:100]  # Tri complet puis limite
    
    # KPI 3: Cohorte analysis (self-join sans broadcast)
    first_trans = full_data.groupBy("customer_id").agg(
        min("timestamp").alias("first_purchase")
    )
    
    cohort_analysis = full_data \
        .join(first_trans, "customer_id") \
        .withColumn("cohort_month", extract_month(col("first_purchase"))) \
        .groupBy("cohort_month") \
        .agg(countDistinct("customer_id").alias("customers"))
    
    # KPI 4: Détection fraude (UDF complexe)
    @udf(returnType=BooleanType())
    def is_suspicious(amount, customer_segment):
        if customer_segment == "Basic" and amount > 1000:
            return True
        if customer_segment == "Premium" and amount > 5000:
            return True
        return False
    
    fraud_detection = full_data \
        .withColumn("is_fraud", is_suspicious(col("amount"), col("customer_segment"))) \
        .filter(col("is_fraud") == True) \
        .select("*")
    
    # Sauvegardes non optimisées
    fraud_detection.write.mode("overwrite").json("/tmp/fraud_cases")  # JSON !
    cohort_analysis.write.mode("overwrite").csv("/tmp/cohorts")  # CSV !
    
    return len(revenue_by_region), len(top_products), fraud_detection.count()

# Exécuter le pipeline problématique
print("Exécution du pipeline NON OPTIMISÉ...")
time_challenge_baseline = measure_time(challenge_pipeline, "Challenge BASELINE")[1]
```

### Grille d'évaluation

Optimisez le pipeline ci-dessus. Vous serez évalué sur :

1. **Performance** (40 points)
   - Réduction du temps d'exécution
   - Élimination des opérations coûteuses

2. **Qualité du code** (30 points)
   - Lisibilité et maintenabilité
   - Utilisation des bonnes pratiques Spark

3. **Justification** (20 points)
   - Explication des choix d'optimisation
   - Analyse avant/après

4. **Innovation** (10 points)
   - Techniques avancées utilisées
   - Solutions créatives

### Template de solution

```python
def challenge_pipeline_optimized():
    """
    VOTRE SOLUTION OPTIMISÉE
    
    Documentez vos optimisations ici:
    1. [Optimisation 1]
    2. [Optimisation 2]
    3. [etc.]
    """
    
    # VOTRE CODE ICI
    
    pass

# Benchmark de votre solution
time_challenge_optimized = measure_time(
    challenge_pipeline_optimized, 
    "Challenge OPTIMISÉ"
)[1]

print(f"\n{'=' * 60}")
print(f"RÉSULTAT DU CHALLENGE")
print(f"{'=' * 60}")
print(f"Temps initial: {time_challenge_baseline:.2f}s")
print(f"Temps optimisé: {time_challenge_optimized:.2f}s")
print(f"Amélioration: {((time_challenge_baseline - time_challenge_optimized) / time_challenge_baseline) * 100:.1f}%")
print(f"Score performance: {min(40, int((time_challenge_baseline / time_challenge_optimized - 1) * 40))} / 40")
```

---

## Annexe : Solutions et corrigés

### Solution TP2.1 - Repartitionnement

Le nombre optimal de partitions dépend de :
- Volume de données : 1M transactions ≈ 100-200 MB
- Ressources : 4 cœurs locaux
- **Recommandation** : 16-32 partitions (4-8 par cœur)

### Solution TP4.1 - Broadcast join

Broadcast join est optimal quand :
- Table < 10 MB (configurable)
- Évite le shuffle de la grande table
- **Gain typique** : 2-10x sur nos données de test

### Solution TP6.1 - UDFs

Les fonctions natives sont plus rapides car :
- Compilées par Catalyst (code generation)
- Exécutées en Java/Scala (pas de sérialisation Python)
- Optimisables par l'optimizer
- **Gain typique** : 10-100x

### Solution Challenge final (exemple)

```python
def challenge_pipeline_optimized_solution():
    """Solution optimisée du challenge"""
    
    # 1. OPTIM: Lecture avec projection
    trans = spark.read.parquet("/tmp/transactions.parquet") \
        .select("transaction_id", "customer_id", "product_id", "amount", "timestamp")
    
    cust = spark.read.parquet("/tmp/customers_large.parquet") \
        .select("customer_id", "city", "customer_segment")
    
    # 2. OPTIM: Broadcast join pour products (petit)
    full_data = trans \
        .join(broadcast(df_products), "product_id") \
        .join(cust, "customer_id")
    
    # 3. OPTIM: Cache car réutilisé 4 fois
    full_data_cached = full_data.cache()
    full_data_cached.count()
    
    # 4. OPTIM: Fonction native au lieu d'UDF
    df_with_month = full_data_cached.withColumn(
        "month", date_format(col("timestamp"), "yyyy-MM")
    )
    
    # KPI 1: Pas de collect !
    revenue_by_region = df_with_month \
        .groupBy("city", "month") \
        .agg(sum("amount").alias("revenue"))
    
    revenue_by_region.write.mode("overwrite").parquet("/tmp/revenue_optimized")
    
    # KPI 2: OPTIM: Limite avant collect
    top_products = df_with_month \
        .groupBy("category", "product_name") \
        .agg(sum("amount").alias("sales")) \
        .orderBy(desc("sales")) \
        .limit(100)  # Limite AVANT collect
    
    top_products.write.mode("overwrite").parquet("/tmp/top_products_optimized")
    
    # KPI 3: OPTIM: Broadcast self-join
    first_trans = df_with_month.groupBy("customer_id").agg(
        min("timestamp").alias("first_purchase")
    )
    
    cohort_analysis = df_with_month \
        .join(broadcast(first_trans), "customer_id") \
        .withColumn("cohort_month", date_format(col("first_purchase"), "yyyy-MM")) \
        .groupBy("cohort_month") \
        .agg(countDistinct("customer_id").alias("customers"))
    
    cohort_analysis.write.mode("overwrite").parquet("/tmp/cohorts_optimized")
    
    # KPI 4: OPTIM: Fonction native au lieu d'UDF
    fraud_detection = full_data_cached \
        .withColumn("is_fraud", 
            when((col("customer_segment") == "Basic") & (col("amount") > 1000), True)
            .when((col("customer_segment") == "Premium") & (col("amount") > 5000), True)
            .otherwise(False)
        ) \
        .filter(col("is_fraud") == True)
    
    # OPTIM: Parquet au lieu de JSON
    fraud_detection.write.mode("overwrite").parquet("/tmp/fraud_cases_optimized")
    
    full_data_cached.unpersist()
    
    return revenue_by_region.count(), top_products.count(), fraud_detection.count()
```

**Optimisations appliquées :**
1. Projection précoce (select)
2. Broadcast join pour petites tables
3. Cache intelligent (réutilisation)
4. Fonctions natives au lieu d'UDFs
5. Élimination des collect()
6. Parquet au lieu de CSV/JSON
7. Limite avant collect pour top-N

**Amélioration attendue : 70-90%**

---

## 🎓 Évaluation finale

### QCM (30 points)

1. Quelle stratégie pour gérer le data skew ?
   - a) Augmenter la mémoire
   - b) Salting
   - c) Désactiver AQE
   - d) Utiliser CSV

2. Broadcast join est optimal quand :
   - a) Les deux tables sont grandes
   - b) Une table < 10MB
   - c) Pas de clé de jointure
   - d) On veut un shuffle

3. Les UDFs Python sont lentes car :
   - a) Python est interprété
   - b) Sérialisation Row-by-row
   - c) Pas d'optimisation Catalyst
   - d) Toutes ces réponses

*[... 10 questions au total]*

### Projet pratique (70 points)

Réaliser un pipeline ETL complet avec :
- Import de données réelles (CSV/JSON)
- Transformations complexes
- Jointures multiples
- Optimisations avancées
- Documentation des choix
- Benchmark avant/après

---

## Ressources additionnelles

### Datasets publics pour s'entraîner
- Kaggle E-commerce: https://www.kaggle.com/carrie1/ecommerce-data
- NYC Taxi: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Amazon Reviews: https://nijianmo.github.io/amazon/index.html

### Outils recommandés
- Databricks Community Edition (gratuit)
- Google Colab avec PySpark
- Docker Spark local

### Pour aller plus loin
- Certification Databricks Spark Developer
- Cours Spark Tuning and Best Practices
- Streaming avec Structured Streaming
- Delta Lake et Lakehouse architecture

---

**Bon courage pour les TP ! 🚀**
