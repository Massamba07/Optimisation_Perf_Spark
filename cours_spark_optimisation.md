# Optimisation et Performance Apache Spark
## Master 2 - Data Engineering

---

## Leçon 1 : Architecture et Fondamentaux Spark

### 1.1 Architecture Spark - Rappels essentiels

#### Composants clés
- **Driver** : Orchestre l'exécution, maintient le SparkContext
- **Executors** : Exécutent les tâches, stockent les données en cache
- **Cluster Manager** : Gère les ressources (YARN, Mesos, Kubernetes, Standalone)

#### Concepts fondamentaux
- **RDD (Resilient Distributed Dataset)** : Structure de données immuable et distribuée
- **DataFrame/Dataset** : API structurée avec optimisations Catalyst
- **Transformations lazy** : map, filter, groupBy (évaluées à l'action)
- **Actions** : collect, count, save (déclenchent l'exécution)

#### DAG (Directed Acyclic Graph)
```
Transformation 1 → Transformation 2 → Transformation 3 → Action
                    ↓
            Optimisation Catalyst
                    ↓
            Planification physique
```

### 1.2 Modèle d'exécution

#### Jobs, Stages et Tasks
- **Job** : Déclenché par une action
- **Stage** : Ensemble de transformations sans shuffle
- **Task** : Unité de travail sur une partition

#### Shuffles - Point critique de performance
Un shuffle se produit lors de :
- `groupBy`, `reduceByKey`, `join`
- `repartition`, `coalesce`
- `distinct`, `intersection`

**Impact** : Lecture/écriture disque, transfert réseau, sérialisation/désérialisation

---

## Leçon 2 : Optimisation des Partitions

### 2.1 Comprendre le partitionnement

#### Taille optimale des partitions
- **Trop petites** : Overhead de scheduling, nombreuses tâches
- **Trop grandes** : Déséquilibre de charge, risque OOM
- **Règle d'or** : 128 MB - 1 GB par partition

#### Calcul du nombre de partitions
```python
# Formule recommandée
num_partitions = (data_size_gb * 1024) / partition_size_mb

# Exemple : 100 GB de données, partitions de 256 MB
num_partitions = (100 * 1024) / 256 = 400 partitions
```

### 2.2 Stratégies de repartitionnement

#### `repartition()` vs `coalesce()`
```python
# repartition() : Full shuffle, peut augmenter ou diminuer
df.repartition(100)  # Redistribution complète

# coalesce() : Sans shuffle, uniquement pour diminuer
df.coalesce(50)  # Combine les partitions existantes
```

#### Partitionnement par colonne
```python
# Optimise les opérations groupBy et join ultérieures
df.repartition(200, "customer_id")
df.repartition(col("date"), col("region"))
```

### 2.3 Data Skew - Le défi du déséquilibre

#### Détection du skew
```python
from pyspark.sql.functions import spark_partition_id, count

# Vérifier la distribution des données
df.withColumn("partition_id", spark_partition_id()) \
  .groupBy("partition_id") \
  .count() \
  .orderBy("count", ascending=False) \
  .show()
```

#### Solutions au data skew

**Technique 1 : Salting**
```python
from pyspark.sql.functions import rand, concat, lit

# Ajouter un salt pour distribuer les clés populaires
df_salted = df.withColumn("salt", (rand() * 10).cast("int")) \
              .withColumn("salted_key", concat(col("key"), lit("_"), col("salt")))
```

**Technique 2 : Broadcast join pour petites tables**
```python
from pyspark.sql.functions import broadcast

# Diffuser la petite table sur tous les workers
result = large_df.join(broadcast(small_df), "key")
```

**Technique 3 : Split-Combine**
```python
# Séparer les clés skewed et normales
skewed_keys = ["key1", "key2", "key3"]
df_skewed = df.filter(col("key").isin(skewed_keys))
df_normal = df.filter(~col("key").isin(skewed_keys))

# Traiter séparément avec stratégies différentes
result_skewed = df_skewed.repartition(100, "key")
result_normal = df_normal.repartition(20, "key")

# Combiner
final_result = result_skewed.union(result_normal)
```

---

## 🚀 Module 3 : Optimisations Catalyst et Tungsten

### 3.1 Catalyst Optimizer

#### Phases d'optimisation
1. **Analyse** : Résolution des références
2. **Logical Optimization** : Règles d'optimisation
3. **Physical Planning** : Génération de plans physiques
4. **Code Generation** : Compilation en bytecode Java

#### Optimisations automatiques
- **Predicate Pushdown** : Filtres appliqués tôt
- **Projection Pushdown** : Lecture uniquement des colonnes nécessaires
- **Constant Folding** : Évaluation des constantes
- **Join Reordering** : Ordre optimal des jointures

### 3.2 Exploitation de Catalyst

#### Utiliser les fonctions natives
```python
# ❌ LENT : UDF Python
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

@udf(returnType=IntegerType())
def custom_length(s):
    return len(s) if s else 0

df.withColumn("len", custom_length(col("name")))

# ✅ RAPIDE : Fonction native
from pyspark.sql.functions import length
df.withColumn("len", length(col("name")))
```

#### Vectorized UDFs (Pandas UDF)
```python
from pyspark.sql.functions import pandas_udf
import pandas as pd

# 10-100x plus rapide qu'un UDF standard
@pandas_udf(IntegerType())
def vectorized_length(s: pd.Series) -> pd.Series:
    return s.str.len()

df.withColumn("len", vectorized_length(col("name")))
```

### 3.3 Tungsten - Gestion mémoire

#### Project Tungsten optimise
- **Gestion mémoire off-heap** : Évite le GC Java
- **Code generation** : Compile les expressions
- **Cache-aware computation** : Optimise les accès CPU
- **Encodage binaire compact** : Réduit l'empreinte mémoire

#### Configuration Tungsten
```python
spark.conf.set("spark.sql.tungsten.enabled", "true")
spark.conf.set("spark.sql.codegen.wholeStage", "true")
spark.conf.set("spark.sql.codegen.aggregate", "true")
```

---

## 💾 Module 4 : Gestion de la Mémoire et du Cache

### 4.1 Architecture mémoire Spark

#### Répartition mémoire executor
```
Total Memory (spark.executor.memory)
    ├── Reserved Memory (300 MB)
    ├── User Memory (40% par défaut)
    └── Spark Memory (60% par défaut)
        ├── Storage Memory (50%) : cache, broadcast
        └── Execution Memory (50%) : shuffles, joins, aggregations
```

### 4.2 Stratégies de cache

#### Niveaux de stockage
```python
from pyspark import StorageLevel

# MEMORY_ONLY : Rapide mais risque OOM
df.persist(StorageLevel.MEMORY_ONLY)

# MEMORY_AND_DISK : Équilibré, recommandé
df.persist(StorageLevel.MEMORY_AND_DISK)

# DISK_ONLY : Pour grands datasets
df.persist(StorageLevel.DISK_ONLY)

# MEMORY_ONLY_SER : Sérialisé, moins d'espace
df.persist(StorageLevel.MEMORY_ONLY_SER)

# OFF_HEAP : Évite GC Java
df.persist(StorageLevel.OFF_HEAP)
```

#### Quand cacher ?
```python
# ✅ BON : DataFrame utilisé plusieurs fois
df_filtered = df.filter(col("status") == "active").cache()

result1 = df_filtered.groupBy("category").count()
result2 = df_filtered.groupBy("region").agg(sum("revenue"))
result3 = df_filtered.join(other_df, "id")

# ❌ MAUVAIS : Cache inutile
df.cache().count()  # Utilisé une seule fois

# Libérer le cache
df_filtered.unpersist()
```

#### Cache dynamique avec Adaptive Query Execution
```python
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

### 4.3 Configuration mémoire optimale

```python
# Configuration type pour cluster production
spark = SparkSession.builder \
    .appName("OptimizedApp") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.instances", "10") \
    .config("spark.driver.memory", "4g") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.5") \
    .config("spark.default.parallelism", "200") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()
```

---

## 🔗 Module 5 : Optimisation des Jointures

### 5.1 Types de jointures Spark

#### Broadcast Hash Join
- **Quand ?** : Table < 10 MB (configurable)
- **Comment ?** : Table diffusée à tous les executors
- **Performance** : O(n), pas de shuffle

```python
# Auto-broadcast si < spark.sql.autoBroadcastJoinThreshold
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10485760)  # 10 MB

# Forcer le broadcast
from pyspark.sql.functions import broadcast
result = large_df.join(broadcast(small_df), "key")
```

#### Shuffle Hash Join
- **Quand ?** : Tables moyennes, équilibrées
- **Comment ?** : Hash partitioning + shuffle
- **Performance** : O(n + m)

#### Sort Merge Join
- **Quand ?** : Grandes tables
- **Comment ?** : Sort + merge
- **Performance** : O(n log n + m log m)

```python
# Spark choisit automatiquement, mais on peut influencer
spark.conf.set("spark.sql.join.preferSortMergeJoin", "true")
```

### 5.2 Optimisations avancées

#### Bucketing
```python
# Pré-partitionner et trier à l'écriture
df.write \
    .bucketBy(100, "user_id") \
    .sortBy("timestamp") \
    .saveAsTable("users_bucketed")

# Lecture sans shuffle si même bucketing
df1 = spark.table("users_bucketed")
df2 = spark.table("transactions_bucketed")
result = df1.join(df2, "user_id")  # Pas de shuffle !
```

#### Bloom Filter Join
```python
spark.conf.set("spark.sql.optimizer.bloomFilter.enabled", "true")
spark.conf.set("spark.sql.optimizer.bloomFilter.expectedItems", "1000000")
```

---

## 📁 Module 6 : Formats de Fichiers et I/O

### 6.1 Comparaison des formats

| Format | Compression | Colonaire | Schema Evolution | Cas d'usage |
|--------|-------------|-----------|------------------|-------------|
| CSV | Faible | Non | Non | Import/Export simple |
| JSON | Faible | Non | Oui | APIs, semi-structuré |
| Avro | Bonne | Non | Oui | Streaming, évolution schema |
| Parquet | Excellente | Oui | Oui | Analytics, OLAP |
| ORC | Excellente | Oui | Oui | Hive, analytics |
| Delta | Excellente | Oui | Oui | ACID, time travel |

### 6.2 Parquet - Best practices

```python
# Écriture optimisée
df.write \
    .mode("overwrite") \
    .partitionBy("year", "month") \
    .option("compression", "snappy") \
    .parquet("output/data.parquet")

# Lecture avec projection et predicate pushdown
df = spark.read.parquet("output/data.parquet") \
    .select("id", "name", "revenue") \
    .filter(col("year") == 2024)
```

#### Taille optimale des fichiers Parquet
- **Cible** : 128 MB - 1 GB par fichier
- **Éviter** : Milliers de petits fichiers (small files problem)

```python
# Compaction des petits fichiers
df.coalesce(10).write.parquet("output/compacted")

# Ou avec repartition si besoin de redistribuer
df.repartition(10).write.parquet("output/redistributed")
```

### 6.3 Delta Lake - Nouvelle génération

```python
from delta import DeltaTable

# Écriture Delta
df.write.format("delta") \
    .mode("overwrite") \
    .save("/delta/table")

# MERGE pour upserts
deltaTable = DeltaTable.forPath(spark, "/delta/table")
deltaTable.alias("target").merge(
    updates.alias("source"),
    "target.id = source.id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()

# Optimisation compaction
deltaTable.optimize().executeCompaction()

# Z-ordering pour colonnes fréquemment filtrées
deltaTable.optimize().executeZOrderBy("date", "customer_id")
```

---

## ⚡ Module 7 : Tuning Avancé

### 7.1 Paramètres critiques

```python
# Parallelisme
spark.conf.set("spark.default.parallelism", num_executors * executor_cores * 2)
spark.conf.set("spark.sql.shuffle.partitions", 200)

# Compression shuffle
spark.conf.set("spark.shuffle.compress", "true")
spark.conf.set("spark.shuffle.spill.compress", "true")
spark.conf.set("spark.io.compression.codec", "snappy")

# Sérialisation
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
spark.conf.set("spark.kryo.registrationRequired", "false")

# Speculation (réexécution tâches lentes)
spark.conf.set("spark.speculation", "true")
spark.conf.set("spark.speculation.multiplier", "1.5")

# Dynamic allocation
spark.conf.set("spark.dynamicAllocation.enabled", "true")
spark.conf.set("spark.dynamicAllocation.minExecutors", "2")
spark.conf.set("spark.dynamicAllocation.maxExecutors", "50")
```

### 7.2 Monitoring et Debugging

#### Spark UI - Points clés
- **Jobs** : Durée, stages, tâches
- **Stages** : Shuffles, temps par tâche
- **Storage** : Cache utilisé
- **Executors** : CPU, mémoire, GC time
- **SQL** : Plan d'exécution, métriques

#### Métriques à surveiller
```python
# Activer les métriques détaillées
spark.conf.set("spark.sql.statistics.histogram.enabled", "true")
spark.conf.set("spark.sql.cbo.enabled", "true")

# Analyser le plan d'exécution
df.explain(mode="extended")
df.explain(mode="cost")
df.explain(mode="formatted")
```

#### Logging personnalisé
```python
import logging

# Configuration logging
logger = logging.getLogger("SparkOptimization")
logger.setLevel(logging.INFO)

# Metrics custom
from pyspark import AccumulatorParam
records_processed = spark.sparkContext.accumulator(0)

def process_partition(partition):
    count = 0
    for record in partition:
        # traitement
        count += 1
    records_processed.add(count)
    return partition

df.rdd.mapPartitions(process_partition).count()
print(f"Records processed: {records_processed.value}")
```

---

## 🎯 Module 8 : Patterns et Anti-patterns

### 8.1 Best Practices

#### ✅ Bonnes pratiques

1. **Filter early, filter often**
```python
# ✅ Filtrer tôt
df.filter(col("date") >= "2024-01-01") \
  .filter(col("status") == "active") \
  .groupBy("category").count()
```

2. **Utiliser les colonnes natives**
```python
# ✅ Sélectionner par nom
df.select("id", "name", "revenue")

# ❌ Éviter select *
df.select("*")
```

3. **Réutiliser les DataFrames intermédiaires**
```python
# ✅ Chaîner intelligemment
filtered_df = df.filter(col("year") == 2024).cache()
result1 = filtered_df.groupBy("region").count()
result2 = filtered_df.agg(sum("revenue"))
```

4. **Partitionnement logique à l'écriture**
```python
# ✅ Partitions basées sur les requêtes futures
df.write.partitionBy("year", "month", "day").parquet("output")
```

### 8.2 Anti-patterns à éviter

#### ❌ Anti-pattern 1 : Collect sur grands datasets
```python
# ❌ DANGEREUX : Ramène tout au driver
all_data = df.collect()  # OOM si > driver memory

# ✅ Alternative : Traiter distribué
df.write.parquet("output")
df.foreach(lambda row: process(row))
```

#### ❌ Anti-pattern 2 : UDFs Python non optimisés
```python
# ❌ LENT : UDF row-by-row
@udf(returnType=DoubleType())
def calculate(value):
    return complex_calculation(value)

# ✅ Pandas UDF vectorisé
@pandas_udf(DoubleType())
def calculate_vectorized(values: pd.Series) -> pd.Series:
    return values.apply(complex_calculation)
```

#### ❌ Anti-pattern 3 : Shuffles inutiles
```python
# ❌ Plusieurs repartitions
df.repartition(100).filter(...).repartition(50)

# ✅ Une seule repartition bien placée
df.filter(...).repartition(50)
```

#### ❌ Anti-pattern 4 : Cartesian products accidentels
```python
# ❌ Join sans condition = cartesian
df1.crossJoin(df2)  # n * m lignes !

# ✅ Join avec condition
df1.join(df2, df1.id == df2.id)
```

---

## 📝 Checklist d'optimisation

### Avant développement
- [ ] Comprendre le volume de données (GB/TB)
- [ ] Identifier les opérations coûteuses (joins, aggregations)
- [ ] Estimer les ressources nécessaires

### Développement
- [ ] Utiliser DataFrames/Datasets plutôt que RDDs
- [ ] Privilégier les fonctions natives Spark
- [ ] Filtrer et projeter tôt dans le pipeline
- [ ] Éviter les UDFs Python quand possible
- [ ] Partitionner intelligemment

### Avant production
- [ ] Analyser les plans d'exécution (explain)
- [ ] Tester avec données réelles
- [ ] Configurer les partitions shuffle appropriées
- [ ] Activer AQE (Adaptive Query Execution)
- [ ] Définir les stratégies de cache

### Production
- [ ] Monitorer Spark UI régulièrement
- [ ] Surveiller les métriques executors
- [ ] Ajuster selon les patterns observés
- [ ] Optimiser itérativement

---

## 🔬 Méthodes de benchmarking

### Test de performance
```python
import time

def benchmark(func, *args, **kwargs):
    """Mesure le temps d'exécution"""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")
    return result

# Utilisation
result = benchmark(df.groupBy("category").count().collect)
```

### Comparaison A/B
```python
# Version 1 : Sans optimisation
v1_time = benchmark(lambda: df.groupBy("key").count().collect())

# Version 2 : Avec repartition
v2_time = benchmark(lambda: df.repartition(100, "key")
                                .groupBy("key").count().collect())

print(f"Amélioration: {((v1_time - v2_time) / v1_time) * 100:.1f}%")
```

---

## 📚 Ressources complémentaires

### Documentation officielle
- Apache Spark Documentation : https://spark.apache.org/docs/latest/
- Spark Performance Tuning : https://spark.apache.org/docs/latest/tuning.html
- Databricks Best Practices : https://www.databricks.com/learn

### Livres recommandés
- "Spark: The Definitive Guide" - Bill Chambers & Matei Zaharia
- "High Performance Spark" - Holden Karau & Rachel Warren
- "Learning Spark, 2nd Edition" - Jules Damji et al.

### Outils
- Spark UI : Analyse des jobs
- Ganglia/Prometheus : Monitoring cluster
- Dr. Elephant (LinkedIn) : Analyse automatique jobs Spark

---

**Fin du cours théorique**

Les travaux pratiques suivent dans un document séparé.
