# Prédiction de la Durée des Déplacements sous Apache Spark
### Projet Big Data & Machine Learning (Spark MLlib) — Mobilité Urbaine à Dakar

---

## Description du Projet

Ce projet vise à modéliser et prédire la durée de déplacement (`Duree_mn`) dans la région métropolitaine de Dakar en s'appuyant sur l'écosystème **Apache Spark** (PySpark, Spark SQL, Spark MLlib).

Le système analyse et apprend à partir de multiples facteurs de mobilité :
* **Moyen de transport** (`Moyen_transport`)
* **Localisation géographique** (communes et départements d'origine et de destination)
* **Profil temporel** (heure de départ, impact des heures de pointe)
* **Caractéristiques du trajet** (correspondances, profils des usagers)

---

## Structure du Répertoire

* **`projet_spark.ipynb`** : Notebook Jupyter contenant l'analyse exploratoire des données, les transformations Spark SQL / DataFrame, et les modèles de Machine Learning (Spark MLlib).
* **`generate_notebook.py`** : Script Python de génération et structuration du notebook d'analyse.
* **`deplacement.csv`** : Données des trajets et déplacements.
* **`communes.csv`** : Table de correspondance géographique des communes de la région de Dakar.
* **`donnees_deplacement_PSpark 2026.xlsx`** : Données de mobilité au format Excel.

---

## Technologies Utilisées

* **Apache Spark / PySpark** : Traitement distribué des données volumineuses & Machine Learning (`pyspark.ml`)
* **Python 3** : Pandas, NumPy, Matplotlib, Seaborn, openpyxl
* **Jupyter Notebook** : Environnement d'analyse interactive

---

## Utilisation

1. **Prérequis** :
   * Java (JDK 8 ou 11)
   * Apache Spark & PySpark
   * Jupyter Notebook / JupyterLab

2. **Lancement de l'analyse** :
   ```bash
   jupyter notebook projet_spark.ipynb
   ```
