# Big Data - Analyse Exploratoire Appliquée à l'Assurance

##  Présentation du Projet
Ce projet présente une analyse exploratoire de données (EDA) appliquée au secteur de l'assurance. L'objectif est de comprendre les facteurs clés influençant les risques et les coûts, et d'en extraire des insights décisionnels.
Les différents modules ont permis de modéliser et de valoriser un jeu de données opérationnel lié à la gestion des dossiers d'assistance assurance.

## Contenu des Analyses
La démarche s'articule autour de trois grands axes :


**Traitement des données :**
Nettoyage et harmonisation des différents jeu de données. Création d'une base unique en joignant toutes les  bases, qui servira pour l'analyse économétrique.  (Vidéo de présentation disponible à la racine)


**Modélisation Économétrique (Régression MCO) & Analyse Exploratoire (ACP) :**
Identification des dimensions indépendantes qui régissent l'activité (Profil/Périmètre vs Typologie technique).
Quantification des facteurs influences les temps de traitement (log_duree) et mise en évidence du paradoxe quant à l'expérience des agents sur la variable Population_CAS. (Vidéo de présentation disponible à la racine)


**Machine Learning :**
Entraînement, validation et comparaison de quelques modèles de ML en s'assurant d'éviter le surapprentissage dans leur fonctionnement.


**Dashboard Interactive avec l'outil Streamlit :**
Conception d'un simulateur destiné aux managers pour estimer les temps de traitement des dossiers en configurant des options. Ce dashboard résume aussi des résultats des différents modèles  explorés.

**Pour accéder au Dashboard :**
Ouvrez un terminal, placez-vous dans le dossier contenant dashboard.py, puis exécutez :
bashcd /chemin/vers/le/dossier
streamlit run dashboard.py  
**-> Lecture du Dashboard :**    
Une fois le dashboard ouvert, dans la barre latérale à gauche ,    
cliquez sur "📂 Charger votre CSV"
Naviguez vers votre fichier base_econometrie_clean.csv peu importe où il se trouve sur votre machine
Sélectionnez-le et le dashboard se met à jour automatiquement.   
Attention !!! Sans fichier chargé, le dashboard fonctionne avec des données simulées à titre de démonstration.


## Stack Technique
* **Langage :** Python 3
* **Environnement :**  Notebook (VS Code)
* **Librairies principales :** Pandas, NumPy, Matplotlib, Seaborn

##  Structure du Dépôt
* `notebooks/` : Contient les notebooks d'analyse pas à pas.
* `data/` : Jeux de données d'exemple.
