La prolifération des discours haineux sur les réseaux sociaux est un problème majeur qui nécessite des solutions automatisées et nuancées. 
Ce projet vise à classer des tweets en trois catégories distinctes en utilisant des techniques avancées de Traitement du Langage Naturel (NLP) :
Hate Speech (Discours Haineux) : Propos racistes, sexistes ou discriminatoires ciblant un groupe.
Offensive Language (Langage Offensant) : Insultes vulgaires sans caractère haineux systémique.
Neither (Neutre) : Contenu non toxique.
Le défi principal de ce projet réside dans la distinction subtile entre "Haine" et "Insulte", ainsi que la gestion d'un dataset fortement déséquilibré.
Fonctionnalités
  Approche Hybride : Comparaison et déploiement de deux architectures (Machine Learning classique vs Deep Learning).
  Machine Learning Classique : Régression Logistique avec pondération des classes (Class Balancing).
  Deep Learning : Réseau de neurones LSTM (Long Short-Term Memory) pour capturer le contexte séquentiel.
  Gestion du Déséquilibre : Utilisation de class_weight='balanced' et de métriques adaptées (F1-Score Macro) pour ne pas négliger la classe minoritaire.
  Interface Web Interactive : Dashboard complet développé avec Streamlit permettant une analyse en temps réel avec visualisation des probabilités.

Stack Technique
  Langage : Python
  Data Processing : Pandas, NumPy, Regex (Nettoyage de texte)
  NLP & Vectorisation : TF-IDF (n-grams), Tokenizer Keras
  Machine Learning : Scikit-Learn (Logistic Regression, Random Forest, Benchmark)
  Deep Learning : TensorFlow/Keras (Embedding, LSTM, Dropout)
  Déploiement : Streamlit
Interface Utilisateur
  L'interface développée permet de :
  Saisir un texte brut.
  Choisir le moteur d'analyse (Machine Learning ou Deep Learning).
  Visualiser la classification avec un code couleur et les probabilités de confiance.
