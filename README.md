# Projet NLP - RNN vs LSTM pour la classification de sentiments

## 📌 Description

Ce projet compare deux architectures de réseaux de neurones récurrents sur une tâche de classification de sentiments binaire (positif/négatif) à partir des critiques de films IMDb.

- **Tâche** : Classification de sentiments (Sentiment Analysis)
- **Modèles** : RNN bidirectionnel vs LSTM bidirectionnel
- **Dataset** : IMDb Reviews (HuggingFace) - 50 000 critiques

## 🎯 Objectifs

- Construire un pipeline NLP complet
- Entraîner deux modèles (RNN et LSTM)
- Évaluer et comparer leurs performances
- Déployer une application de démonstration interactive

## 📊 Résultats

| Modèle | Accuracy (Test) | Precision | Recall | F1-score | Temps inférence |
|--------|-----------------|-----------|--------|----------|-----------------|
| **RNN** | 73.24% | 0.7471 | 0.7028 | 0.7243 | 73s |
| **LSTM** | **84.30%** | **0.8596** | **0.8200** | **0.8393** | 161s |

**Le LSTM surpasse le RNN de +11.06% d'accuracy.**

## 📁 Structure du projet
📂 Projet_NLP_RNN_vs_LSTM/
│
├── 📄 README.md # Ce fichier
├── 📄 requirements.txt # Dépendances Python
├── 📄 Rapport_NLP_RNN_vs_LSTM.docx # Rapport Word (6-10 pages)
├── 📄 NLP_RNN_vs_LSTM.pptx # PowerPoint de présentation
├── 🐍 app.py # Application Streamlit (démo)
│
├── 📓 notebooks/
│ ├── dataset.ipynb # Téléchargement du dataset
  ├── Exploration.ipynb # Exploration des données
│ ├── preprocessing.ipynb # Prétraitement
│ ├── train_rnn.ipynb # Entraînement RNN
│ ├── train_lstm.ipynb # Entraînement LSTM
│ ├── evaluation.ipynb # Évaluation et comparaison
│
└── 📁 data/ # Données et modèles sauvegardés
├── imdb_train.csv # 25 000 critiques (train)
├── imdb_test.csv # 25 000 critiques (test)
├── vocab.pkl # Vocabulaire (20 000 mots)
├── rnn_model.pth # Poids du modèle RNN
├── lstm_model.pth # Poids du modèle LSTM
└── ... # Autres fichiers de données