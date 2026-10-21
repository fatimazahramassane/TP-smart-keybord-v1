Fatimazahra massane 
#  Smart Keyboard V1

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tp-smart-keybord-v1.streamlit.app/)

** Live Demo :** [https://tp-smart-keybord-v1.streamlit.app/](https://tp-smart-keybord-v1.streamlit.app/)

---

##  Description
Application de prédiction de mots (Smart Keyboard) basée sur des modèles de langage N-Gram (Unigramme, Bigramme, Trigramme) avec lissage Add-k et mécanisme de back-off.

##  Fonctionnalités
- Interface interactive construite avec **Streamlit**.
- Choix dynamique du modèle (N-gram ou Interpolé) et du paramètre de lissage `k`.
- Suggestions de mots en temps réel avec affichage des probabilités.
- Module d'évaluation intégré (Perplexité, Précision Top-3).

##  Structure du projet
```text
├── app.py                 # Interface utilisateur Streamlit
├── requirements.txt       # Dépendances du projet
├── src/
│   ├── pipeline.py        # Logique des modèles N-Gram et évaluation
│   └── evaluate.py        # Script d'évaluation et génération de métriques
├── data/                  # Corpus d'entraînement
└── results/               # Sorties des évaluations (métriques, graphiques)
