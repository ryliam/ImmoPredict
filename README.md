# 🏠 ImmoPredict AI — Plateforme Prédictive & Décisionnelle en Investissement Immobilier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](https://docs.pytest.org/)

**ImmoPredict AI** est une solution complète d'intelligence immobilière combinant **Machine Learning**, **Architecture MCP (Model Context Protocol)** et **Traitement du Langage Naturel (NLP)** pour orienter les particuliers et investisseurs dans leurs décisions d'achat, de location et d'évaluation de rentabilité.

---

## 🎯 Cas d'Usage Principaux

1. **📍 Faisabilité & Opportunité d'un Projet :**  
   *« J'ai 200 000 €, j'aimerais acheter un appartement à Ambérieu-en-Bugey, est-ce une bonne idée ? »*  
   → Analyse du score d'opportunité locale, calcul du taux d'effort par rapport aux revenus fiscaux, taux de vacance et comparaison achat vs location.

2. **📈 Rentabilité Prévisionnelle à 2 ans :**  
   *« Ce bien à 280 000 € sera-t-il rentable dans 2 ans ? »*  
   → Simulation financière dynamique (indexation IRL sur les loyers, charges/taxes, plus-value prévisionnelle, cash-flow net et ROI global).

3. **🏆 Recommandation Intelligente de Villes :**  
   *« Quel est le meilleur endroit pour acheter une maison avec 35 000 € de ressources ? »*  
   → Moteur de scoring multicritère classant les meilleures communes selon l'accessibilité financière, le rendement locatif brut et la tension du marché.

---

## 🏗️ Architecture Technique

```
Immo/
├── data/                         # Datasets bruts (Macro, DGFiP, Loyers, Parc, DVF)
├── models_saved/                 # Modèles Random Forest persistés (joblib)
├── src/
│   ├── config.py                 # Configuration centralisée & variables d'environnement
│   ├── data_pipeline/            # Ingestion, imputation 0 NaN et Feature Store
│   │   ├── ingestion.py
│   │   ├── cleaning.py
│   │   └── feature_store.py
│   ├── models/                   # Modèles ML & Simulateurs financiers
│   │   ├── price_predictor.py    # Random Forest Regressor (Prix/Loyers au m²)
│   │   ├── trend_forecaster.py   # Simulateur d'évolution IRL & ROI
│   │   └── zone_recommender.py   # Moteur de scoring & Recommandation Top-K
│   ├── mcp_server/               # Serveur MCP & Outils standardisés
│   │   ├── tools.py
│   │   └── server.py
│   ├── agent/                    # Agent Conversationnel NLP & Dispatcher MCP
│   │   └── agent_service.py
│   └── ui/                       # Interface utilisateur Web interactive
│       ├── app.py                # Application Streamlit
│       └── charts.py             # Visualisations interactives Plotly
├── tests/                        # Suite de tests unitaires & d'intégration
│   └── test_pipeline.py
├── explo.ipynb                   # Notebook d'exploration & EDA structuré
├── requirements.txt              # Dépendances du projet
└── .env.example                  # Modèle de configuration des variables d'environnement
```

---

## ⚙️ Installation & Démarrage

### 1. Cloner le dépôt
```bash
git clone https://github.com/votre-compte/ImmoPredict.git
cd ImmoPredict
```

### 2. Créer et activer l'environnement virtuel
```bash
python -m venv venv

# Sous Windows :
.\venv\Scripts\activate

# Sous Linux / macOS :
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement (Optionnel)
Créez un fichier `.env` à la racine :
```env
GEMINI_API_KEY=votre_cle_api_ici
GEMINI_MODEL=gemini-2.5-flash
```

---

## 🚀 Lancement de l'Application

```bash
streamlit run src/ui/app.py
```

L'application s'ouvre automatiquement dans votre navigateur sur `http://localhost:8501`.

---

## 🧪 Exécution des Tests

```bash
python -m pytest tests/
```

---

## 📊 Sources de Données Utilisées
- **DGFiP :** Données fiscales des foyers et revenus moyens par commune.
- **Ministère du Logement :** Carte des loyers d'appartements et de maisons.
- **INSEE :** Parc de logements, vacance locative et Indice de Référence des Loyers (IRL).
- **Banque de France / ACPR :** Taux de crédit immobilier, flux d'emprunts et taux d'endettement.
- **DVF (Demandes de Valeurs Foncières) :** Transactions immobilières géolocalisées.

---

## 👤 Auteur
Projet développé par **ryliam** ([darylwilliam25@gmail.com](mailto:darylwilliam25@gmail.com)).

