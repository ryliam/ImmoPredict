import os
from pathlib import Path

# Chargement des variables d'environnement (.env)
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Chemins de base du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models_saved"

# Création des dossiers nécessaires
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Noms des fichiers sources
FILE_EMPRUNTS = DATA_DIR / "flux_nouveaux_emprunts.csv"
FILE_ENDETTEMENT = DATA_DIR / "taux_endettement.csv"
FILE_INTERET = DATA_DIR / "taux_interet.csv"
FILE_IRL = DATA_DIR / "indice_reference_loyers.csv"
FILE_LOYERS = DATA_DIR / "loyers.csv"
FILE_FISCAUX = DATA_DIR / "foyers_fiscaux.csv"
FILE_PARC = DATA_DIR / "parc_immobilier.csv"
FILE_TRANSACTIONS_SAMPLE = DATA_DIR / "transactions_sample.csv"
FILE_TRANSACTIONS_NPZ = DATA_DIR / "transactions.npz"

# Fichiers de persistance des modèles de Machine Learning (joblib)
PATH_MODEL_PRIX_APT = MODELS_DIR / "model_prix_apt.joblib"
PATH_MODEL_PRIX_MAI = MODELS_DIR / "model_prix_mai.joblib"
PATH_MODEL_LOYER_APT = MODELS_DIR / "model_loyer_apt.joblib"
PATH_MODEL_LOYER_MAI = MODELS_DIR / "model_loyer_mai.joblib"
PATH_SCALER = MODELS_DIR / "scaler.joblib"
PATH_PROCESSED_DATA = MODELS_DIR / "df_features.parquet"

# Configuration LLM (Google Gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
