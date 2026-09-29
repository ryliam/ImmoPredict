import pandas as pd
from typing import Dict, Any, Optional
from config.settings import (
    FILE_EMPRUNTS,
    FILE_ENDETTEMENT,
    FILE_INTERET,
    FILE_IRL,
    FILE_LOYERS,
    FILE_FISCAUX,
    FILE_PARC,
    FILE_TRANSACTIONS_SAMPLE,
)


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """
    Charge l'ensemble des fichiers CSV bruts du dossier data.
    Retourne un dictionnaire contenant les DataFrames correspondants.
    """
    data = {}
    
    if FILE_EMPRUNTS.exists():
        data['emprunts'] = pd.read_csv(FILE_EMPRUNTS)
    if FILE_ENDETTEMENT.exists():
        data['endettement'] = pd.read_csv(FILE_ENDETTEMENT)
    if FILE_INTERET.exists():
        data['interet'] = pd.read_csv(FILE_INTERET)
    if FILE_IRL.exists():
        data['irl'] = pd.read_csv(FILE_IRL)
    if FILE_LOYERS.exists():
        data['loyers'] = pd.read_csv(FILE_LOYERS)
    if FILE_FISCAUX.exists():
        data['fiscaux'] = pd.read_csv(FILE_FISCAUX)
    if FILE_PARC.exists():
        data['parc'] = pd.read_csv(FILE_PARC)
    if FILE_TRANSACTIONS_SAMPLE.exists():
        data['transactions'] = pd.read_csv(FILE_TRANSACTIONS_SAMPLE)
        
    return data

