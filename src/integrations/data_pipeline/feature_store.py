import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from src.integrations.data_pipeline.ingestion import load_raw_data
from src.integrations.data_pipeline.cleaning import clean_macro_data, clean_territorial_data


class FeatureStore:
    """
    Gestionnaire centralisé des données enrichies pour les modèles de Machine Learning
    et les outils d'analyse MCP.
    """
    _instance = None

    def __init__(self):
        self.raw_data = load_raw_data()
        self.df_macro = clean_macro_data(
            emprunts_df=self.raw_data['emprunts'],
            irl_df=self.raw_data['irl'],
            interet_df=self.raw_data.get('interet'),
            endettement_df=self.raw_data.get('endettement')
        )
        self.df_territorial = clean_territorial_data(
            loyers_df=self.raw_data['loyers'],
            fiscaux_df=self.raw_data['fiscaux'],
            parc_df=self.raw_data['parc']
        )
        self._build_consolidated_features()

    @classmethod
    def get_instance(cls) -> "FeatureStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _build_consolidated_features(self):
        """
        Construit les métriques d'analyse financière et immobilière par commune.
        """
        df = self.df_territorial.copy()

        # Estimation du prix d'achat au m² moyen basé sur le multiplicateur de loyer et le revenu
        facteur_rendement = np.clip(0.09 - (df['revenu_fiscal_moyen'] / 1000000), 0.045, 0.09)
        df['prix_m2_appartement'] = (df['loyer_m2_appartement'] * 12) / facteur_rendement
        df['prix_m2_maison'] = (df['loyer_m2_maison'] * 12) / (facteur_rendement * 1.05)

        # Rendements locatifs bruts (%)
        df['rendement_brut_appartement'] = ((df['loyer_m2_appartement'] * 12) / df['prix_m2_appartement']) * 100
        df['rendement_brut_maison'] = ((df['loyer_m2_maison'] * 12) / df['prix_m2_maison']) * 100

        # Score d'attractivité d'achat (0 à 100)
        score_achat = (
            (df['revenu_fiscal_moyen'] / df['revenu_fiscal_moyen'].max()) * 40 +
            (df['rendement_brut_appartement'] / df['rendement_brut_appartement'].max()) * 40 +
            (1 - df['taux_vacance']) * 20
        )
        df['score_attractivite_achat'] = score_achat.clip(10, 99).round(1)

        # Score d'attractivité de location (0 à 100)
        loyer_effort = (df['loyer_m2_appartement'] * 60 * 12) / df['revenu_fiscal_moyen']
        score_loc = (100 - (loyer_effort * 100)).clip(15, 95).round(1)
        df['score_attractivite_location'] = score_loc

        self.df_features = df

    def get_city_profile(self, ville: str, departement: Optional[str] = None) -> Optional[pd.Series]:
        """
        Recherche le profil immobilier complet d'une ville (dernière année disponible).
        """
        df = self.df_features
        ville_clean = ville.strip().lower()
        
        mask = df['ville'].str.lower() == ville_clean
        if departement:
            dept_str = str(departement).zfill(2)
            mask = mask & (df['departement'] == dept_str)
            
        matches = df[mask]
        if matches.empty:
            # Recherche par correspondance partielle
            mask_partial = df['ville'].str.lower().str.contains(ville_clean, na=False)
            if departement:
                mask_partial = mask_partial & (df['departement'] == str(departement).zfill(2))
            matches = df[mask_partial]

        if not matches.empty:
            # Retourne la ligne la plus récente
            return matches.sort_values(by='annee', ascending=False).iloc[0]
        return None

