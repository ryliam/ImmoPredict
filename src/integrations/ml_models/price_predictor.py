import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional
from src.integrations.data_pipeline.feature_store import FeatureStore
from config.settings import (
    PATH_MODEL_PRIX_APT,
    PATH_MODEL_PRIX_MAI,
    PATH_MODEL_LOYER_APT,
    PATH_MODEL_LOYER_MAI,
    PATH_SCALER
)


class PricePredictor:
    """
    Modèle de Machine Learning pour prédire les prix d'achat et les loyers au m²
    d'un appartement ou d'une maison selon la commune et ses caractéristiques.
    Les modèles entraînés sont automatiquement persistés sur disque via joblib.
    """

    def __init__(self):
        self.feature_store = FeatureStore.get_instance()
        self.model_prix_apt = None
        self.model_prix_mai = None
        self.model_loyer_apt = None
        self.model_loyer_mai = None
        self.scaler = None
        self.feature_names = [
            'revenu_fiscal_moyen', 'montant_impot_moyen', 'n_foyers_fiscaux',
            'n_logements_vacants', 'taux_vacance'
        ]
        self._load_or_train_models()

    def _load_or_train_models(self):
        """
        Charge les modèles sauvegardés sur disque s'ils existent,
        ou les entraîne une unique fois et les sauvegarde avec joblib.
        """
        all_models_exist = (
            PATH_MODEL_PRIX_APT.exists() and
            PATH_MODEL_PRIX_MAI.exists() and
            PATH_MODEL_LOYER_APT.exists() and
            PATH_MODEL_LOYER_MAI.exists() and
            PATH_SCALER.exists()
        )

        if all_models_exist:
            # Chargement ultra-rapide (<50ms) sans réentraînement
            self.model_prix_apt = joblib.load(PATH_MODEL_PRIX_APT)
            self.model_prix_mai = joblib.load(PATH_MODEL_PRIX_MAI)
            self.model_loyer_apt = joblib.load(PATH_MODEL_LOYER_APT)
            self.model_loyer_mai = joblib.load(PATH_MODEL_LOYER_MAI)
            self.scaler = joblib.load(PATH_SCALER)
        else:
            # Entraînement initial
            self._train_and_save_models()

    def _train_and_save_models(self):
        """
        Entraîne les régresseurs Random Forest sur le dataset territorial consolidé
        et les persiste sur disque via joblib.
        """
        df = self.feature_store.df_features.dropna(subset=self.feature_names).copy()

        X = df[self.feature_names]
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model_prix_apt = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=12)
        self.model_prix_mai = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=12)
        self.model_loyer_apt = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=12)
        self.model_loyer_mai = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=12)

        self.model_prix_apt.fit(X_scaled, df['prix_m2_appartement'])
        self.model_prix_mai.fit(X_scaled, df['prix_m2_maison'])
        self.model_loyer_apt.fit(X_scaled, df['loyer_m2_appartement'])
        self.model_loyer_mai.fit(X_scaled, df['loyer_m2_maison'])

        # Sauvegarde sur disque pour éviter tout réentraînement ultérieur
        joblib.dump(self.model_prix_apt, PATH_MODEL_PRIX_APT)
        joblib.dump(self.model_prix_mai, PATH_MODEL_PRIX_MAI)
        joblib.dump(self.model_loyer_apt, PATH_MODEL_LOYER_APT)
        joblib.dump(self.model_loyer_mai, PATH_MODEL_LOYER_MAI)
        joblib.dump(self.scaler, PATH_SCALER)

    def predict(
        self,
        ville: str,
        type_bien: str = "appartement",
        surface_m2: float = 60.0,
        departement: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prédit le prix d'achat total et le loyer mensuel estimé pour un bien donné.
        """
        type_clean = type_bien.strip().lower()
        is_apt = "app" in type_clean or "flat" in type_clean
        
        city_data = self.feature_store.get_city_profile(ville, departement)
        
        if city_data is not None:
            loyer_m2 = city_data['loyer_m2_appartement'] if is_apt else city_data['loyer_m2_maison']
            prix_m2 = city_data['prix_m2_appartement'] if is_apt else city_data['prix_m2_maison']
            dept_found = city_data['departement']
            ville_found = city_data['ville']
            revenu_moyen = city_data['revenu_fiscal_moyen']
            taux_vacance = city_data['taux_vacance']
        else:
            default_features = np.array([[30000.0, 1800.0, 2000.0, 100.0, 0.08]])
            scaled = self.scaler.transform(default_features)
            loyer_m2 = float(self.model_loyer_apt.predict(scaled)[0] if is_apt else self.model_loyer_mai.predict(scaled)[0])
            prix_m2 = float(self.model_prix_apt.predict(scaled)[0] if is_apt else self.model_prix_mai.predict(scaled)[0])
            dept_found = departement or "National"
            ville_found = ville
            revenu_moyen = 30000.0
            taux_vacance = 0.08

        loyer_mensuel = round(loyer_m2 * surface_m2, 2)
        prix_achat_total = round(prix_m2 * surface_m2, 2)
        rendement_brut = round(((loyer_mensuel * 12) / prix_achat_total) * 100, 2) if prix_achat_total > 0 else 0.0

        return {
            "ville": ville_found,
            "departement": dept_found,
            "type_bien": "Appartement" if is_apt else "Maison",
            "surface_m2": surface_m2,
            "loyer_m2": round(loyer_m2, 2),
            "prix_m2": round(prix_m2, 2),
            "loyer_mensuel_estime": loyer_mensuel,
            "prix_achat_estime": prix_achat_total,
            "rendement_locatif_brut_pct": rendement_brut,
            "revenu_fiscal_moyen": round(revenu_moyen, 2),
            "taux_vacance_pct": round(taux_vacance * 100, 2)
        }

