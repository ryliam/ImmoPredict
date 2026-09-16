import json
from typing import Dict, Any, List, Optional
from src.models.price_predictor import PricePredictor
from src.models.trend_forecaster import TrendForecaster
from src.models.zone_recommender import ZoneRecommender
from src.data_pipeline.feature_store import FeatureStore


# Singletons pour les modèles
_price_predictor = None
_trend_forecaster = None
_zone_recommender = None
_feature_store = None


def _get_services():
    global _price_predictor, _trend_forecaster, _zone_recommender, _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore.get_instance()
    if _price_predictor is None:
        _price_predictor = PricePredictor()
    if _trend_forecaster is None:
        _trend_forecaster = TrendForecaster()
    if _zone_recommender is None:
        _zone_recommender = ZoneRecommender()
    return _price_predictor, _trend_forecaster, _zone_recommender, _feature_store


def evaluate_property_tool(
    budget: float,
    action: str = "achat",
    ville: str = "Paris",
    type_bien: str = "appartement",
    surface_m2: float = 60.0,
    revenu_foyer_annuel: Optional[float] = None,
    departement: Optional[str] = None
) -> Dict[str, Any]:
    """
    Évalue si l'achat ou la location d'un bien est une bonne opportunité (Cas 1).
    """
    _, _, recommender, _ = _get_services()
    return recommender.evaluate_city_opportunity(
        budget=budget,
        action=action,
        ville=ville,
        type_bien=type_bien,
        surface_m2=surface_m2,
        revenu_foyer_annuel=revenu_foyer_annuel,
        departement=departement
    )


def forecast_profitability_tool(
    prix_achat: float,
    loyer_mensuel: Optional[float] = None,
    ville: Optional[str] = None,
    departement: Optional[str] = None,
    horizon_annees: int = 2
) -> Dict[str, Any]:
    """
    Prédit la rentabilité future d'un bien immobilier à 1, 2 ou 5 ans (Cas 2).
    """
    _, forecaster, _, _ = _get_services()
    return forecaster.forecast_profitability(
        prix_achat=prix_achat,
        loyer_mensuel=loyer_mensuel,
        ville=ville,
        departement=departement,
        horizon_annees=horizon_annees
    )


def recommend_best_zones_tool(
    action: str = "achat",
    budget: Optional[float] = None,
    revenu_foyer_annuel: Optional[float] = None,
    type_bien: str = "appartement",
    departement_pref: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Recommande le top des communes idéales pour acheter ou louer selon les ressources du foyer (Cas 3).
    """
    _, _, recommender, _ = _get_services()
    return recommender.recommend_top_zones(
        action=action,
        budget=budget,
        revenu_foyer_annuel=revenu_foyer_annuel,
        type_bien=type_bien,
        departement_pref=departement_pref,
        top_k=top_k
    )


def get_market_chart_data_tool(
    ville: Optional[str] = None,
    departement: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extrait les données de marché et tendances pour générer les graphiques d'analyse.
    """
    _, _, _, store = _get_services()
    
    # 1. Macro trends (IRL et Emprunts)
    macro_sample = store.df_macro.tail(24).copy()
    macro_data = {
        "dates": macro_sample['date'].astype(str).tolist(),
        "emprunts_M€": macro_sample['emprunts_M€'].tolist(),
        "irl": macro_sample['IRL'].tolist(),
        "taux_interet": macro_sample['taux_interet'].tolist()
    }

    # 2. Données territoriales si ville spécifiée
    city_info = None
    if ville:
        prof = store.get_city_profile(ville, departement)
        if prof is not None:
            city_info = {
                "ville": prof['ville'],
                "departement": prof['departement'],
                "prix_m2_appartement": round(prof['prix_m2_appartement'], 2),
                "prix_m2_maison": round(prof['prix_m2_maison'], 2),
                "loyer_m2_appartement": round(prof['loyer_m2_appartement'], 2),
                "loyer_m2_maison": round(prof['loyer_m2_maison'], 2),
                "revenu_fiscal_moyen": round(prof['revenu_fiscal_moyen'], 2),
                "rendement_brut_pct": round(prof['rendement_brut_appartement'], 2)
            }

    return {
        "macro": macro_data,
        "city": city_info
    }

