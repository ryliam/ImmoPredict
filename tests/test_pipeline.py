import pytest
from unittest.mock import MagicMock
from src.integrations.data_pipeline.feature_store import FeatureStore
from src.integrations.ml_models.price_predictor import PricePredictor
from src.integrations.ml_models.trend_forecaster import TrendForecaster
from src.integrations.ml_models.zone_recommender import ZoneRecommender
from src.integrations.mcp_server.server import MCPServer
from src.core.agent import RealEstateAgent


def test_feature_store_integrity():
    """Vérifie que les datasets nettoyés ne contiennent aucun NaN dans les colonnes critiques."""
    store = FeatureStore.get_instance()
    assert store.df_macro is not None
    assert store.df_macro.isna().sum().sum() == 0, "df_macro ne doit contenir aucun NaN"
    assert store.df_territorial is not None
    assert store.df_territorial['n_foyers_fiscaux'].isna().sum() == 0, "n_foyers_fiscaux ne doit contenir aucun NaN"
    assert store.df_territorial['revenu_fiscal_moyen'].isna().sum() == 0, "revenu_fiscal_moyen ne doit contenir aucun NaN"


def test_price_predictor():
    """Vérifie le fonctionnement du modèle de prédiction de prix et loyers."""
    predictor = PricePredictor()
    pred = predictor.predict(ville="Ambérieu-en-Bugey", type_bien="appartement", surface_m2=60.0)
    
    assert "prix_achat_estime" in pred
    assert "loyer_mensuel_estime" in pred
    assert pred["prix_achat_estime"] > 0
    assert pred["loyer_mensuel_estime"] > 0
    assert pred["rendement_locatif_brut_pct"] > 0


def test_trend_forecaster():
    """Vérifie la simulation de rentabilité à 2 ans (Cas 2)."""
    forecaster = TrendForecaster()
    forecast = forecaster.forecast_profitability(prix_achat=250000.0, horizon_annees=2)
    
    assert forecast["valeur_future_estimee"] > 0
    assert forecast["horizon_annees"] == 2
    assert "timeline" in forecast
    assert len(forecast["timeline"]) >= 2
    assert "roi_global_pct" in forecast


def test_zone_recommender():
    """Vérifie le moteur d'opportunité (Cas 1) et de recommandation (Cas 3)."""
    recommender = ZoneRecommender()
    
    # Test Cas 1
    eval_res = recommender.evaluate_city_opportunity(
        budget=200000.0,
        action="achat",
        ville="Ambérieu-en-Bugey"
    )
    assert "score_opportunite" in eval_res
    assert "bonne_idee" in eval_res

    # Test Cas 3
    top_zones = recommender.recommend_top_zones(action="achat", revenu_foyer_annuel=35000.0, top_k=5)
    assert len(top_zones) == 5
    assert all("score_recommandation" in z for z in top_zones)


def test_mcp_server_dispatch():
    """Vérifie le dispatching des outils via le serveur MCP."""
    server = MCPServer()
    tools = server.list_tools()
    assert len(tools) == 4
    
    res = server.call_tool("evaluate_property", {
        "budget": 180000.0,
        "action": "achat",
        "ville": "Ambérieu-en-Bugey"
    })
    assert res["status"] == "success"
    assert "result" in res


def test_agent_triage_flow():
    """Vérifie que l'agent conversationnel traite correctement une requête avec le LLM."""
    agent = RealEstateAgent()
    res = agent.process_query("Bonjour, qui êtes-vous et est-ce gratuit ?")
    assert "text" in res
    assert res.get("is_handover") is False
    assert len(res["text"]) > 20
