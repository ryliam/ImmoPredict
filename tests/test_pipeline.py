import pytest
from src.data_pipeline.feature_store import FeatureStore
from src.models.price_predictor import PricePredictor
from src.models.trend_forecaster import TrendForecaster
from src.models.zone_recommender import ZoneRecommender
from src.mcp_server.server import MCPServer
from src.agent.agent_service import RealEstateAgent


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


def test_agent_nlp_use_cases():
    """Vérifie que l'agent IA répond correctement aux 3 cas d'usage."""
    agent = RealEstateAgent()
    
    # Cas 0 : Salutation simple
    r0 = agent.process_query("bonjour")
    assert r0["tool_called"] is None
    assert r0["chart_type"] is None
    assert "ImmoPredict AI" in r0["text"]

    # Cas 1 : Faisabilité
    r1 = agent.process_query("J'ai 200 000 €, j'aimerais acheter un appartement à Ambérieu-en-Bugey, est-ce une bonne idée ?")
    assert r1["tool_called"] == "evaluate_property"
    assert len(r1["text"]) > 50

    # Cas 2 : Rentabilité à 2 ans
    r2 = agent.process_query("Ce bien à 300 000 € sera-t-il rentable dans 2 ans ?")
    assert r2["tool_called"] == "forecast_profitability"
    assert r2["chart_type"] == "forecast_timeline"

    # Cas 3 : Recommandation
    r3 = agent.process_query("Quel est le meilleur endroit pour acheter une maison avec 35 000 € de ressources ?")
    assert r3["tool_called"] == "recommend_best_zones"
    assert r3["chart_type"] == "top_zones_bar"


