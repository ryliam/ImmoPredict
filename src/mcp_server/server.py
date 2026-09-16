import json
from typing import Dict, Any, List
from src.mcp_server.tools import (
    evaluate_property_tool,
    forecast_profitability_tool,
    recommend_best_zones_tool,
    get_market_chart_data_tool
)

# Schémas des outils MCP exposés au LLM
MCP_TOOLS_SCHEMAS = [
    {
        "name": "evaluate_property",
        "description": "Évalue la faisabilité et l'opportunité financière d'acheter ou de louer un bien dans une ville avec un budget donné (Cas 1).",
        "parameters": {
            "type": "object",
            "properties": {
                "budget": {"type": "number", "description": "Budget d'achat ou de loyer disponible en euros."},
                "action": {"type": "string", "enum": ["achat", "location"], "description": "Action envisagée ('achat' ou 'location')."},
                "ville": {"type": "string", "description": "Nom de la commune ciblée."},
                "type_bien": {"type": "string", "enum": ["appartement", "maison"], "description": "Type de logement."},
                "surface_m2": {"type": "number", "description": "Surface en m2 envisagée (défaut 60m2)."},
                "revenu_foyer_annuel": {"type": "number", "description": "Revenu fiscal annuel du foyer en euros (optionnel)."},
                "departement": {"type": "string", "description": "Numéro du département (optionnel, ex: '01', '69', '75')."}
            },
            "required": ["budget", "action", "ville"]
        }
    },
    {
        "name": "forecast_profitability",
        "description": "Prédit la rentabilité financière, la plus-value et le gain net d'un bien immobilier à horizon 1 à 5 ans (Cas 2).",
        "parameters": {
            "type": "object",
            "properties": {
                "prix_achat": {"type": "number", "description": "Prix d'acquisition du bien en euros."},
                "loyer_mensuel": {"type": "number", "description": "Loyer mensuel perçu estimé (optionnel)."},
                "ville": {"type": "string", "description": "Nom de la ville de localisation."},
                "departement": {"type": "string", "description": "Numéro de département (optionnel)."},
                "horizon_annees": {"type": "integer", "description": "Horizon temporel de l'investissement en années (défaut: 2 ans)."}
            },
            "required": ["prix_achat"]
        }
    },
    {
        "name": "recommend_best_zones",
        "description": "Recommande le top des meilleures villes pour acheter ou louer selon les ressources financières et le budget du foyer (Cas 3).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["achat", "location"], "description": "Objectif : 'achat' ou 'location'."},
                "budget": {"type": "number", "description": "Budget disponible en euros (optionnel)."},
                "revenu_foyer_annuel": {"type": "number", "description": "Revenu annuel du foyer fiscal en euros (optionnel)."},
                "type_bien": {"type": "string", "enum": ["appartement", "maison"], "description": "Type de bien recherché."},
                "departement_pref": {"type": "string", "description": "Département spécifique recherché (ex: '69', '13', optionnel)."},
                "top_k": {"type": "integer", "description": "Nombre de recommandations (défaut 5)."}
            },
            "required": ["action"]
        }
    },
    {
        "name": "get_market_chart_data",
        "description": "Fournit les données de séries temporelles de marché (IRL, taux d'intérêt, emprunts) et prix au m² pour affichage graphique.",
        "parameters": {
            "type": "object",
            "properties": {
                "ville": {"type": "string", "description": "Nom de la commune (optionnel)."},
                "departement": {"type": "string", "description": "Département (optionnel)."}
            }
        }
    }
]


class MCPServer:
    """
    Serveur d'outils MCP pour exécuter les requêtes de l'Agent IA.
    """

    def __init__(self):
        self.tools = {
            "evaluate_property": evaluate_property_tool,
            "forecast_profitability": forecast_profitability_tool,
            "recommend_best_zones": recommend_best_zones_tool,
            "get_market_chart_data": get_market_chart_data_tool
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        return MCP_TOOLS_SCHEMAS

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"error": f"Outil '{tool_name}' non reconnu par le serveur MCP."}
        
        try:
            func = self.tools[tool_name]
            result = func(**arguments)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

