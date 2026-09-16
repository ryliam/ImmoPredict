import re
from typing import Dict, Any, Tuple, Optional
from src.mcp_server.server import MCPServer


class RealEstateAgent:
    """
    Agent IA conversationnel orchestrant les requêtes utilisateurs en langage naturel,
    l'appel aux outils MCP et la mise en forme experte des réponses avec métriques et graphiques.
    """

    def __init__(self):
        self.mcp = MCPServer()

    def process_query(self, user_message: str) -> Dict[str, Any]:
        """
        Analyse l'intention utilisateur en langage naturel, exécute le bon outil MCP
        et génère une réponse professionnelle structurée avec données graphiques.
        """
        msg_lower = user_message.lower()

        # Extraction d'entités par motifs
        numbers = [float(n.replace(" ", "").replace(",", ".")) for n in re.findall(r'\b\d+(?:[\s,.]\d+)?\b', user_message) if float(n.replace(" ", "").replace(",", ".")) > 50]
        
        # Détection type de bien
        type_bien = "maison" if "maison" in msg_lower or "house" in msg_lower else "appartement"
        
        # Détection action
        action = "location" if "louer" in msg_lower or "location" in msg_lower or "loyer" in msg_lower else "achat"

        # Détection de ville (mots avec majuscules ou après "à", "dans", "vers", "sur")
        ville_match = re.search(r'(?:à|a|dans|vers|sur|pour)\s+([A-ZÀ-ÿ][a-zà-ÿ\-]+(?:\s+[A-ZÀ-ÿ][a-zà-ÿ\-]+)*)', user_message)
        ville = ville_match.group(1).strip() if ville_match else None

        # -------------------------------------------------------------
        # CAS 0 : Salutations et questions conversationnelles générales
        # -------------------------------------------------------------
        greetings = ["bonjour", "salut", "hello", "bonsoir", "coucou", "hey", "yo", "aide", "help", "qui es-tu", "qui es tu", "merci", "merci beaucoup", "au revoir"]
        is_greeting = any(re.search(rf"\b{re.escape(g)}\b", msg_lower) for g in greetings)
        has_real_estate_intent = any(w in msg_lower for w in [
            "louer", "achat", "acheter", "location", "loyer", "rentable", "rentabilité", 
            "plus-value", "investir", "investissement", "prix", "ville", "zone", "foyer fiscal",
            "maison", "appartement", "budget", "m2", "endettement", "emprunt", "taux", "apport"
        ]) or bool(numbers) or (ville is not None)

        if (is_greeting and not has_real_estate_intent) or (not has_real_estate_intent and len(user_message.split()) < 4):
            return {
                "text": (
                    "👋 **Bonjour ! Je suis ImmoPredict AI, votre assistant d'aide à la décision immobilière.**\n\n"
                    "Je suis connecté aux bases de données officielles (DGFiP, DVF, Indices IRL, Taux de crédit) et à des modèles de Machine Learning pour vous guider.\n\n"
                    "#### 💡 Voici comment je peux vous aider :\n"
                    "1. **📍 Faisabilité & Opportunité :** *« J'ai 200 000 €, j'aimerais acheter un appartement à Ambérieu-en-Bugey, est-ce une bonne idée ? »*\n"
                    "2. **📈 Rentabilité à 2 ans :** *« Ce bien à 280 000 € sera-t-il rentable dans 2 ans ? »*\n"
                    "3. **🏆 Recommandation de villes :** *« Quel est le meilleur endroit pour acheter une maison avec 35 000 € de ressources fiscales ? »*\n\n"
                    "👉 *Indiquez-moi votre ville, votre budget ou votre situation pour lancer une analyse précise !*"
                ),
                "tool_called": None,
                "chart_type": None,
                "chart_data": None
            }

        # -------------------------------------------------------------
        # CAS 2 : Prédiction de rentabilité future ("rentable dans X ans", "plus-value")
        # -------------------------------------------------------------
        if any(w in msg_lower for w in ["rentable", "rentabilité", "plus-value", "2 ans", "dans 2 ans", "dans 3 ans", "futur"]):
            prix_achat = numbers[0] if numbers else 250000.0
            horizon = 2
            horizon_match = re.search(r'(\d+)\s*(?:an|ans|années)', msg_lower)
            if horizon_match:
                horizon = int(horizon_match.group(1))

            response = self.mcp.call_tool("forecast_profitability", {
                "prix_achat": prix_achat,
                "ville": ville,
                "horizon_annees": horizon
            })

            data = response.get("result", {})
            text_response = (
                f"### 📈 Analyse de Rentabilité Prévisionnelle à {horizon} ans\n\n"
                f"**Verdict :** {data.get('verdict')}\n\n"
                f"{data.get('explication')}\n\n"
                f"#### 📊 Chiffres Clés du Projet :\n"
                f"- **Prix d'acquisition :** {data.get('prix_achat_initial'):,.0f} €\n"
                f"- **Valeur estimée à {horizon} ans :** {data.get('valeur_future_estimee'):,.0f} € (Plus-value : `+{data.get('plus_value_estimee'):,.0f} €`)\n"
                f"- **Revenus locatifs bruts cumulés :** `+{data.get('total_loyers_bruts'):,.0f} €`\n"
                f"- **Charges & taxes estimées :** `-{data.get('total_charges_et_taxes'):,.0f} €`\n"
                f"- **Gain net global :** **`+{data.get('gain_total_net'):,.0f} €`**\n"
                f"- **Retour sur Investissement (ROI global) :** **`{data.get('roi_global_pct')}%`** (`{data.get('rendement_net_annuel_pct')}% / an net`)\n"
            )

            return {
                "text": text_response,
                "tool_called": "forecast_profitability",
                "chart_type": "forecast_timeline",
                "chart_data": data.get("timeline", [])
            }

        # -------------------------------------------------------------
        # CAS 3 : Recommandation de meilleures zones / villes ("meilleur endroit", "où acheter", "où louer")
        # -------------------------------------------------------------
        elif any(w in msg_lower for w in ["meilleur endroit", "meilleure ville", "meilleures zones", "où acheter", "où louer", "recommande", "foyer fiscal"]):
            budget = numbers[0] if (numbers and numbers[0] > 5000) else None
            revenu = numbers[0] if (numbers and numbers[0] <= 100000 and "revenu" in msg_lower) else (numbers[1] if len(numbers) > 1 else 32000.0)

            # Recherche d'un numéro de département
            dept_match = re.search(r'\b(0[1-9]|[1-8][0-9]|9[0-8]|2[abAB])\b', user_message)
            dept_pref = dept_match.group(1) if dept_match else None

            response = self.mcp.call_tool("recommend_best_zones", {
                "action": action,
                "budget": budget,
                "revenu_foyer_annuel": revenu,
                "type_bien": type_bien,
                "departement_pref": dept_pref,
                "top_k": 5
            })

            zones = response.get("result", [])
            text_response = (
                f"### 🏆 Top 5 des Meilleures Villes pour votre Profil ({action.upper()})\n\n"
                f"Sélection basée sur votre profil fiscal (revenus : ~`{revenu:,.0f} €/an`), l'accessibilité financière et la dynamique locale :\n\n"
            )

            for i, z in enumerate(zones, 1):
                text_response += (
                    f"**{i}. {z['ville']} (Dép. {z['departement']})** — *Score : {z['score_recommandation']}/100*\n"
                    f"- Prix moyen : `{z['prix_m2']:,.0f} €/m²` | Loyer moyen : `{z['loyer_m2']} €/m²`\n"
                    f"- Rendement brut estimé : **`{z['rendement_brut_pct']}%`** | Vacance locative : `{z['taux_vacance_pct']}%`\n\n"
                )

            return {
                "text": text_response,
                "tool_called": "recommend_best_zones",
                "chart_type": "top_zones_bar",
                "chart_data": zones
            }

        # -------------------------------------------------------------
        # CAS 1 (Défaut) : Évaluation d'opportunité d'une ville / projet spécifique
        # -------------------------------------------------------------
        else:
            ville_target = ville or "Ambérieu-en-Bugey"
            budget = numbers[0] if numbers else (250000.0 if action == "achat" else 800.0)

            response = self.mcp.call_tool("evaluate_property", {
                "budget": budget,
                "action": action,
                "ville": ville_target,
                "type_bien": type_bien,
                "surface_m2": 60.0
            })

            data = response.get("result", {})
            text_response = (
                f"### 📍 Évaluation du Projet : {action.capitalize()} à {data.get('ville', ville_target)}\n\n"
                f"**Verdict :** {data.get('verdict', 'Analyse effectuée')}\n\n"
                f"{data.get('conseil', '')}\n\n"
                f"#### 🔎 Indicateurs Marché :\n"
                f"- **Score d'opportunité :** **`{data.get('score_opportunite', 50)}/100`**\n"
                f"- **Prix moyen estimé (60m²) :** `{data.get('prix_achat_estime', 0):,.0f} €`\n"
                f"- **Loyer moyen estimé (60m²) :** `{data.get('loyer_mensuel_estime', 0):,.0f} € / mois`\n"
                f"- **Taux d'effort estimé :** `{data.get('taux_effort_pct', 0)}%`\n"
                f"- **Vacance des logements :** `{data.get('taux_vacance_pct', 0)}%`\n"
            )

            # Enrichissement avec les données macro pour le graphique
            chart_res = self.mcp.call_tool("get_market_chart_data", {"ville": data.get('ville', ville_target)})

            return {
                "text": text_response,
                "tool_called": "evaluate_property",
                "chart_type": "macro_market",
                "chart_data": chart_res.get("result", {})
            }

