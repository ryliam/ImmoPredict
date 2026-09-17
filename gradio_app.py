"""
Gradio Web Interface for ImmoPredict AI on Hugging Face Spaces
"""
import sys
from pathlib import Path
import pandas as pd
import gradio as gr

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.agent_service import RealEstateAgent
from src.mcp_server.tools import (
    evaluate_property_tool,
    forecast_profitability_tool,
    recommend_best_zones_tool,
    get_market_chart_data_tool
)
from src.ui.charts import (
    create_forecast_timeline_chart,
    create_top_zones_chart,
    create_macro_market_chart
)

agent = RealEstateAgent()


def chat_response(message, history):
    """Handles chat messages with the Real Estate Agent."""
    if not message.strip():
        return "Veuillez poser une question."
    result = agent.process_query(message)
    return result.get("text", "Une erreur est survenue lors de l'analyse.")


def evaluate_feasibility(action, ville, budget, surface, type_bien, revenu_annuel):
    """Cas 1 : Faisabilité & Opportunité d'un projet."""
    try:
        res = evaluate_property_tool(
            budget=float(budget),
            action=action.lower(),
            ville=ville.strip(),
            type_bien=type_bien.lower(),
            surface_m2=float(surface),
            revenu_foyer_annuel=float(revenu_annuel) if revenu_annuel else None
        )
        if "erreur" in res:
            return f"❌ {res['erreur']}", None

        verdict = res.get("verdict", "")
        taux_effort = f"{res.get('taux_effort_pct', 0):.1f} %" if res.get("taux_effort_pct") else "N/A"
        
        md_res = f"""
### 📊 Résultat de l'analyse — {res.get('ville', ville)} ({res.get('departement', '')})

- **Verdict :** **{verdict.upper()}**
- **Type de bien :** {res.get('type_bien')} ({res.get('surface_m2')} m²)
- **Prix / Loyer estimé :** {res.get('estimation_prix_ou_loyer', 0):,.0f} €
- **Prix moyen constaté :** {res.get('prix_ou_loyer_moyen_m2', 0):.1f} €/m²
- **Taux d'effort calculé :** {taux_effort}
- **Taux de vacance locale :** {res.get('taux_vacance_pct', 0):.1f} %
- **Revenu moyen des ménages :** {res.get('revenu_moyen_commune', 0):,.0f} €/an

> **💡 Conseil expert :** {res.get('conseil', '')}
        """

        # Chart
        chart_data = get_market_chart_data_tool(ville=ville)
        fig = create_macro_market_chart(chart_data)
        return md_res, fig
    except Exception as e:
        return f"❌ Erreur lors de l'évaluation : {str(e)}", None


def simulate_profitability(prix_achat, loyer_mensuel, ville, horizon):
    """Cas 2 : Rentabilité prévisionnelle."""
    try:
        res = forecast_profitability_tool(
            prix_achat=float(prix_achat),
            loyer_mensuel=float(loyer_mensuel) if loyer_mensuel else None,
            ville=ville.strip() if ville else None,
            horizon_annees=int(horizon)
        )
        if "erreur" in res:
            return f"❌ {res['erreur']}", None

        md_res = f"""
### 📈 Simulation de Rentabilité à {horizon} ans

- **Prix d'acquisition :** {res.get('prix_achat', 0):,.0f} €
- **Rendement locatif brut :** **{res.get('rendement_brut_initial_pct', 0):.2f} %**
- **Loyer mensuel projeté (an {horizon}) :** {res.get('loyer_mensuel_futur', 0):,.0f} €/mois
- **Plus-value prévisionnelle :** {res.get('plus_value_estimee', 0):,.0f} €
- **Cash-flow net cumulé :** {res.get('cashflow_net_cumule', 0):,.0f} €
- **Gain financier total :** **{res.get('gain_total_estime', 0):,.0f} €**
- **ROI global estimé :** **{res.get('roi_global_pct', 0):.2f} %**

> **💡 Commentaire :** {res.get('analyse', '')}
        """

        chart_data = res.get("projection_annuelle", [])
        fig = create_forecast_timeline_chart(chart_data)
        return md_res, fig
    except Exception as e:
        return f"❌ Erreur lors de la simulation : {str(e)}", None


def recommend_zones(action, budget, revenu_annuel, type_bien, top_k):
    """Cas 3 : Recommandation de meilleures communes."""
    try:
        zones = recommend_best_zones_tool(
            action=action.lower(),
            budget=float(budget) if budget else None,
            revenu_foyer_annuel=float(revenu_annuel) if revenu_annuel else None,
            type_bien=type_bien.lower(),
            top_k=int(top_k)
        )
        if not zones:
            return "Aucune zone trouvée pour ces critères.", None

        df_display = pd.DataFrame(zones)
        fig = create_top_zones_chart(zones)
        return df_display, fig
    except Exception as e:
        return f"❌ Erreur lors de la recommandation : {str(e)}", None


# Construction de l'interface Gradio
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")

with gr.Blocks(title="ImmoPredict AI", theme=theme) as demo:
    gr.Markdown("""
    # 🏠 ImmoPredict AI — Conseil & Prédiction Immobilière
    *Plateforme d'aide à la décision basée sur le Machine Learning, les données DGFiP/DVF/INSEE et l'architecture MCP.*
    """)

    with gr.Tabs():
        # TAB 1 : Chatbot IA
        with gr.TabItem("💬 Conseiller Conversationnel IA"):
            gr.Markdown("Posez votre question en langage naturel pour une évaluation instantanée.")
            gr.ChatInterface(
                fn=chat_response,
                examples=[
                    "J'ai 200 000 €, j'aimerais acheter un appartement à Ambérieu-en-Bugey, est-ce une bonne idée ?",
                    "Ce bien à 280 000 € sera-t-il rentable dans 2 ans ?",
                    "Quel est le meilleur endroit pour acheter une maison avec un foyer fiscal de 38 000 € ?"
                ]
            )

        # TAB 2 : Faisabilité & Opportunité (Cas 1)
        with gr.TabItem("📍 Faisabilité & Opportunité"):
            with gr.Row():
                with gr.Column():
                    c1_action = gr.Radio(["Achat", "Location"], label="Projet", value="Achat")
                    c1_ville = gr.Textbox(label="Ville ou Commune", value="Ambérieu-en-Bugey")
                    c1_budget = gr.Number(label="Budget total (€)", value=200000)
                    c1_surface = gr.Number(label="Surface souhaitée (m²)", value=65)
                    c1_type = gr.Radio(["appartement", "maison"], label="Type de bien", value="appartement")
                    c1_revenu = gr.Number(label="Revenu annuel fiscal du foyer (€, optionnel)", value=35000)
                    c1_btn = gr.Button("🔍 Analyser l'opportunité", variant="primary")

                with gr.Column():
                    c1_out_md = gr.Markdown()
                    c1_out_plot = gr.Plot(label="Graphique Marché")

            c1_btn.click(
                evaluate_feasibility,
                inputs=[c1_action, c1_ville, c1_budget, c1_surface, c1_type, c1_revenu],
                outputs=[c1_out_md, c1_out_plot]
            )

        # TAB 3 : Rentabilité 2 ans (Cas 2)
        with gr.TabItem("📈 Simulateur Rentabilité & ROI"):
            with gr.Row():
                with gr.Column():
                    c2_prix = gr.Number(label="Prix d'achat du bien (€)", value=280000)
                    c2_loyer = gr.Number(label="Loyer mensuel espéré (€, optionnel)", value=1100)
                    c2_ville = gr.Textbox(label="Ville (optionnel)", value="Lyon")
                    c2_horizon = gr.Slider(minimum=1, maximum=10, step=1, value=2, label="Horizon de détention (années)")
                    c2_btn = gr.Button("📊 Simuler la rentabilité", variant="primary")

                with gr.Column():
                    c2_out_md = gr.Markdown()
                    c2_out_plot = gr.Plot(label="Trajectoire ROI & Cash-Flow")

            c2_btn.click(
                simulate_profitability,
                inputs=[c2_prix, c2_loyer, c2_ville, c2_horizon],
                outputs=[c2_out_md, c2_out_plot]
            )

        # TAB 4 : Recommandation Top Villes (Cas 3)
        with gr.TabItem("🏆 Recommandation de Communes"):
            with gr.Row():
                with gr.Column():
                    c3_action = gr.Radio(["Achat", "Location"], label="Action", value="Achat")
                    c3_budget = gr.Number(label="Budget (€)", value=250000)
                    c3_revenu = gr.Number(label="Revenu fiscal annuel du foyer (€)", value=40000)
                    c3_type = gr.Radio(["appartement", "maison"], label="Type de bien", value="appartement")
                    c3_top_k = gr.Slider(minimum=3, maximum=15, step=1, value=5, label="Nombre de villes recommandées")
                    c3_btn = gr.Button("🏆 Trouver les meilleures communes", variant="primary")

                with gr.Column():
                    c3_out_df = gr.DataFrame(label="Top Communes Recommandées")
                    c3_out_plot = gr.Plot(label="Comparatif des Villes")

            c3_btn.click(
                recommend_zones,
                inputs=[c3_action, c3_budget, c3_revenu, c3_type, c3_top_k],
                outputs=[c3_out_df, c3_out_plot]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

