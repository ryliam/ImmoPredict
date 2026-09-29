"""
Interface Web Gradio pour ImmoPredict AI - Assistant Conversationnel Exclusif.
Respect strict du mandat : Uniquement un assistant conversationnel, aucune interface secondaire.
"""
import sys
from pathlib import Path
import gradio as gr

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.agent import RealEstateAgent

agent = RealEstateAgent()


def chat_response(message, history):
    """Gère les messages conversationnels avec l'agent ImmoPredict AI."""
    if not message or not str(message).strip():
        return "Veuillez poser une question."
    result = agent.process_query(str(message))
    return result.get("text", "Une erreur est survenue lors de l'analyse.")


CUSTOM_CSS = """
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.header-banner {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: #ffffff;
    padding: 24px 30px;
    border-radius: 14px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
}
.header-title {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.header-badge {
    background: #2563eb;
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.header-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.4;
}
.side-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
.side-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.step-row {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.step-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 0.84rem;
    color: #475569;
    display: flex;
    align-items: center;
    gap: 10px;
}
.step-box.highlight {
    background: #eff6ff;
    border-color: #bfdbfe;
    color: #1d4ed8;
    font-weight: 500;
}
"""


def build_app():
    """Construit l'application Gradio strictement conversationnelle."""
    with gr.Blocks(title="ImmoPredict AI — Assistant Conversationnel", css=CUSTOM_CSS) as demo:
        gr.HTML("""
        <div class="header-banner">
            <div class="header-title">
                <span>🏢 ImmoPredict AI</span>
                <span class="header-badge">Frontline AI Concierge</span>
            </div>
            <p class="header-subtitle">
                Assistant conversationnel intelligent pour l'accueil instantané, la réponse certifiée RAG et la qualification progressive de leads immobiliers.
            </p>
        </div>
        """)

        with gr.Row():
            # Colonne Principale : Interface de Chat totalement libérée (70%)
            with gr.Column(scale=7):
                gr.ChatInterface(
                    fn=chat_response,
                    textbox=gr.Textbox(
                        placeholder="Écrivez votre message ou décrivez votre projet d'investissement...",
                        container=False,
                        scale=7
                    )
                )

            # Colonne Latérale : Tableau de Bord & Indicateurs de Qualification (30%)
            with gr.Column(scale=3):
                gr.HTML("""
                <div class="side-card">
                    <div class="side-card-title">🛡️ Garanties de Prise en Charge</div>
                    <div class="step-row">
                        <div class="step-box highlight">
                            <span>⚡</span> <strong>Accueil Instantané :</strong> Zéro latence 24/7
                        </div>
                        <div class="step-box highlight">
                            <span>📚</span> <strong>RAG Strict :</strong> Connaissances certifiées
                        </div>
                        <div class="step-box highlight">
                            <span>🔄</span> <strong>Machine à États :</strong> Suivi souple
                        </div>
                        <div class="step-box highlight">
                            <span>🤝</span> <strong>Handover CRM :</strong> Escalade humaine qualifiée
                        </div>
                    </div>
                </div>

                <div class="side-card">
                    <div class="side-card-title">📊 Étapes de Qualification</div>
                    <div class="step-row">
                        <div class="step-box">
                            <span>1️⃣</span> Présentation des Outils & Mission
                        </div>
                        <div class="step-box">
                            <span>2️⃣</span> Définition du Projet (Achat / Locatif)
                        </div>
                        <div class="step-box">
                            <span>3️⃣</span> Recueil Budget, Surface & Zone
                        </div>
                        <div class="step-box">
                            <span>4️⃣</span> Transfert & Clôture avec Conseiller
                        </div>
                    </div>
                </div>

                <div class="side-card">
                    <div class="side-card-title">ℹ️ Escalade Commerciale</div>
                    <p style="font-size: 0.83rem; color: #64748b; margin: 0; line-height: 1.45;">
                        Dès que votre profil est qualifié ou si votre demande dépasse le périmètre certifié, 
                        un <strong>Ticket CRM</strong> est généré pour transmettre votre dossier sans perte de contexte.
                    </p>
                </div>
                """)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
