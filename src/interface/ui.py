"""
src/interface/ui.py — Interface UX/UI Senior pour ImmoPredict AI Frontline Concierge
Design épuré : Workspace unique, dédié à l'accueil, au triage et à la qualification de leads.
Zéro onglet secondaire d'exemple, zéro puce d'exemple sous le chat.
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
    """Gère le flux conversationnel et le rendu des escalades CRM."""
    if not message or not str(message).strip():
        return "Veuillez saisir votre message."
    
    result = agent.process_query(str(message))
    reply = result.get("text", "Une erreur est survenue lors du traitement.")
    
    if result.get("is_handover"):
        ticket_id = result.get("ticket_id", "TICKET-AUTO")
        reason = result.get("trigger_reason", "escalation")
        reply += (
            f"\n\n---\n"
            f"📋 **Ticket CRM créé :** `{ticket_id}`\n"
            f"🎯 **Motif de transfert :** `{reason}`\n"
            f"🤝 *Votre dossier de qualification complet a été transmis à un conseiller humain.*"
        )
    return reply


CUSTOM_CSS = """
/* Reset et container global */
.gradio-container {
    max-width: 1250px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Bannière d'en-tête professionnelle */
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #ffffff;
    padding: 24px 30px;
    border-radius: 16px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.header-text h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: #f8fafc;
    letter-spacing: -0.01em;
}

.header-text p {
    font-size: 0.92rem;
    color: #94a3b8;
    margin: 0;
}

/* Badge de statut actif 0s d'attente */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #10b981;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
}

.status-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
}

/* Cartes latérales */
.side-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
}

.dark .side-card {
    background: #1e293b;
    border-color: #334155;
    color: #f8fafc;
}

.side-card-title {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.dark .side-card-title {
    color: #94a3b8;
}

.step-row {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.step-box {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.85rem;
    padding: 8px 12px;
    background: #f8fafc;
    border-radius: 8px;
    border-left: 3px solid #cbd5e1;
}

.dark .step-box {
    background: #0f172a;
    border-left-color: #475569;
}

.step-box.highlight {
    border-left-color: #3b82f6;
    background: #eff6ff;
    font-weight: 600;
}

.dark .step-box.highlight {
    background: #1e3a8a;
    border-left-color: #60a5fa;
}
"""


def build_app():
    """Construit et retourne l'application Gradio (Workspace Unique & Épuré)."""
    with gr.Blocks(title="ImmoPredict AI — Frontline Concierge", css=CUSTOM_CSS) as demo:
        
        # En-tête exécutif épuré
        gr.HTML("""
        <div class="header-banner">
            <div class="header-text">
                <h1>🏠 ImmoPredict AI — Frontline Concierge</h1>
                <p>Assistant de Triage & Qualification de Leads en Temps Réel · Diagnostic & Transfert CRM</p>
            </div>
            <div class="status-pill">
                <span class="status-dot"></span> Agent IA Actif (0s d'attente)
            </div>
        </div>
        """)

        with gr.Row(equal_height=True):
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
