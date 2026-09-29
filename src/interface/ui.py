"""
src/interface/ui.py — Interface UX/UI Senior pour ImmoPredict AI Frontline Concierge
Design : Workspace moderne, tableau de bord de qualification de lead 24/7.
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
/* Theme overrides & custom UX styling */
.main-container {
    max-width: 1300px;
    margin: 0 auto;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.header-banner {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: #ffffff;
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-title h1 {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: #f8fafc;
    letter-spacing: -0.02em;
}

.header-title p {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #10b981;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
}

.sidebar-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.dark .sidebar-card {
    background: #1e293b;
    border-color: #334155;
    color: #f8fafc;
}

.card-title {
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.dark .card-title {
    color: #94a3b8;
}

.step-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.88rem;
    padding: 8px 12px;
    background: #f8fafc;
    border-radius: 8px;
    border-left: 3px solid #cbd5e1;
}

.dark .step-item {
    background: #0f172a;
    border-left-color: #475569;
}

.step-item.active {
    border-left-color: #3b82f6;
    background: #eff6ff;
    font-weight: 600;
}

.dark .step-item.active {
    background: #1e3a8a;
    border-left-color: #60a5fa;
}
"""


def build_app():
    """Construit et retourne l'instance Gradio Blocks révisée (UX/UI Senior)."""
    with gr.Blocks(title="ImmoPredict AI — Frontline Concierge", css=CUSTOM_CSS) as demo:
        
        # En-tête exécutif moderne
        gr.HTML("""
        <div class="header-banner">
            <div class="header-title">
                <h1>🏠 ImmoPredict AI — Frontline Concierge</h1>
                <p>Assistant de triage & qualification de leads en temps réel · Diagnostic & Transfert CRM</p>
            </div>
            <div class="status-badge">
                <span class="status-dot"></span> Agent IA Actif (0s d'attente)
            </div>
        </div>
        """)

        with gr.Row(equal_height=True):
            # Colonne Principale (Chatbot de Triage) - 70%
            with gr.Column(scale=7):
                gr.ChatInterface(
                    fn=chat_response,
                    textbox=gr.Textbox(
                        placeholder="Écrivez votre message ou présentez votre projet d'investissement...",
                        container=False,
                        scale=7
                    )
                )

            # Colonne Latérale (Tableau de Bord de Qualification & Garanties UX) - 30%
            with gr.Column(scale=3):
                gr.HTML("""
                <div class="sidebar-card">
                    <div class="card-title">🛡️ Garanties de l'Assistant</div>
                    <div class="step-list">
                        <div class="step-item active">
                            <span>⚡</span> <strong>Accueil Instantané :</strong> 0s de latence 24/7
                        </div>
                        <div class="step-item active">
                            <span>📚</span> <strong>RAG Strict :</strong> Source certifiée entreprise
                        </div>
                        <div class="step-item active">
                            <span>🔄</span> <strong>Machine à États :</strong> Progression à votre rythme
                        </div>
                        <div class="step-item active">
                            <span>🤝</span> <strong>Handover CRM :</strong> Escalade humaine transparente
                        </div>
                    </div>
                </div>

                <div class="sidebar-card">
                    <div class="card-title">📊 Parcours de Qualification</div>
                    <div class="step-list">
                        <div class="step-item">
                            <span>1️⃣</span> Accueil & Présentation des Services
                        </div>
                        <div class="step-item">
                            <span>2️⃣</span> Identification du Besoin Immobilier
                        </div>
                        <div class="step-item">
                            <span>3️⃣</span> Qualification Budget & Zone
                        </div>
                        <div class="step-item">
                            <span>4️⃣</span> Handover & Prise de RDV Conseiller
                        </div>
                    </div>
                </div>

                <div class="sidebar-card">
                    <div class="card-title">ℹ️ Information Handover</div>
                    <p style="font-size: 0.85rem; color: #64748b; margin: 0; line-height: 1.4;">
                        Dès que votre projet est qualifié ou si votre demande dépasse le périmètre certifié, 
                        un <strong>Ticket CRM</strong> est automatiquement émis pour votre suivi commercial.
                    </p>
                </div>
                """)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
