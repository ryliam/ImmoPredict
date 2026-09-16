import streamlit as st
import sys
from pathlib import Path

# Ajout de la racine au sys.path pour les imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.agent_service import RealEstateAgent
from src.ui.charts import (
    create_forecast_timeline_chart,
    create_top_zones_chart,
    create_macro_market_chart
)

# Configuration de la page
st.set_page_config(
    page_title="ImmoPredict AI — Conseil & Prédiction Immobilière",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .chat-bubble {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_agent():
    return RealEstateAgent()


agent = get_agent()

# Initialisation de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Bonjour ! Je suis votre conseiller IA en investissement immobilier.**\n\nJe peux analyser la faisabilité d'un achat/location, prédire la rentabilité future d'un bien à 2 ans, ou vous recommander les meilleures villes selon votre budget et vos revenus fiscaux.\n\nPosez-moi votre question en langage naturel ci-dessous !",
            "chart_type": None,
            "chart_data": None
        }
    ]

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/real-estate.png", width=70)
    st.markdown("### 🏠 **ImmoPredict AI**")
    st.markdown("*Plateforme d'aide à la décision basée sur le Machine Learning & MCP.*")
    st.markdown("---")

    st.markdown("#### 💡 **Exemples de questions rapides :**")
    
    if st.button("📍 1. Faisabilité d'un Achat à Ambérieu"):
        st.session_state.user_prompt_input = "J'ai 200 000 €, j'aimerais acheter un appartement à Ambérieu-en-Bugey, est-ce une bonne idée ?"

    if st.button("📈 2. Rentabilité d'un bien à 2 ans"):
        st.session_state.user_prompt_input = "Ce bien à 280 000 € sera-t-il rentable dans 2 ans ?"

    if st.button("🏆 3. Meilleures villes selon mes revenus"):
        st.session_state.user_prompt_input = "Quel est le meilleur endroit pour acheter une maison avec un foyer fiscal de 38 000 € de ressources ?"

    st.markdown("---")
    st.markdown("#### 📊 **Sources & Modèles actifs :**")
    st.markdown("✅ **Macro :** Emprunts, IRL, Taux crédit")
    st.markdown("✅ **Territoire :** Loyers & Parc (35k communes)")
    st.markdown("✅ **Fiscalité :** DGFiP Foyers fiscaux")
    st.markdown("✅ **Modèles :** Random Forest & Forecaster")
    
    if st.button("🗑️ Réinitialiser la conversation"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# -------------------------------------------------------------
# CORPS DE L'APPLICATION
# -------------------------------------------------------------
st.markdown('<div class="main-header">🏠 ImmoPredict — Assistant Prédictif & Décisionnel</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analysez vos opportunités d\'achat, de location et de rentabilité à 2 ans grâce à l\'IA et aux données territoriales consolidées.</div>', unsafe_allow_html=True)

# Affichage des messages de l'historique
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Affichage du graphique associé avec clé unique
        chart_type = msg.get("chart_type")
        chart_data = msg.get("chart_data")
        
        if chart_type and chart_data:
            unique_key = f"chart_{idx}_{chart_type}"
            if chart_type == "forecast_timeline":
                fig = create_forecast_timeline_chart(chart_data)
                st.plotly_chart(fig, use_container_width=True, key=unique_key)
            elif chart_type == "top_zones_bar":
                fig = create_top_zones_chart(chart_data)
                st.plotly_chart(fig, use_container_width=True, key=unique_key)
            elif chart_type == "macro_market":
                fig = create_macro_market_chart(chart_data)
                st.plotly_chart(fig, use_container_width=True, key=unique_key)

# Gestion de l'input utilisateur
prompt_from_button = st.session_state.pop("user_prompt_input", None)
user_prompt = st.chat_input("Ex: J'ai 250k€, quel est le meilleur endroit pour acheter un appartement rentable ?")

active_prompt = prompt_from_button or user_prompt

if active_prompt:
    # 1. Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": active_prompt})

    # 2. Appel de l'Agent IA & MCP
    with st.spinner("🧠 Analyse des données de marché et calcul des prédictions..."):
        response = agent.process_query(active_prompt)
        
        # Sauvegarde dans l'historique
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["text"],
            "chart_type": response.get("chart_type"),
            "chart_data": response.get("chart_data")
        })

    # 3. Rafraîchissement propre de l'affichage
    st.rerun()
