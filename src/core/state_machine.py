"""
state_machine.py - Gestionnaire d'états de conversation (Pilier 3).
Mémorise le palier le plus avancé atteint (High-Water Mark)
et génère le rebond contextuel de courtoisie (Backtracking & Resumption hook).
"""
import re
from typing import Dict, Any, Optional, Tuple


class DialogState:
    STATE_0_DISCOVERY = "STATE_0_DISCOVERY"
    STATE_1_SERVICES = "STATE_1_SERVICES"
    STATE_2_QUALIFICATION = "STATE_2_QUALIFICATION"
    STATE_3_HANDOVER = "STATE_3_HANDOVER"

    ORDER = {
        "STATE_0_DISCOVERY": 0,
        "STATE_1_SERVICES": 1,
        "STATE_2_QUALIFICATION": 2,
        "STATE_3_HANDOVER": 3
    }

    LABELS = {
        "STATE_0_DISCOVERY": "Accueil & Découverte",
        "STATE_1_SERVICES": "Présentation des Offres & Services",
        "STATE_2_QUALIFICATION": "Cadrage & Qualification du Projet",
        "STATE_3_HANDOVER": "Mise en Relation avec un Conseiller"
    }


class LeadProfile:
    """Profil et besoins du prospect."""
    def __init__(self):
        self.intent: Optional[str] = None
        self.budget: Optional[float] = None
        self.city: Optional[str] = None
        self.property_type: Optional[str] = None
        self.contact_info: Optional[str] = None

    def update_from_llm(self, llm_lead_info: Optional[Dict[str, Any]]) -> None:
        """Met à jour le profil avec les éléments détectés par le LLM."""
        if not llm_lead_info or not isinstance(llm_lead_info, dict):
            return

        if llm_lead_info.get("intent"):
            self.intent = str(llm_lead_info["intent"])
        if llm_lead_info.get("budget"):
            try:
                self.budget = float(llm_lead_info["budget"])
            except (ValueError, TypeError):
                pass
        if llm_lead_info.get("city"):
            self.city = str(llm_lead_info["city"]).capitalize()
        if llm_lead_info.get("property_type"):
            self.property_type = str(llm_lead_info["property_type"])
        if llm_lead_info.get("contact"):
            self.contact_info = str(llm_lead_info["contact"])

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent or "Non précisé",
            "budget": f"{self.budget:,.0f} €" if self.budget else "Non précisé",
            "city": self.city or "Non précisée",
            "property_type": self.property_type or "Non précisé",
            "contact": self.contact_info or "Non fourni"
        }


class ConversationStateManager:
    """
    Gestionnaire du contexte conversationnel.
    Suit la progression décidée au fil des échanges et applique le rebond Pilier 3.
    """

    def __init__(self, session_id: str = "default_session"):
        self.session_id = session_id
        self.active_state = DialogState.STATE_0_DISCOVERY
        self.deepest_state_reached = DialogState.STATE_0_DISCOVERY
        self.lead_profile = LeadProfile()
        self.last_advanced_topic = "votre projet immobilier"
        self.interaction_count = 0

    def update_state(self, new_state: str, llm_lead_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Met à jour l'état conversationnel et détecte si l'échange est un retour arrière.
        Retourne : is_backtracking (True si l'état courant est inférieur au palier le plus avancé atteint).
        """
        self.interaction_count += 1
        if llm_lead_info:
            self.lead_profile.update_from_llm(llm_lead_info)

        if new_state not in DialogState.ORDER:
            new_state = self.active_state

        new_order = DialogState.ORDER[new_state]
        deepest_order = DialogState.ORDER[self.deepest_state_reached]

        # Détection de retour arrière (l'utilisateur pose une question antérieure)
        is_backtracking = (new_order < deepest_order)

        if new_order > deepest_order:
            self.deepest_state_reached = new_state
            self._refresh_advanced_topic()

        self.active_state = new_state
        return is_backtracking

    def _refresh_advanced_topic(self) -> None:
        """Construit le résumé du point le plus avancé pour la relance de courtoisie."""
        p = self.lead_profile
        elements = []
        if p.intent:
            intent_map = {
                "investissement_locatif": "votre investissement locatif",
                "achat_residence": "votre projet d'achat",
                "location": "votre recherche de location"
            }
            elements.append(intent_map.get(p.intent, "votre projet"))
        if p.city:
            elements.append(f"sur {p.city}")
        if p.budget:
            elements.append(f"(budget de {p.budget:,.0f} €)")

        if elements:
            self.last_advanced_topic = " ".join(elements)
        elif self.deepest_state_reached == DialogState.STATE_2_QUALIFICATION:
            self.last_advanced_topic = "la qualification des critères de votre projet immobilier"
        elif self.deepest_state_reached == DialogState.STATE_1_SERVICES:
            self.last_advanced_topic = "nos outils de simulation et d'analyse immobilière"

    def generate_resumption_hook(self) -> str:
        """
        Génère la relance de courtoisie du Pilier 3 après une réponse rétrograde.
        """
        return (
            f"\n\n---\n"
            f"💡 *J'espère que cette précision vous éclaire ! "
            f"Pour en revenir à **{self.last_advanced_topic}** dont nous parlions juste avant, "
            f"souhaitez-vous que nous reprenions là où nous en étions ?*"
        )