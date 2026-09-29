"""
handover.py - Gestionnaire d'enregistrement et de packaging du transfert humain.
Ne prend AUCUNE décision heuristique : il enregistre et structure la décision prise par le LLM.
"""
from datetime import datetime
from typing import Dict, Any, Optional


class HandoverManager:
    """Enregistre le ticket CRM issu de la décision souveraine du LLM."""

    @staticmethod
    def create_lead_ticket_from_llm(
        trigger_reason: str,
        lead_profile: Dict[str, Any],
        session_state: Dict[str, Any],
        last_user_query: str
    ) -> Dict[str, Any]:
        """Formate le ticket officiel pour le CRM avec le trigger_reason décidé par le LLM."""
        urgency = "HAUTE" if lead_profile.get("contact") else "MOYENNE"
        if trigger_reason == "explicit_request":
            urgency = "HAUTE"

        ticket = {
            "ticket_id": f"LEAD-{int(datetime.utcnow().timestamp())}",
            "created_at": datetime.utcnow().isoformat(),
            "decided_by": "LLM_DECISION_ENGINE",
            "trigger_reason": trigger_reason,
            "session_state": session_state,
            "lead_qualification": lead_profile,
            "latest_question": last_user_query,
            "urgency": urgency
        }
        return ticket