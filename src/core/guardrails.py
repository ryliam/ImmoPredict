"""
guardrails.py - Filtres de sécurité et de modération de premier niveau.
Se concentre sur la toxicité, sans interférer dans la logique décisionnelle du LLM.
"""
import re
from typing import Tuple, Optional

# Termes / expressions de haine, toxicité et insultes
TOXIC_PATTERNS = [
    r"\b(connard|salope|merde|pute|fdp|encul[eé]|bâtard|nique|ferme\s+ta\s+gueule)\b",
    r"\b(idiot|abruti|imbécile|degage|tais-toi)\b",
    r"\b(nazi|terrorist|raciste|nègre)\b",
]

# Tentatives flagrantes d'injection de prompt ou jailbreak destructeur
JAILBREAK_PATTERNS = [
    r"ignore\s+(toutes\s+les\s+instructions|tes\s+règles|ton\s+système)",
    r"forget\s+(all\s+previous\s+instructions|your\s+rules)",
    r"dan\s+mode|jailbreak|dev\s+mode",
]


class GuardrailManager:
    """Filtres de sécurité bas niveau."""

    @staticmethod
    def check_input_safety(user_message: str) -> Tuple[bool, str, Optional[str]]:
        """
        Intercepte uniquement les insultes et attaques pour protéger le modèle.
        La compréhension métier et les décisions de dialogue sont déléguées au LLM.
        """
        cleaned = user_message.lower().strip()

        for pattern in TOXIC_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return (
                    False,
                    "toxicity",
                    "Je suis un assistant professionnel dédié à l'accompagnement immobilier. "
                    "Je ne peux pas répondre aux messages contenant des propos déplacés ou injurieux. "
                    "Restons courtois dans nos échanges."
                )

        for pattern in JAILBREAK_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return (
                    False,
                    "jailbreak_attempt",
                    "Je suis configuré exclusivement pour vous renseigner et vous orienter sur vos projets immobiliers "
                    "avec ImmoPredict AI. Je ne peux déroger à cette mission."
                )

        return True, "safe", None