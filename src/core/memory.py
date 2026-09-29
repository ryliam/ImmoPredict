"""
memory.py - Gestion de l'historique et du contexte conversationnel du chatbot.
"""
from typing import List, Dict, Any, Optional


class ConversationMemory:
    """
    Gestionnaire simple d'historique de messages pour maintenir le contexte
    au cours des échanges avec l'utilisateur.
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, message: str) -> None:
        self.history.append({"role": "user", "content": message})
        self._trim()

    def add_assistant_message(self, message: str) -> None:
        self.history.append({"role": "assistant", "content": message})
        self._trim()

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def clear(self) -> None:
        self.history.clear()

    def _trim(self) -> None:
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]

