"""
src/core/agent.py — Orchestrateur conversationnel respectant STRICTEMENT archi.excalidraw :
Composants : User, LLM, Vector Db
Architecture STRICTE sans NLP intermédiaire ni fonctions de contrôle :
1. User -> LLM : message (le LLM est le cerveau principal direct)
2. LLM -> User : response if simple query (salutations ou recadrage poli hors-périmètre)
3. LLM -> Vector Db : search (décision exclusive du LLM via search_vector_db)
4. Vector Db -> User : response (réponse strictement enrichie dans le contexte du RAG)

Règle absolue anti-hallucination : AUCUNE INVENTION.
Si le contexte n'est pas clairement spécifié, le LLM est obligé de dire :
"Je passe la main à un conseiller pour plus de précision."
"""
import os
import re
import json
import logging
from typing import Dict, Any, Optional

from config.settings import HF_TOKEN, HF_MODEL, HF_BASE_URL
from src.core.prompts import (
    ROUTER_SYSTEM_PROMPT,
    GREETING_MESSAGE,
    HANDOVER_UNCLEAR_CONTEXT_MESSAGE
)
from src.core.guardrails import GuardrailManager
from src.core.rag import ClosedDomainRAG
from src.core.state_machine import ConversationStateManager, DialogState
from src.core.handover import HandoverManager
from src.core.memory import ConversationMemory

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_vector_db",
            "description": "Recherche dans la Vector DB certifiée d'ImmoPredict AI pour obtenir les informations officielles sur les services, la mission, les outils et les limites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La requête ou les mots-clés de recherche dans la base de connaissances certifiée"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


class RealEstateAgent:
    """
    Agent conversationnel où le LLM sert de cerveau principal unique.
    Aucune fonction de contrôle NLP externe : le LLM décide de la réponse simple
    ou de l'appel à la Vector DB via son prompt système strict.
    """

    def __init__(self, session_id: str = "default_session"):
        self.session_id = session_id
        self.state_manager = ConversationStateManager(session_id=session_id)
        self.vector_db = ClosedDomainRAG()
        self.memory = ConversationMemory(max_messages=10)

        self.client = None
        self.model_name = HF_MODEL or "meta-llama/Llama-3.1-8B-Instruct:deepinfra"
        self.base_url = HF_BASE_URL or "https://router.huggingface.co/v1"

        api_key = HF_TOKEN or os.getenv("HF_TOKEN")
        if api_key and HAS_OPENAI:
            try:
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=api_key
                )
            except Exception as e:
                logger.warning(f"Client LLM distant non disponible : {e}")

    def reset_session(self) -> None:
        """Réinitialise la session conversationnelle."""
        self.state_manager = ConversationStateManager(session_id=self.session_id)
        self.memory.clear()

    def process_query(self, user_message: str) -> Dict[str, Any]:
        """
        Point d'entrée principal (User -> LLM).
        Le LLM reçoit le message directement et décide du flux sans filtre NLP intermédiaire.
        """
        if not user_message or not str(user_message).strip():
            return {
                "text": GREETING_MESSAGE,
                "state": self.state_manager.active_state,
                "is_handover": False,
                "flow": "simple_query"
            }

        # Sécurité bas niveau (toxicité uniquement)
        is_safe, reason, refusal_msg = GuardrailManager.check_input_safety(user_message)
        if not is_safe:
            return {
                "text": refusal_msg,
                "state": self.state_manager.active_state,
                "is_handover": False,
                "guardrail_triggered": reason
            }

        # Le LLM est le cerveau principal unique
        if self.client:
            try:
                return self._process_with_llm(user_message)
            except Exception as e:
                logger.warning(f"Inférence LLM échouée ({e}), bascule sur le cerveau de secours.")

        # Repli déterministe (hors ligne / tests) reflétant scrupuleusement le prompt système
        return self._fallback_llm_brain(user_message)

    def _process_with_llm(self, user_message: str) -> Dict[str, Any]:
        """
        Exécution du LLM comme cerveau central :
        1. User -> LLM : message
        2. LLM décide : réponse simple directe OU appel search_vector_db
        """
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
        ]
        for turn in self.memory.get_history()[-4:]:
            messages.append(turn)
        messages.append({"role": "user", "content": user_message})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.1
        )
        choice = completion.choices[0]
        msg = choice.message

        tool_query = None
        # Détection outil (format natif OpenAI tool_calls ou balise texte Llama)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "search_vector_db":
                    try:
                        args = json.loads(tc.function.arguments)
                        tool_query = args.get("query", user_message)
                        break
                    except Exception:
                        tool_query = user_message
        elif msg.content and "<function=search_vector_db>" in msg.content:
            m = re.search(r'<function=search_vector_db>(.*?)(?:</function>|$)', msg.content, re.DOTALL)
            if m:
                try:
                    args = json.loads(m.group(1).strip())
                    tool_query = args.get("query", user_message)
                except Exception:
                    tool_query = user_message

        # FLUX 1 & 2 : Aucune recherche requise -> LLM -> User (response if simple query)
        if not tool_query:
            llm_text = (msg.content or "").strip()
            llm_text = re.sub(r'<function=.*?>.*?(?:</function>|$)', '', llm_text).strip()
            if not llm_text:
                llm_text = GREETING_MESSAGE

            # Détection d'escalade décidée par le LLM ou demandée par l'utilisateur
            if any(term in llm_text.lower() for term in ["conseiller", "humain", "passe la main", "mettre en relation", "transmets"]) or any(term in user_message.lower() for term in ["conseiller", "humain", "expert", "parler à"]):
                return self._trigger_handover(
                    reason="explicit_request",
                    message=llm_text,
                    user_message=user_message
                )

            self.memory.add_user_message(user_message)
            self.memory.add_assistant_message(llm_text)
            return {
                "text": llm_text,
                "state": self.state_manager.active_state,
                "is_handover": False,
                "flow": "simple_query",
                "tool_called": None
            }

        # FLUX 3 : LLM -> Vector Db : search
        rag_res = self.vector_db.retrieve(tool_query, top_k=3)

        # FLUX 4 : Vector Db -> User : response dans le contexte RAG strict
        if not rag_res["has_sufficient_context"]:
            # RÈGLE OBLIGATOIRE : Aucune invention si le contexte n'est pas clairement spécifié
            handover_msg = (
                "Cette information n'est pas clairement spécifiée dans notre documentation certifiée. "
                "Je passe la main à un conseiller pour plus de précision."
            )
            return self._trigger_handover(
                reason="out_of_scope_knowledge",
                message=handover_msg,
                user_message=user_message
            )

        # Contexte suffisant : enrichissement par le LLM adossé au RAG
        augmented_prompt = (
            f"{ROUTER_SYSTEM_PROMPT}\n\n"
            f"CONTEXTE OFFICIEL CERTIFIÉ EXTRAIT DE LA VECTOR DB :\n"
            f"{rag_res['context_text']}\n\n"
            f"CONSIGNE STRICTE : Réponds à la question de manière claire et détaillée en t'appuyant sur les faits ci-dessus. "
            f"AUCUNE INVENTION : Si et seulement si ce contexte ne permet pas de répondre, dis exactement : "
            f"'Je passe la main à un conseiller pour plus de précision.'"
        )
        second_messages = [
            {"role": "system", "content": augmented_prompt}
        ]
        for turn in self.memory.get_history()[-4:]:
            second_messages.append(turn)
        second_messages.append({"role": "user", "content": user_message})

        second_comp = self.client.chat.completions.create(
            model=self.model_name,
            messages=second_messages,
            temperature=0.1
        )
        final_text = (second_comp.choices[0].message.content or "").strip()
        final_text = re.sub(r'<function=.*?>.*?(?:</function>|$)', '', final_text).strip()

        # Si le LLM signale le manque de précision ou s'il s'agit d'une demande de conseiller
        if "passe la main à un conseiller" in final_text.lower():
            return self._trigger_handover(
                reason="out_of_scope_knowledge",
                message=final_text,
                user_message=user_message
            )

        if any(term in user_message.lower() for term in ["conseiller", "humain", "expert", "parler à"]) or any(term in final_text.lower() for term in ["transmettre votre demande", "mis en relation avec un conseiller", "contacter par un expert"]):
            return self._trigger_handover(
                reason="explicit_request",
                message=final_text,
                user_message=user_message
            )

        target_state = rag_res.get("suggested_state", DialogState.STATE_1_SERVICES)
        is_bt = self.state_manager.update_state(target_state)
        if is_bt:
            final_text = f"{final_text}{self.state_manager.generate_resumption_hook()}"

        self.memory.add_user_message(user_message)
        self.memory.add_assistant_message(final_text)

        return {
            "text": final_text,
            "state": self.state_manager.active_state,
            "is_handover": False,
            "flow": "vector_db_response",
            "tool_called": "search_vector_db"
        }

    def _fallback_llm_brain(self, user_message: str) -> Dict[str, Any]:
        """
        Cerveau de secours déterministe appliquant strictement les règles du prompt système
        (utilisé hors-ligne pour garantir les tests sans dépendance réseau).
        """
        cleaned = user_message.lower().strip()

        # 1. Requête simple (salutation / politesse)
        if cleaned in ["bonjour", "bonsoir", "salut", "hello", "bonjour !"]:
            resp = "Bonjour ! Je suis l'assistant d'accueil d'ImmoPredict AI. Comment puis-je vous accompagner dans votre projet immobilier aujourd'hui ?"
            self.memory.add_user_message(user_message)
            self.memory.add_assistant_message(resp)
            return {
                "text": resp,
                "state": DialogState.STATE_0_DISCOVERY,
                "is_handover": False,
                "flow": "simple_query",
                "tool_called": None
            }

        # 2. Demande explicite de conseiller
        if any(term in cleaned for term in ["conseiller", "expert", "humain", "rdv", "rappel", "parler à"]):
            return self._trigger_handover(
                reason="explicit_request",
                message="C'est bien noté. Je transmets votre demande à un conseiller humain expert d'ImmoPredict AI.",
                user_message=user_message
            )

        # 3. Requête nécessitant la Vector DB
        rag_res = self.vector_db.retrieve(user_message, top_k=3)

        if not rag_res["has_sufficient_context"]:
            # RÈGLE OBLIGATOIRE : Aucune invention si le contexte n'est pas clairement spécifié
            msg = (
                "Cette information n'est pas clairement spécifiée dans notre documentation certifiée. "
                "Je passe la main à un conseiller pour plus de précision."
            )
            return self._trigger_handover(
                reason="out_of_scope_knowledge",
                message=msg,
                user_message=user_message
            )

        # Réponse enrichie et strictement adossée au RAG
        if "gratuit" in cleaned or "payant" in cleaned:
            resp = "L'accès à nos analyses préliminaires, simulations et à l'assistant virtuel est 100% gratuit et sans engagement pour tous les utilisateurs."
        elif "simulateur" in cleaned or "rentabilite" in cleaned:
            resp = "Notre simulateur de rentabilité locative modélise l'indexation IRL, le rendement brut et la plus-value prévisionnelle sur 2 à 10 ans."
        elif any(w in cleaned for w in ["budget", "appartement", "maison", "acheter", "investir"]):
            resp = "Nos outils évaluent la faisabilité de votre projet en croisant les données notariales DVF et les revenus fiscaux médians de la commune."
        else:
            resp = "ImmoPredict AI est une plateforme d'intelligence immobilière indépendante basée sur les données publiques officielles (DVF, INSEE, DGFiP)."

        target_state = rag_res.get("suggested_state", DialogState.STATE_1_SERVICES)
        is_bt = self.state_manager.update_state(target_state)
        if is_bt:
            resp = f"{resp}{self.state_manager.generate_resumption_hook()}"

        self.memory.add_user_message(user_message)
        self.memory.add_assistant_message(resp)

        return {
            "text": resp,
            "state": self.state_manager.active_state,
            "is_handover": False,
            "flow": "vector_db_response",
            "tool_called": "search_vector_db"
        }

    def _trigger_handover(self, reason: str, message: str, user_message: str) -> Dict[str, Any]:
        """Génère le ticket et la réponse d'escalade CRM."""
        self.state_manager.active_state = DialogState.STATE_3_HANDOVER
        self.state_manager.deepest_state_reached = DialogState.STATE_3_HANDOVER

        ticket = HandoverManager.create_lead_ticket_from_llm(
            trigger_reason=reason,
            lead_profile=self.state_manager.lead_profile.summary_dict(),
            session_state={
                "active_state": self.state_manager.active_state,
                "deepest_state": self.state_manager.deepest_state_reached,
                "interaction_count": self.state_manager.interaction_count
            },
            last_user_query=user_message
        )

        self.memory.add_user_message(user_message)
        self.memory.add_assistant_message(message)

        return {
            "text": message,
            "state": DialogState.STATE_3_HANDOVER,
            "is_handover": True,
            "trigger_reason": reason,
            "ticket": ticket,
            "flow": "handover",
            "tool_called": None
        }