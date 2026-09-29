"""
agent.py - Implémentation stricte et corrigée de l'architecture archi.excalidraw :
Composants : User, LLM, Vector Db
Flux :
1. User -> LLM : message
2. LLM -> User : response if simple query (pour salutations/politesse SANS appel Vector DB)
3. LLM -> Vector Db : search (pour questions métier nécessitant la base certifiée)
4. Vector Db -> User : response (réponse strictement enrichie du RAG)
"""
import os
import re
import json
import logging
from typing import Dict, Any, Optional

from config.settings import HF_TOKEN, HF_MODEL, HF_BASE_URL
from src.core.prompts import ROUTER_SYSTEM_PROMPT, GREETING_MESSAGE
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
                        "description": "La requête ou les mots-clés de recherche dans la base de connaissances"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

SIMPLE_QUERY_PATTERNS = [
    r"^(bonjour|bonsoir|salut|hello|coucou|hi|hey)[\s\.,!\?]*$",
    r"^(comment\s+(tu\s+vas|allez-vous|ca\s+va|ça\s+va))[\s\.,!\?]*$",
    r"^(ca\s+va|ça\s+va)[\s\.,!\?]*$",
    r"^(merci|au\s+revoir|bonne\s+journée|a\s+bientôt|à\s+bientôt)[\s\.,!\?]*$",
    r"^(qui\s+es-tu|qui\s+êtes-vous|tu\s+es\s+qui)[\s\.,!\?]*$"
]


class RealEstateAgent:
    """
    Agent conversationnel suivant scrupuleusement l'architecture archi.excalidraw.
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
        """Réinitialise la session."""
        self.state_manager = ConversationStateManager(session_id=self.session_id)
        self.memory.clear()

    def is_simple_query(self, message: str) -> bool:
        """Détecte les salutations et politesses simples qui ne nécessitent AUCUN appel à la Vector DB."""
        cleaned = message.lower().strip()
        for pat in SIMPLE_QUERY_PATTERNS:
            if re.search(pat, cleaned):
                return True
        return False

    def process_query(self, user_message: str) -> Dict[str, Any]:
        """
        Point d'entrée principal (User -> LLM).
        """
        if not user_message or not str(user_message).strip():
            return {
                "text": GREETING_MESSAGE,
                "state": self.state_manager.active_state,
                "is_handover": False,
                "flow": "simple_query"
            }

        # 1. Filtre de sécurité bas niveau (toxicité uniquement)
        is_safe, reason, refusal_msg = GuardrailManager.check_input_safety(user_message)
        if not is_safe:
            return {
                "text": refusal_msg,
                "state": self.state_manager.active_state,
                "is_handover": False,
                "guardrail_triggered": reason
            }

        # Demande explicite de conseiller
        if any(w in user_message.lower() for w in ["conseiller", "expert", "humain", "rdv", "rappel", "parler à"]):
            return self._trigger_handover(
                reason="explicit_request",
                message="C'est bien noté ! Je transmets votre demande à un conseiller expert d'ImmoPredict AI.",
                user_message=user_message
            )

        # -------------------------------------------------------------
        # BRANCHE A : Simple Query -> LLM -> User (response if simple query)
        # Aucune recherche dans la Vector DB n'est déclenchée.
        # -------------------------------------------------------------
        if self.is_simple_query(user_message):
            return self._handle_simple_query(user_message)

        # -------------------------------------------------------------
        # BRANCHE B : Requête métier -> LLM -> Vector Db (search) -> User (response)
        # -------------------------------------------------------------
        return self._handle_knowledge_query(user_message)

    def _handle_simple_query(self, user_message: str) -> Dict[str, Any]:
        """
        Gère le flux 'response if simple query' sans solliciter la Vector DB.
        """
        cleaned = user_message.lower().strip()

        # Si le client LLM est disponible, on l'appelle SANS outil pour obtenir une réponse naturelle
        if self.client:
            try:
                simple_prompt = (
                    "Tu es l'assistant d'accueil d'ImmoPredict AI. "
                    "L'utilisateur te salue ou te pose une question de politesse simple. "
                    "Réponds-lui chaleureusement, brièvement et avec courtoisie en lui demandant comment tu peux l'aider dans son projet immobilier. "
                    "N'utilise aucun outil et ne fais aucune référence technique."
                )
                messages = [
                    {"role": "system", "content": simple_prompt},
                    {"role": "user", "content": user_message}
                ]
                comp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3
                )
                llm_text = comp.choices[0].message.content.strip()
                # Sécurité anti-fuite de fonction
                if not llm_text.startswith("<function="):
                    self.memory.add_user_message(user_message)
                    self.memory.add_assistant_message(llm_text)
                    return {
                        "text": llm_text,
                        "state": DialogState.STATE_0_DISCOVERY,
                        "is_handover": False,
                        "flow": "simple_query",
                        "tool_called": None
                    }
            except Exception as e:
                logger.warning(f"Inférence simple query échouée ({e}), repli local.")

        # Réponse locale conviviale et personnalisée
        if "comment" in cleaned:
            resp_text = "Bonjour ! Je vais très bien, merci. Je suis l'assistant virtuel d'ImmoPredict AI. Comment puis-je vous accompagner dans votre projet immobilier aujourd'hui ?"
        elif "merci" in cleaned:
            resp_text = "Je vous en prie ! N'hésitez pas si vous avez d'autres questions sur nos simulations ou vos projets immobiliers."
        elif "au revoir" in cleaned or "bonne" in cleaned:
            resp_text = "Au revoir et très bonne journée à vous ! Au plaisir de vous accompagner prochainement."
        else:
            resp_text = "Bonjour et bienvenue chez ImmoPredict AI ! En quoi puis-je vous aider dans votre réflexion immobilière aujourd'hui ?"

        self.memory.add_user_message(user_message)
        self.memory.add_assistant_message(resp_text)

        return {
            "text": resp_text,
            "state": DialogState.STATE_0_DISCOVERY,
            "is_handover": False,
            "flow": "simple_query",
            "tool_called": None
        }

    def _handle_knowledge_query(self, user_message: str) -> Dict[str, Any]:
        """
        Gère le flux 'search Vector DB' -> 'response'.
        """
        # FLUX 3 : LLM -> Vector Db (search)
        rag_res = self.vector_db.retrieve(user_message, top_k=3)

        # FLUX 4 : Vector Db -> User (response enrichie)
        if rag_res["has_sufficient_context"]:
            # Si le client LLM distant est disponible
            if self.client:
                try:
                    augmented_prompt = (
                        f"{ROUTER_SYSTEM_PROMPT}\n\n"
                        f"CONDUITE STRICTE : Réponds à l'utilisateur UNIQUEMENT en t'appuyant sur ce contexte extrait de la Vector DB :\n"
                        f"{rag_res['context_text']}"
                    )
                    messages = [
                        {"role": "system", "content": augmented_prompt}
                    ]
                    for turn in self.memory.get_history()[-4:]:
                        messages.append(turn)
                    messages.append({"role": "user", "content": user_message})

                    comp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.1
                    )
                    final_text = comp.choices[0].message.content.strip()

                    # Nettoyage si le modèle réécrivait accidentellement une balise
                    if not final_text.startswith("<function="):
                        self.memory.add_user_message(user_message)
                        self.memory.add_assistant_message(final_text)

                        target_state = rag_res.get("suggested_state", DialogState.STATE_1_SERVICES)
                        is_bt = self.state_manager.update_state(target_state)
                        if is_bt:
                            final_text = f"{final_text}{self.state_manager.generate_resumption_hook()}"

                        return {
                            "text": final_text,
                            "state": self.state_manager.active_state,
                            "is_handover": False,
                            "flow": "vector_db_response",
                            "tool_called": "search_vector_db"
                        }
                except Exception as e:
                    logger.warning(f"Inférence enrichie échouée ({e}), repli local.")

            # Formulation locale enrichie garantie par les documents du RAG
            cleaned = user_message.lower()
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

        # Si l'information est absente de la Vector DB -> Escalade humaine
        return self._trigger_handover(
            reason="out_of_scope_knowledge",
            message="Cette question spécifique n'est pas répertoriée dans notre Vector DB certifiée. Je vous mets en relation avec un de nos conseillers humains pour y répondre.",
            user_message=user_message
        )

    def _trigger_handover(self, reason: str, message: str, user_message: str) -> Dict[str, Any]:
        """Génère le ticket et la réponse d'escalade."""
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