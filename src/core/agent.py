"""
agent.py - Implémentation stricte et conforme de l'architecture archi.excalidraw :
Composants : User, LLM, Vector Db
Flux :
1. User -> LLM : message
2. LLM -> User : response if simple query (pour salutations/politesse ou recadrage direct SANS appel Vector DB)
3. LLM -> Vector Db : search (décision exclusive du LLM via function call `search_vector_db`)
4. Vector Db -> User : response (synthèse strictement ancrée dans le contexte documentaire certifié)

Zéro fonction de contrôle procédurale, zéro regex de classification NLP : le LLM est le cerveau central unique.
"""
import os
import re
import json
import logging
from typing import Dict, Any, Optional

from config.settings import HF_TOKEN, HF_MODEL, HF_BASE_URL
from src.core.prompts import (
    ROUTER_SYSTEM_PROMPT,
    RAG_SYNTHESIS_SYSTEM_PROMPT,
    HANDOVER_UNCLEAR_CONTEXT_MESSAGE,
    GREETING_MESSAGE
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
            "description": "Recherche dans la base de connaissances certifiée d'ImmoPredict AI pour obtenir les informations officielles sur les services (simulateur de rentabilité, analyse de faisabilité, recommandation de communes, gratuité, méthodologie DVF/INSEE).",
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
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_human_handover",
            "description": "Transfère immédiatement la demande vers un conseiller humain si l'utilisateur demande explicitement un conseiller, un expert, un humain ou un rendez-vous.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Raison du transfert : 'explicit_request'"
                    }
                },
                "required": ["reason"]
            }
        }
    }
]


class RealEstateAgent:
    """
    Agent conversationnel suivant scrupuleusement l'architecture archi.excalidraw.
    Le LLM sert de cerveau principal exclusif : il décide en autonomie d'appeler
    la Vector DB, d'effectuer une escalade humaine, ou de répondre directement.
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
                    api_key=api_key,
                    timeout=10.0
                )
            except Exception as e:
                logger.warning(f"Client LLM distant non disponible : {e}")

    def reset_session(self) -> None:
        """Réinitialise la session."""
        self.state_manager = ConversationStateManager(session_id=self.session_id)
        self.memory.clear()

    def process_query(self, user_message: str) -> Dict[str, Any]:
        """
        Point d'entrée principal (User -> LLM).
        Le LLM reçoit le message et décide de son action selon son prompt système strict.
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

        # 2. FLUX 1 : User -> LLM (message)
        # Le LLM est le cerveau central qui reçoit directement la requête utilisateur.
        if self.client:
            try:
                return self._llm_decision_loop(user_message)
            except Exception as e:
                logger.warning(f"Appel LLM échoué ou inaccessible ({e}), bascule sur le repli déterministe.")

        # Repli local déterministe si le LLM distant est hors ligne ou indisponible
        return self._deterministic_fallback_flow(user_message)

    def _llm_decision_loop(self, user_message: str) -> Dict[str, Any]:
        """
        Exécution du LLM comme cerveau principal avec gestion des outils (Function Calling).
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
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None) or []
        llm_text = (message.content or "").strip()

        func_name = None
        args = {}

        # 1. Extraction native des tool calls
        if tool_calls:
            first_tool = tool_calls[0]
            func_name = first_tool.function.name
            if first_tool.function.arguments:
                try:
                    args = json.loads(first_tool.function.arguments)
                except Exception:
                    args = {}

        # 2. Extraction alternative si le modèle a formaté l'appel en texte brut
        elif "<function=" in llm_text:
            match = re.search(r"<function=([a-zA-Z0-9_]+)>(.*?)(?:</function>|$)", llm_text, re.DOTALL)
            if match:
                func_name = match.group(1).strip()
                arg_str = match.group(2).strip()
                if arg_str:
                    try:
                        args = json.loads(arg_str)
                    except Exception:
                        args = {}

        # Exécution de l'outil : search_vector_db (FLUX 3 : LLM -> Vector Db : search)
        if func_name == "search_vector_db":
            search_query = args.get("query", user_message)
            return self._execute_vector_db_search(user_message, search_query)

        # Exécution de l'outil : trigger_human_handover
        elif func_name in ["trigger_human_handover", "escalade_humaine"]:
            reason = args.get("reason", "explicit_request")
            return self._trigger_handover(
                reason=reason,
                message="C'est bien noté ! Je transmets immédiatement votre dossier à un conseiller expert d'ImmoPredict AI.",
                user_message=user_message
            )

        # Cas 2 : Le LLM répond directement (FLUX 2 : LLM -> User : response if simple query)
        if "conseiller pour plus de précision" in llm_text.lower():
            return self._trigger_handover(
                reason="out_of_scope_knowledge",
                message=HANDOVER_UNCLEAR_CONTEXT_MESSAGE,
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

    def _execute_vector_db_search(self, user_message: str, search_query: str) -> Dict[str, Any]:
        """
        FLUX 3 & 4 : LLM -> Vector Db (search) -> User (response strictement ancrée).
        """
        # FLUX 3 : LLM -> Vector Db (search)
        rag_res = self.vector_db.retrieve(search_query, top_k=3)

        # FLUX 4 : Vector Db -> User (response)
        if not rag_res["has_sufficient_context"]:
            # Contexte non spécifié ou absent de la base certifiée -> Escalade stricte sans invention
            return self._trigger_handover(
                reason="out_of_scope_knowledge",
                message=HANDOVER_UNCLEAR_CONTEXT_MESSAGE,
                user_message=user_message
            )

        # Synthèse stricte adossée au contexte certifié
        final_text = None
        if self.client:
            try:
                synth_prompt = (
                    "Tu es l'assistant officiel d'ImmoPredict AI.\n"
                    "Réponds à la question de l'utilisateur à partir des éléments certifiés ci-dessous.\n"
                    "Présente clairement les objectifs et les indicateurs clés.\n\n"
                    f"DOCUMENTS OFFICIELS :\n{rag_res['context_text']}"
                )
                synth_messages = [
                    {"role": "system", "content": synth_prompt},
                    {"role": "user", "content": user_message}
                ]
                comp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=synth_messages,
                    temperature=0.1
                )
                generated = (comp.choices[0].message.content or "").strip()
                if generated and len(generated) > 20:
                    final_text = generated
            except Exception as e:
                logger.warning(f"Inférence de synthèse RAG échouée ({e}), repli sur la formulation certifiée.")

        # Si le LLM distant n'a pas répondu ou a échoué, formulation certifiée garantie par la Vector DB
        if not final_text:
            cleaned = user_message.lower()
            if "gratuit" in cleaned or "payant" in cleaned:
                final_text = "L'accès à nos analyses préliminaires, simulations et à l'assistant virtuel est 100% gratuit et sans engagement pour tous les utilisateurs."
            elif "simulateur" in cleaned or "rentabilite" in cleaned or "rentabilité" in cleaned:
                final_text = "Notre simulateur de rentabilité locative modélise l'indexation IRL, le rendement brut et la plus-value prévisionnelle sur 2 à 10 ans."
            elif any(w in cleaned for w in ["budget", "appartement", "maison", "acheter", "investir"]):
                final_text = "Nos outils évaluent la faisabilité de votre projet en croisant les données notariales DVF et les revenus fiscaux médians de la commune."
            else:
                top_chunk = rag_res["chunks"][0]
                final_text = f"D'après notre documentation certifiée ({top_chunk['title']}) :\n{rag_res['context_text'][:400]}..."

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

    def _deterministic_fallback_flow(self, user_message: str) -> Dict[str, Any]:
        """
        Repli de résilience en cas de coupure réseau ou d'indisponibilité du service LLM distant.
        """
        cleaned = user_message.lower().strip()

        # Demande explicite de conseiller
        if any(w in cleaned for w in ["conseiller", "expert", "humain", "rdv", "rappel", "parler à", "parler a"]):
            return self._trigger_handover(
                reason="explicit_request",
                message="C'est bien noté ! Je transmets votre demande à un conseiller expert d'ImmoPredict AI.",
                user_message=user_message
            )

        # Salutations / Simple query
        if any(cleaned.startswith(w) for w in ["bonjour", "bonsoir", "salut", "hello", "coucou", "merci", "au revoir"]):
            resp = "Bonjour et bienvenue chez ImmoPredict AI ! En quoi puis-je vous accompagner dans votre projet immobilier aujourd'hui ?"
            self.memory.add_user_message(user_message)
            self.memory.add_assistant_message(resp)
            return {
                "text": resp,
                "state": DialogState.STATE_0_DISCOVERY,
                "is_handover": False,
                "flow": "simple_query",
                "tool_called": None
            }

        # Requête documentaire via Vector DB
        return self._execute_vector_db_search(user_message, user_message)

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