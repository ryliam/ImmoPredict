"""
test_conversational_triage.py - Tests de conformité stricte avec archi.excalidraw.
Composants : User, LLM, Vector Db
Flux :
1. User -> LLM : message
2. LLM -> User : response if simple query
3. LLM -> Vector Db : search
4. Vector Db -> User : response
"""
import pytest
from src.core.guardrails import GuardrailManager
from src.core.rag import ClosedDomainRAG
from src.core.state_machine import ConversationStateManager, DialogState
from src.core.handover import HandoverManager
from src.core.agent import RealEstateAgent


def test_guardrails_toxicity():
    is_safe, reason, msg = GuardrailManager.check_input_safety("C'est quoi cette merde ?")
    assert not is_safe
    assert reason == "toxicity"
    assert "courtois" in msg.lower()


def test_rag_closed_domain_grounding():
    rag = ClosedDomainRAG()

    # Question documentée dans la Vector DB
    res_in = rag.retrieve("Vos services sont-ils payants ou gratuits ?")
    assert res_in["has_sufficient_context"] is True
    assert res_in["suggested_state"] == DialogState.STATE_0_DISCOVERY
    assert "gratuit" in res_in["context_text"].lower()

    # Question absente de la Vector DB
    res_out = rag.retrieve("Donnez-moi la recette de la tarte tatin aux pommes")
    assert res_out["has_sufficient_context"] is False


def test_flow_simple_query():
    """Vérifie le flux : LLM -> User (response if simple query) sans appel Vector DB."""
    agent = RealEstateAgent(session_id="test_simple_query")
    res = agent.process_query("Bonjour !")
    assert res["flow"] == "simple_query"
    assert not res["is_handover"]
    assert res["tool_called"] is None


def test_flow_vector_db_search_and_response():
    """Vérifie le flux : LLM -> Vector Db (search) -> User (response)."""
    agent = RealEstateAgent(session_id="test_rag_flow")
    res = agent.process_query("Comment fonctionne le simulateur de rentabilité locative ?")
    assert res["flow"] == "vector_db_response"
    assert not res["is_handover"]
    assert "rentabilité" in res["text"].lower()


def test_flow_handover_on_uncovered_query():
    """Vérifie l'escalade lorsque la Vector DB ne contient pas l'information."""
    agent = RealEstateAgent(session_id="test_uncovered")
    res = agent.process_query("Donnez-moi la recette de la tarte tatin aux pommes")
    assert res["is_handover"] is True
    assert res["trigger_reason"] == "out_of_scope_knowledge"
    assert "ticket" in res


def test_flow_explicit_handover():
    """Vérifie l'escalade sur demande directe de l'utilisateur."""
    agent = RealEstateAgent(session_id="test_handover")
    res = agent.process_query("Je souhaite impérativement parler à un conseiller humain")
    assert res["is_handover"] is True
    assert res["trigger_reason"] == "explicit_request"
    assert "ticket" in res