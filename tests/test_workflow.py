import sys
sys.path.append("/content/easy_support_ai")
import pytest
from workflow.graph import support_graph

def test_workflow_low_urgency_with_knowledge():
    """Valida o fluxo para uma dúvida comum coberta pela base de conhecimento."""
    input_state = {
        "question": "Como redefinir a minha senha de acesso de acordo com as regras?"
    }
    result = support_graph.invoke(input_state)
    
    assert "answer" in result
    answer = result["answer"]
    assert answer["urgency"] == "baixa"
    assert answer["needs_human_review"] is False
    assert len(answer["sources"]) > 0
    assert any("senha" in s["source"].lower() for s in answer["sources"])

def test_workflow_high_urgency_creates_ticket():
    """Valida se incidentes críticos resultam em proposta de chamado e requerem aprovação."""
    input_state = {
        "question": "URGENTE: O servidor de banco de dados de produção está completamente indisponível!"
    }
    result = support_graph.invoke(input_state)
    
    assert "answer" in result
    answer = result["answer"]
    assert answer["urgency"] == "alta"
    assert result["requires_approval"] is True
    assert result["proposed_action"] == "create_ticket"

def test_workflow_empty_query_fails():
    """Garante o tratamento gracioso de entradas inválidas."""
    input_state = {
        "question": "   "
    }
    result = support_graph.invoke(input_state)
    
    assert "error" in result
    assert result["error"] is not None
    assert "vazia" in result["error"].lower()
