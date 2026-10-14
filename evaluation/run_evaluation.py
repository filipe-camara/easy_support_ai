import sys
sys.path.append("/content/easy_support_ai")
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from classifier.service import UrgencyClassifier
from workflow.graph import support_graph

CLASSIFIER_DATA = Path("/content/easy_support_ai/data/tickets_treino.csv")
RAG_CASES = Path("/content/easy_support_ai/evaluation/rag_cases.json")

def run_classifier_evaluation():
    print("=== AVALIAÇÃO QUANTITATIVA DO CLASSIFICADOR ===")
    if not CLASSIFIER_DATA.exists():
        print("Dataset de treino não encontrado.")
        return
    
    df = pd.read_csv(CLASSIFIER_DATA)
    classifier = UrgencyClassifier()
    
    y_true = []
    y_pred = []
    
    for _, row in df.iterrows():
        text = row["texto"]
        true_urgency = row["urgencia"]
        try:
            pred = classifier.predict(text)["urgencia"]
            y_true.append(true_urgency)
            y_pred.append(pred)
        except Exception as e:
            print(f"Erro ao processar texto: {text[:30]} -> {e}")
            
    acc = accuracy_score(y_true, y_pred)
    print(f"Acurácia geral: {acc:.2%}\n")
    print(classification_report(y_true, y_pred, zero_division=0))

def run_rag_evaluation():
    print("\n=== AVALIAÇÃO DE COBERTURA E ABSTENÇÃO DO RAG ===")
    if not RAG_CASES.exists():
        # Criando o arquivo padrão de cenários caso não exista
        default_cases = [
            {
                "question": "Como redefinir a minha senha de acesso?",
                "expected_source": "politica_senhas.md",
                "must_abstain": False
            },
            {
                "question": "Como exportar um relatório do sistema corporativo?",
                "expected_source": "exportacao_relatorios.md",
                "must_abstain": False
            },
            {
                "question": "Qual é a previsão do tempo para amanhã em São Paulo?",
                "expected_source": None, 
                "must_abstain": True
            }
        ]
        RAG_CASES.parent.mkdir(parents=True, exist_ok=True)
        with open(RAG_CASES, 'w', encoding='utf-8') as f:
            json.dump(default_cases, f, indent=2, ensure_ascii=False)
            
    with open(RAG_CASES, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        
    for idx, case in enumerate(cases):
        q = case["question"]
        must_abstain = case.get("must_abstain", False)
        print(f"Caso #{idx+1}: {q}")
        
        result = support_graph.invoke({"question": q})
        answer = result.get("answer", {})
        
        print(f"  - Urgência predita: {answer.get('urgency')}")
        print(f"  - Necessita revisão humana: {answer.get('needs_human_review')}")
        
        sources = [s["source"] for s in answer.get("sources", [])]
        print(f"  - Fontes citadas: {sources}")
        
        if must_abstain:
            # RAG deve se abster se não houver fonte pertinente na base
            if len(sources) == 0:
                print("  -> [OK] RAG absteve-se corretamente de responder sem fonte base.")
            else:
                print("  -> [AVISO] O RAG deveria ter se abstido para esta pergunta fora de contexto.")
        else:
            expected = case.get("expected_source")
            if any(expected in src for src in sources if expected):
                print(f"  -> [OK] Encontrou a fonte esperada: {expected}")
            else:
                print(f"  -> [AVISO] Fonte esperada '{expected}' não foi localizada nas referências.")

if __name__ == "__main__":
    run_classifier_evaluation()
    run_rag_evaluation()
