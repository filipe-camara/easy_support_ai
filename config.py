import os
from google.colab import userdata

def load_settings():
    # Recupera de forma segura a chave de API real do Colab Secrets
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        try:
            openai_key = userdata.get("OPENAI_API_KEY")
        except Exception:
            pass
            
    if not openai_key:
        # Caso falhe por algum motivo de escopo, mantém a string para não quebrar a compilação do container de teste
        openai_key = "mock-key-para-testes-se-necessario-substitua-pela-sua"

    return {
        "OPENAI_API_KEY": openai_key,
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        "RAG_TOP_K": int(os.getenv("RAG_TOP_K", 5)),
        "RAG_MIN_SCORE": float(os.getenv("RAG_MIN_SCORE", 0.45)),
        "MAX_CONTEXT_TOKENS": int(os.getenv("MAX_CONTEXT_TOKENS", 3500)),
        "CLASSIFIER_CONFIDENCE_MIN": float(os.getenv("CLASSIFIER_CONFIDENCE_MIN", 0.65))
    }
