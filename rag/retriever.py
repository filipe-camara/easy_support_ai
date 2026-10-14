from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config import load_settings

CHROMA_DIR = Path("/content/easy_support_ai/data/chroma")
COLLECTION_NAME = "easy_support_docs"

class KnowledgeRetriever:
    def __init__(self) -> None:
        settings = load_settings()
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Índice não encontrado em {CHROMA_DIR}. Execute rag/index_documents.py."
            )
        self.settings = settings
        self.store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=OpenAIEmbeddings(
                model=settings["OPENAI_EMBEDDING_MODEL"],
                openai_api_key=settings["OPENAI_API_KEY"]
            ),
        )

    def search(self, query: str) -> list[dict]:
        results = self.store.similarity_search_with_relevance_scores(
            query,
            k=self.settings["RAG_TOP_K"],
        )
        filtered = []
        for document, score in results:
            if score < self.settings["RAG_MIN_SCORE"]:
                continue
            filtered.append({
                "content": document.page_content,
                "score": float(score),
                "source": document.metadata.get(
                    "source_name",
                    document.metadata.get("source", "desconhecida"),
                ),
                "chunk_id": document.metadata.get("chunk_id"),
                "metadata": document.metadata,
            })
        return filtered
