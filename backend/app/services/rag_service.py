import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "filosofia_obras"

# Carrega o mesmo modelo multilíngue usado na ingestão
embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def get_qdrant_client():
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Credenciais do Qdrant Cloud não configuradas nas variáveis de ambiente.")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def search_philosophical_context(query: str, limit: int = 4) -> list[dict]:
    """Busca os trechos mais relevantes nas obras filosóficas."""
    client = get_qdrant_client()
    query_vector = embed_model.encode(query).tolist()
    
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )
    
    contexts = []
    for hit in search_result:
        contexts.append({
            "obra": hit.payload.get("obra", "Obra desconhecida"),
            "texto": hit.payload.get("texto", ""),
            "score": round(hit.score, 3)
        })
    return contexts