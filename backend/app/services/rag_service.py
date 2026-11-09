from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "filosofia_obras"

embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
qdrant_client = QdrantClient(path=QDRANT_PATH)

def search_philosophical_context(query: str, limit: int = 4) -> list[dict]:
    """Busca os trechos mais relevantes usando a nova API query_points do Qdrant."""
    query_vector = embed_model.encode(query).tolist()
    
    # Na versão atual do qdrant-client, usa-se query_points em vez de search
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit
    ).points
    
    return [
        {
            "obra": hit.payload.get("obra", "Obra desconhecida"),
            "texto": hit.payload.get("texto", "")
        }
        for hit in search_result
    ]