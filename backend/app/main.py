# API principal do FastAPI

from fastapi import FastAPI, HTTPException                 # FastAPI cria a API; HTTPException retorna erros padronizados (400, 500)
from fastapi.middleware.cors import CORSMiddleware           # Middleware de segurança que autoriza o frontend a acessar esta API
from pydantic import BaseModel                               # Valida o formato dos dados JSON enviados pelo usuário
from typing import List, Optional                            # Tipagem para listas e campos opcionais
from dotenv import load_dotenv                               # Carrega variáveis do arquivo .env

# Carrega as chaves de API do arquivo .env (se existir) antes de importar os serviços
load_dotenv()

# Importa as funções que criamos nos outros dois arquivos
from app.services.rag_service import search_philosophical_context
from app.services.llm_service import generate_philosophical_response

# Inicializa a aplicação FastAPI
app = FastAPI(title="IA Filosófica API", version="1.0.0")

# Configuração de CORS (Cross-Origin Resource Sharing):
# Permite que páginas web hospedadas em outros endereços (como Vercel) façam requisições para esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                                     # "*" libera para qualquer origem (ideal durante o desenvolvimento)
    allow_credentials=True,
    allow_methods=["*"],                                     # Libera todos os métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],                                     # Libera todos os cabeçalhos HTTP
)

# Define o formato esperado de cada mensagem antiga no histórico
class MessageHistory(BaseModel):
    role: str                                                # Quem falou: "user" (usuário) ou "assistant" (IA)
    content: str                                             # O texto da mensagem

# Define o formato exato do JSON que o Frontend deve enviar para a rota /api/chat
class ChatRequest(BaseModel):
    message: str                                             # A pergunta atual do usuário (obrigatória)
    history: Optional[List[MessageHistory]] = []             # Lista de mensagens anteriores (opcional)

# Define o formato exato do JSON que a API vai devolver para o Frontend
class ChatResponse(BaseModel):
    reply: str                                               # A resposta final escrita pela IA
    sources: List[dict]                                      # Os trechos originais usados como fonte (para transparência)

# Rota de verificação de saúde (útil para testar no navegador se o servidor está online)
@app.get("/")
def health_check():
    return {"status": "online", "message": "API da IA Filosófica operando normalmente."}

# Rota principal do chat (recebe requisições POST)
@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    # Valida se o usuário não enviou apenas espaços em branco
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="A mensagem não pode estar vazia.")
    
    try:
        # Passo 1: Vai ao Qdrant e busca os 4 trechos filosóficos mais próximos da pergunta
        contexts = search_philosophical_context(request.message, limit=4)
        
        # Passo 2: Converte os objetos Pydantic do histórico para dicionários comuns do Python
        history_dicts = [{"role": m.role, "content": m.content} for m in request.history]
        
        # Passo 3: Envia pergunta + trechos + histórico para o Gemini/Groq gerar a resposta
        ai_reply = generate_philosophical_response(
            user_message=request.message,
            contexts=contexts,
            history=history_dicts
        )
        
        # Passo 4: Devolve a resposta pronta e as fontes consultadas para o site exibir
        return ChatResponse(
            reply=ai_reply,
            sources=contexts
        )
        
    except Exception as e:
        # Se ocorrer qualquer falha interna, devolve erro 500 com a descrição do problema
        raise HTTPException(status_code=500, detail=f"Erro ao processar reflexão: {str(e)}")