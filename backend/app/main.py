# API principal do FastAPI

from fastapi import FastAPI, HTTPException                 # FastAPI cria a API; HTTPException retorna erros padronizados (400, 500)
from fastapi.middleware.cors import CORSMiddleware           # Middleware de segurança que autoriza o frontend a acessar esta API
from pydantic import BaseModel                               # Valida o formato dos dados JSON enviados pelo usuário
from app.services.rag_service import search_philosophical_context
from app.services.llm_service import generate_philosophical_response

app = FastAPI(title="IA Filosófica API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def health_check():
    return {"status": "online", "message": "API da IA Filosófica operacional"}

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        # 1. Busca os trechos nos livros (continua funcionando normalmente nos bastidores)
        contexts = search_philosophical_context(request.message, limit=4)
        
        # 2. Gera a resposta empática com a citação traduzida
        reply = generate_philosophical_response(
            user_message=request.message,
            contexts=contexts,
            history=request.history
        )
        
        # 3. Retorna apenas a resposta limpa, sem o bloco extenso de sources
        return ChatResponse(reply=reply)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar reflexão: {str(e)}")