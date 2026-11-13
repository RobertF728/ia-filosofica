# 🏛️ IA Filosófica — Sabedoria & Reflexão

Assistente conversacional baseado em **RAG (Retrieval-Augmented Generation)** que oferece acolhimento emocional e reflexões filosóficas fundamentadas exclusivamente em obras clássicas da humanidade, operando de forma 100% gratuita.

---

## 🧠 Como a Arquitetura Funciona

O sistema evita "alucinações" (citações inventadas) dividindo o trabalho em duas etapas:

1. **Busca Semântica (RAG Local com Qdrant):** Quando o usuário envia uma angústia ou dúvida em português, o sistema converte a pergunta em um vetor numérico (`Sentence-Transformers`) e busca no banco de dados local (`backend/qdrant_db/`) os **4 trechos mais relevantes** dentro dos livros processados.

2. **Síntese Empática (LLM com Fallback Automático):** Os trechos originais encontrados são enviados junto com a pergunta do usuário para o **Google Gemini (`gemini-3.5-flash-lite`)**. Caso a API do Google apresente instabilidade ou limite de requisições, o sistema aciona automaticamente a **Groq Cloud** como provedor de reserva, traduzindo a citação escolhida e aplicando-a ao contexto do usuário.

### 📚 Acervo Filosófico Processado
* *Meditations* — Marco Aurélio
* *The Enchiridion* — Epicteto
* *The Republic* — Platão
* *Beyond Good and Evil* — Friedrich Nietzsche
* *The Prince* — Nicolau Maquiavel

---

## 📂 Estrutura do Projeto

```text
ia-filosofica/
├── backend/
│   ├── app/
│   │   ├── services/
│   │   │   ├── llm_service.py   # Comunicação com Gemini (principal) e Groq (fallback)
│   │   │   └── rag_service.py   # Busca vetorial no banco local Qdrant (query_points)
│   │   └── main.py              # API FastAPI (rotas / e /api/chat + configuração CORS)
│   ├── data/
│   │   └── books/               # Arquivos .txt das obras filosóficas originais
│   ├── qdrant_db/               # Banco de dados vetorial persistido em disco
│   ├── .env                     # Chaves secretas de API (não versionado no Git)
│   ├── ingest.py                # Script que fatia os livros e gera o banco qdrant_db
│   └── requirements.txt         # Dependências Python do backend
└── README.md                    # Documentação do projeto
```

## ⚙️ Passo a Passo: Como Rodar o Backend


### 1. Pré-requisitos e Chaves Gratuitas
Para executar a API sem custos, gere duas chaves gratuitas (sem necessidade de cartão de crédito):

```
Google Gemini API Key: Gerada no Google AI Studio.

Groq API Key (Reserva): Gerada no Groq Console.
```

### 2. Instalar as Dependências
Abra o terminal, entre na pasta backend e instale as bibliotecas necessárias:

```
cd backend
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configurar as Variáveis de Ambiente (.env)
Dentro da pasta backend/, crie um arquivo chamado .env contendo suas chaves (sem aspas e sem espaços ao redor do =):

```
GEMINI_API_KEY=sua_chave_do_google_gemini_aqui
GROQ_API_KEY=sua_chave_da_groq_aqui
```

(Certifique-se de que .env esteja listado no .gitignore para nunca expor suas chaves no GitHub).

### 4. Gerar o Banco Vetorial (qdrant_db)
(Nota: Se a pasta backend/qdrant_db/ já estiver no repositório, você só precisa rodar este passo novamente caso adicione novos livros .txt em backend/data/books/).

```
python ingest.py
```

Este script divide os livros em fragmentos de 700 caracteres (com sobreposição de 100 caracteres para manter o contexto) e salva os embeddings na pasta backend/qdrant_db/.

### 5. Iniciar o Servidor da API
Ainda dentro da pasta backend/, inicie o servidor FastAPI com Uvicorn:

```
uvicorn app.main:app --reload --port 8000
```

### 6. Testar a API no Navegador
Com o servidor rodando:

Acesse http://127.0.0.1:8000/docs (ou o link encaminhado da porta 8000 no GitHub Codespaces seguido de /docs).

Abra a rota POST /api/chat > clique em Try it out.

Envie um JSON de teste:

```
{
  "message": "Por que me sinto tão ansioso com o que não posso controlar?",
  "history": []
}
```

## 🛡️ Boas Práticas e Manutenção do Repositório (Checklist)

Para manter o projeto seguro, organizado e fácil de atualizar no GitHub, siga sempre estas regras essenciais:

### 1. Proteção de Chaves Secretas (`.gitignore`)
Nunca envie suas chaves de API (`GEMINI_API_KEY` e `GROQ_API_KEY`) ou arquivos temporários do Python para o GitHub. Garanta que o arquivo `.gitignore` na raiz do projeto contenha:

```gitignore
# Variáveis de ambiente (chaves secretas)
.env
backend/.env

# Arquivos temporários e cache do Python
__pycache__/
*.py[cod]
.venv/
venv/
```

### 2. O que DEVE ser salvo no GitHub vs. O que NÃO DEVE
✅ Salvar no Git: Código-fonte (backend/app/, frontend/, ingest.py), requirements.txt, textos originais (backend/data/books/), o banco vetorial já processado (backend/qdrant_db/) e este README.md.

❌ Nunca salvar no Git: O arquivo backend/.env e pastas __pycache__/.

### 3. Passo a Passo para Salvar Alterações no GitHub
Sempre que editar qualquer arquivo no Codespaces ou localmente, execute este fluxo na raiz do projeto (/workspaces/ia-filosofica):

### Garante que você está na pasta raiz do projeto

```
cd /workspaces/ia-filosofica
```

### Verifica quais arquivos foram alterados (confira se o .env NÃO aparece na lista)

```
git status
```
  
### Prepara todas as alterações válidas

```
git add .
```
  
### Cria um ponto de salvamento com uma mensagem clara do que foi feito

```
git commit -m "Descreva brevemente o que você alterou"
```
  
### Envia para o repositório remoto no GitHub

```
git push
```

### 4. Atenção ao Diretório ao Rodar Comandos
Para comandos do Git (git status, git add, git commit, git push): execute sempre na pasta raiz (/workspaces/ia-filosofica).

Para comandos do Backend (uvicorn app.main:app --reload --port 8000 ou python ingest.py): entre antes na pasta backend/ (cd backend).

### 5. Se precisar trocar uma chave de API no futuro
Gere a nova chave no Google AI Studio ou Groq Console.

Atualize o valor dentro de backend/.env (sem aspas e sem espaços ao redor do =).

Como o .env é ignorado pelo Git, não é necessário dar git push para essa troca — apenas reinicie o servidor (Ctrl + C e uvicorn app.main:app --reload --port 8000).