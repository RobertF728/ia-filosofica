import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

# Garante o carregamento do .env na pasta backend/
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """
Você é um mentor filosófico acolhedor, sábio, empático e reflexivo. 
O usuário vai compartilhar angústias, dúvidas sobre a vida, relacionamentos, propósito, fracassos ou decisões difíceis.

SUAS REGRAS RÍGIDAS:
1. Responda SEMPRE em português fluente, natural e acolhedor.
2. Baseie a sua resposta EXCLUSIVAMENTE nos trechos das obras filosóficas fornecidos no CONTEXTO abaixo.
3. NÃO invente citações e NÃO atribua frases a filósofos que não estejam no contexto fornecido.
4. Como os trechos do contexto estão em inglês, traduza a citação escolhida para um português elegante e fiel ao sentido original.
5. Estruture a sua resposta de forma clara:
   - Comece validando o sentimento do usuário com empatia.
   - Apresente a citação traduzida em destaque (entre aspas), informando claramente o nome da obra (e o autor correspondente: Meditations = Marco Aurélio; The Enchiridion = Epicteto; The Republic = Platão; Beyond Good and Evil = Nietzsche; The Prince = Maquiavel).
   - Explique o significado daquela reflexão em linguagem simples e aplique-a à situação do usuário.
6. Se o contexto fornecido não tiver relação suficiente com a pergunta, admita com humildade que os textos disponíveis no momento não abordam diretamente esse ponto, mas ofereça uma reflexão geral baseada nos princípios das obras fornecidas.
7. Seja conciso e objetivo: mantenha a resposta acolhedora e profunda, mas sem textos excessivamente longos (máximo de 3 parágrafos curtos além da citação).
"""

def generate_philosophical_response(user_message: str, contexts: list[dict], history: list[dict]) -> str:
    context_text = "\n\n".join(
        [f"[Obra: {c['obra']}]\nTrecho: \"{c['texto']}\"" for c in contexts]
    )
    
    history_text = ""
    if history:
        recent_history = history[-4:]
        history_text = "Histórico recente da conversa:\n" + "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in recent_history]
        ) + "\n\n"

    full_prompt = (
        f"{history_text}"
        f"CONTEXTO FILOSÓFICO RECUPERADO:\n{context_text}\n\n"
        f"PERGUNTA DO USUÁRIO:\n{user_message}"
    )

    erros = []

    # 1. Tentativa Principal: Google Gemini (gemini-3.5-flash-lite indicado pela API + fallback 3.8)
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        for gemini_model in ['gemini-3.5-flash-lite', 'gemini-3.8-flash']:
            try:
                response = client.models.generate_content(
                    model=gemini_model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.5,
                    )
                )
                return response.text
            except Exception as e:
                erros.append(f"Gemini ({gemini_model}): {str(e)}")
    else:
        erros.append("Gemini: GEMINI_API_KEY não encontrada")

    # 2. Tentativa de Reserva (Fallback): Groq Cloud (detecta automaticamente um modelo ativo na conta)
    if GROQ_API_KEY:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            available_models = [
                m.id for m in groq_client.models.list().data
                if "whisper" not in m.id.lower() and "guard" not in m.id.lower()
            ]
            chosen_model = available_models[0] if available_models else "llama3-8b-8192"
            
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt}
                ],
                model=chosen_model,
                temperature=0.5,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            erros.append(f"Groq: {str(e)}")
    else:
        erros.append("Groq: GROQ_API_KEY não encontrada")

    raise RuntimeError(" | ".join(erros))