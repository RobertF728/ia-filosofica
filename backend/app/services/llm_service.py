import os
from google import genai
from google.genai import types
from groq import Groq

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
"""

def generate_philosophical_response(user_message: str, contexts: list[dict], history: list[dict]) -> str:
    # Formata os trechos resgatados do Qdrant
    context_text = "\n\n".join(
        [f"[Obra: {c['obra']}]\nTrecho: \"{c['texto']}\"" for c in contexts]
    )
    
    # Formata o histórico recente da conversa (últimas 4 mensagens para economizar tokens)
    history_text = ""
    if history:
        recent_history = history[-4:]
        history_text = "Histórico recente da conversa:\n" + "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in recent_history]
        ) + "\n\n"

    full_prompt = f"{history_text}CONTEXTO FILOSÓFICO RECUPERADO:\n{context_text}\n\nPERGUNTA DO USUÁRIO:\n{user_message}"

    # 1. Tentativa Principal: Google Gemini (2.5 Flash)
    if GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.5, # Equilíbrio entre empatia e fidelidade ao texto
                )
            )
            return response.text
        except Exception as e:
            print(f"[AVISO] Falha no Gemini API ({e}). Acionando fallback para Groq...")

    # 2. Tentativa de Reserva (Fallback): Groq Cloud (Llama 3.3 70B)
    if GROQ_API_KEY:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.5,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"[ERRO] Falha também na Groq API ({e}).")

    raise RuntimeError("Nenhum provedor de LLM respondeu. Verifique as chaves de API.")