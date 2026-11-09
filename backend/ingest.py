# Script que lê os livros e envia ao Qdrant

import os                                                    # Permite interagir com o sistema operacional (pastas, caminhos de arquivo)
from glob import glob                                        # Facilita encontrar arquivos que seguem um padrão (ex: todos os .txt de uma pasta)
from qdrant_client import QdrantClient                       # Importa o cliente do banco de dados vetorial Qdrant
from qdrant_client.models import VectorParams, Distance, PointStruct # Estruturas de configuração e registro de dados do Qdrant
from sentence_transformers import SentenceTransformer        # Importa o modelo de IA que converte frases em vetores matemáticos

# Configurações globais do banco de dados
QDRANT_PATH = "./qdrant_db"                                  # Nome da pasta local onde o banco de dados será salvo no disco
COLLECTION_NAME = "filosofia_obras"                          # Nome da "tabela" (coleção) dentro do Qdrant onde ficarão os trechos

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    """
    Divide um texto muito longo (um livro inteiro) em pedaços menores (chunks).
    - chunk_size: tamanho máximo de cada pedaço (700 caracteres).
    - overlap: quantidade de caracteres repetidos entre um pedaço e o próximo (100 caracteres)
      para evitar que uma frase importante seja cortada ao meio e perca o sentido.
    """
    chunks = []                                              # Lista que armazenará todos os pedaços gerados
    start = 0                                                # Posição inicial de leitura no texto
    
    while start < len(text):                                 # Continua fatiando até chegar ao final do livro
        end = start + chunk_size                             # Define onde o pedaço atual termina
        chunk = text[start:end]                              # Extrai a fatia do texto
        chunks.append(chunk)                                 # Guarda a fatia na lista
        start += chunk_size - overlap                        # Avança o ponteiro recuando o 'overlap' para manter continuidade
        
    return chunks                                            # Retorna a lista completa de trechos do livro

def run_ingestion():
    print("1. Inicializando Qdrant em disco e carregando modelo de Embeddings...")
    
    # Conecta ao Qdrant no modo local (salva diretamente na pasta definida, sem precisar de servidor externo)
    client = QdrantClient(path=QDRANT_PATH)
    
    # Carrega um modelo leve e multilíngue (entende perguntas em português e busca trechos em inglês)
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Se a coleção já existir de uma execução anterior, apaga para não duplicar os livros
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    # Cria a coleção no banco vetorial:
    # - size=384: é a quantidade exata de números que o modelo MiniLM gera para cada frase
    # - distance=Distance.COSINE: método matemático (Similaridade de Cosseno) para medir proximidade de significado
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # Procura todos os arquivos .txt na pasta de livros (funciona rodando da raiz ou de dentro da pasta backend)
    book_files = glob("backend/data/books/*.txt") + glob("data/books/*.txt")
    print(f"2. Encontrados {len(book_files)} livros para processamento.")

    points = []                                              # Lista que guardará todos os registros prontos para inserir no banco
    point_id = 1                                             # Contador numérico único para identificar cada trecho

    # Percorre cada arquivo .txt encontrado
    for file_path in book_files:
        # Extrai o nome do arquivo, remove ".txt", troca "_" por espaço e coloca iniciais maiúsculas (ex: "The Prince")
        book_title = os.path.basename(file_path).replace(".txt", "").replace("_", " ").title()
        print(f"   -> Processando obra: {book_title}...")
        
        # Abre e lê todo o conteúdo do livro em texto
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Divide o livro em centenas de pequenos trechos
        chunks = chunk_text(content)

        # Para cada trecho gerado, cria o vetor e os metadados
        for idx, chunk in enumerate(chunks):
            # Transforma o texto do trecho em uma lista de 384 números (embedding)
            vector = model.encode(chunk).tolist()
            
            # Cria o objeto "PointStruct" (o registro completo do Qdrant)
            points.append(
                PointStruct(
                    id=point_id,                             # ID numérico único
                    vector=vector,                           # Os números que representam o significado do texto
                    payload={                                # Metadados (informações extras que queremos recuperar depois)
                        "obra": book_title,
                        "texto": chunk,
                        "chunk_id": idx
                    }
                )
            )
            point_id += 1                                    # Incrementa o ID para o próximo trecho

    # Salva todos os pontos gerados de uma só vez no banco Qdrant
    print(f"3. Salvando {len(points)} trechos no Qdrant local...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("4. Ingestão concluída com sucesso! Pasta './qdrant_db' pronta.")

# Garante que a função só rode se este arquivo for executado diretamente no terminal (python ingest.py)
if __name__ == "__main__":
    run_ingestion()