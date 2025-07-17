import os
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# Bibliotecas para processamento de PDF
import pypdf
from sentence_transformers import SentenceTransformer
import faiss
import pickle

import lmstudio as lms

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Classe para representar um documento processado"""

    id: str
    filename: str
    content: str
    chunks: List[str]
    embeddings: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


class LMStudioClient:
    """Cliente para interagir com LM Studio via API HTTP"""

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url
        self.model = "dolphin3.0-llama3.1-8b"
        # self.model = "maritaca-ai_-_sabia-7b"

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Gera resposta usando LM Studio"""
        url = f"{self.base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1000,
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Erro ao gerar resposta via LM Studio: {e}")
            return "Desculpe, ocorreu um erro ao processar sua pergunta."


class OllamaClient:
    """Cliente para interagir com Ollama"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        # self.model = "llama2:7b"
        self.model = "mistral"

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Gera resposta usando Ollama"""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1000},
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            return "Desculpe, ocorreu um erro ao processar sua pergunta."


class DocumentProcessor:
    """Classe para processar documentos PDF"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrai texto de um arquivo PDF"""
        try:
            with open(pdf_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Erro ao extrair texto do PDF {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """Divide o texto em chunks menores"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())

        return chunks

    def process_document(self, pdf_path: str) -> Document:
        """Processa um documento PDF completo"""
        filename = os.path.basename(pdf_path)
        content = self.extract_text_from_pdf(pdf_path)
        print(content[:300])

        if not content:
            logger.warning(f"Nenhum conteúdo extraído de {filename}")
            return None

        chunks = self.chunk_text(content)

        document = Document(
            id=filename,
            filename=filename,
            content=content,
            chunks=chunks,
            metadata={
                "path": pdf_path,
                "processed_at": datetime.now().isoformat(),
                "num_chunks": len(chunks),
            },
        )

        return document


class VectorStore:
    """Sistema de armazenamento vetorial usando FAISS"""

    def __init__(self, embedding_model: str = "distiluse-base-multilingual-cased-v2"):
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de embedding: {e}")
            logger.info("Tentando modelo alternativo...")
            try:
                # self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                self.embedding_model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2"
                )

            except Exception as e2:
                logger.error(f"Erro ao carregar modelo alternativo: {e2}")
                raise Exception("Não foi possível carregar nenhum modelo de embedding")

        self.index = None
        self.documents = []
        self.chunk_to_doc_map = []

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Cria embeddings para uma lista de textos"""
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            logger.error(f"Erro ao criar embeddings: {e}")
            # Tentar sem progress bar
            try:
                embeddings = self.embedding_model.encode(texts)
                return embeddings
            except Exception as e2:
                logger.error(f"Erro crítico ao criar embeddings: {e2}")
                raise

    def add_documents(self, documents: List[Document]):
        """Adiciona documentos ao índice vetorial"""
        all_chunks = []

        for doc in documents:
            for chunk in doc.chunks:
                all_chunks.append(chunk)
                self.chunk_to_doc_map.append(doc.id)

        if not all_chunks:
            logger.warning("Nenhum chunk para processar")
            return

        # Criar embeddings
        embeddings = self.create_embeddings(all_chunks)

        # Inicializar índice FAISS
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)

        # Adicionar ao índice
        self.index.add(embeddings.astype("float32"))
        self.documents.extend(documents)

        logger.info(
            f"Adicionados {len(all_chunks)} chunks de {len(documents)} documentos ao índice"
        )

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Busca por documentos relevantes"""
        if self.index is None:
            return []

        # Criar embedding da query
        query_embedding = self.create_embeddings([query])

        # Buscar no índice
        scores, indices = self.index.search(query_embedding.astype("float32"), k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunk_to_doc_map):
                doc_id = self.chunk_to_doc_map[idx]

                # Encontrar o documento correspondente
                doc = next((d for d in self.documents if d.id == doc_id), None)
                if doc:
                    chunk_idx = idx - sum(
                        len(d.chunks)
                        for d in self.documents
                        if d.id != doc_id
                        and self.documents.index(d) < self.documents.index(doc)
                    )

                    results.append(
                        {
                            "document": doc.filename,
                            "content": (
                                doc.chunks[chunk_idx]
                                if chunk_idx < len(doc.chunks)
                                else ""
                            ),
                            "score": float(score),
                            "metadata": doc.metadata,
                        }
                    )

        return results

    def save_index(self, path: str):
        """Salva o índice em disco"""
        try:
            # Criar diretório se não existir
            os.makedirs(path, exist_ok=True)

            # Verificar se o índice existe
            if self.index is None:
                logger.warning("Nenhum índice para salvar")
                return

            # Salvar índice FAISS
            index_file = os.path.join(path, "faiss_index.idx")
            faiss.write_index(self.index, index_file)
            logger.info(f"Índice FAISS salvo em: {index_file}")

            # Salvar metadados
            metadata_file = os.path.join(path, "metadata.pkl")
            with open(metadata_file, "wb") as f:
                pickle.dump(
                    {
                        "documents": self.documents,
                        "chunk_to_doc_map": self.chunk_to_doc_map,
                    },
                    f,
                )
            logger.info(f"Metadados salvos em: {metadata_file}")

        except Exception as e:
            logger.error(f"Erro ao salvar índice: {e}")
            raise

    def load_index(self, path: str):
        """Carrega o índice do disco"""
        try:
            index_file = os.path.join(path, "faiss_index.idx")
            metadata_file = os.path.join(path, "metadata.pkl")

            # Verificar se os arquivos existem
            if not os.path.exists(index_file) or not os.path.exists(metadata_file):
                logger.info(
                    "Arquivos de índice não encontrados, será necessário reindexar"
                )
                return False

            # Carregar índice FAISS
            self.index = faiss.read_index(index_file)

            # Carregar metadados
            with open(metadata_file, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.chunk_to_doc_map = data["chunk_to_doc_map"]

            logger.info(f"Índice carregado com {len(self.documents)} documentos")
            return True

        except Exception as e:
            logger.error(f"Erro ao carregar índice: {e}")
            return False


class RAGSystem:
    """Sistema RAG principal"""

    def __init__(self, content_dir: str = "./content"):
        self.content_dir = Path(content_dir)
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
        # self.llm_client = OllamaClient()
        self.llm_client = LMStudioClient()

        self.index_path = "./rag_index_v1"

        # Criar diretórios necessários se não existirem
        self.content_dir.mkdir(exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)

        logger.info(f"Diretório de conteúdo: {self.content_dir}")
        logger.info(f"Diretório de índice: {self.index_path}")

    def list_indexed_files(self):
        """Mostra os arquivos PDF já indexados"""
        if not self.vector_store.documents:
            print("📂 Nenhum documento indexado ainda.")
            return

        print("📄 Documentos indexados:")
        for doc in self.vector_store.documents:
            print(
                f"- {doc.filename} ({len(doc.chunks)} chunks, processado em {doc.metadata['processed_at']})"
            )

    def index_documents(self, force_reindex: bool = False):
        """Indexa todos os documentos PDF no diretório de conteúdo"""
        # Criar diretório de índice se não existir
        os.makedirs(self.index_path, exist_ok=True)

        if not force_reindex and self.vector_store.load_index(self.index_path):
            logger.info("Índice carregado do disco")
            return

        logger.info("Iniciando indexação de documentos...")

        # Encontrar todos os PDFs
        pdf_files = list(self.content_dir.rglob("*.pdf"))

        if not pdf_files:
            logger.warning(f"Nenhum arquivo PDF encontrado em {self.content_dir}")
            return

        documents = []
        for pdf_path in pdf_files:
            logger.info(f"Processando {pdf_path}")
            try:
                doc = self.processor.process_document(str(pdf_path))
                if doc:
                    documents.append(doc)
                else:
                    logger.warning(f"Falha ao processar {pdf_path}")
            except Exception as e:
                logger.error(f"Erro ao processar {pdf_path}: {e}")
                continue

        if documents:
            try:
                self.vector_store.add_documents(documents)
                self.vector_store.save_index(self.index_path)
                logger.info(
                    f"Indexação concluída! {len(documents)} documentos processados"
                )
            except Exception as e:
                logger.error(f"Erro durante indexação: {e}")
                raise
        else:
            logger.warning("Nenhum documento foi processado com sucesso")

    def query(self, question: str, k: int = 5) -> str:
        """Responde uma pergunta usando RAG"""
        results = self.vector_store.search(question, k=k)

        if not results:
            return "Não sei. Não encontrei informações relevantes nos documentos indexados."

        # DEBUG dos resultados
        for result in results:
            logger.info(
                f"[DEBUG] Score: {result['score']:.4f} | Doc: {result['document']} | Chunk: {result['content'][:100]}"
            )

        context_parts = []
        sources = set()

        for result in results:
            context_parts.append(result["content"])
            sources.add(result["document"])

        context = "\n\n".join(context_parts)
        source_list = ", ".join(sources)

        system_prompt = """Você é um assistente que responde perguntas baseado exclusivamente no contexto fornecido.
        Regras importantes:
        1. Responda APENAS com base no contexto fornecido
        2. Se a informação não estiver no contexto, responda "Não sei"
        3. Sempre mencione os documentos fonte no início da resposta
        4. Responda em português brasileiro
        5. Seja preciso e objetivo
        6. Não invente respostas, apenas responda com base no contexto fornecido."""

        prompt = f"""Contexto dos documentos:
    {context}

    Pergunta: {question}

    Com base no contexto acima, responda a pergunta mencionando os documentos fonte no início da resposta.
    Se a informação não estiver disponível no contexto, responda apenas "Não sei"."""

        response = self.llm_client.generate(prompt, system_prompt)

        if "não sei" not in response.lower():
            response = f"De acordo com o(s) documento(s) {source_list}:\n\n{response}"

        return response

    def interactive_chat(self):
        """Interface interativa para chat"""
        print("🤖 Sistema RAG inicializado!")
        print("Digite 'sair' para encerrar, 'reindex' para reindexar documentos")
        print("-" * 50)

        while True:
            try:
                question = input("\n📝 Sua pergunta: ").strip()

                if question.lower() in ["sair", "quit", "exit"]:
                    print("Tchau! 👋")
                    break
                if question.lower() == "list":
                    self.list_indexed_files()
                    continue
                if question.lower() == "reindex":
                    self.index_documents(force_reindex=True)
                    continue

                if not question:
                    continue

                print("🔍 Buscando...")
                response = self.query(question)
                print(f"\n🤖 Resposta:\n{response}")

            except KeyboardInterrupt:
                print("\nTchau! 👋")
                break
            except Exception as e:
                print(f"❌ Erro: {e}")


# def main():
#     """Função principal"""
#     print("🚀 Iniciando Sistema RAG com Llama2...")
#     print(f"Python: {sys.version}")

#     # Verificar se Ollama está rodando
#     try:
#         response = requests.get("http://localhost:11434/api/tags", timeout=5)
#         if response.status_code != 200:
#             print(
#                 "❌ Ollama não está rodando. Inicie o Ollama antes de usar o sistema."
#             )
#             print("   Execute: ollama serve")
#             return
#     except requests.exceptions.ConnectionError:
#         print(
#             "❌ Não foi possível conectar ao Ollama. Certifique-se de que está rodando na porta 11434."
#         )
#         print("   Execute: ollama serve")
#         return
#     except requests.exceptions.Timeout:
#         print(
#             "❌ Timeout ao conectar com Ollama. Verifique se o serviço está respondendo."
#         )
#         return
#     except Exception as e:
#         print(f"❌ Erro inesperado ao conectar com Ollama: {e}")
#         return

#     # Verificar se o modelo está disponível
#     try:
#         response = requests.get("http://localhost:11434/api/tags", timeout=5)
#         models = response.json().get("models", [])
#         llama_models = [m for m in models if "mistral" in m.get("name", "").lower()]

#         if not llama_models:
#             print("❌ Modelo mistral não encontrado.")
#             print("   Execute: ollama pull mistral")
#             return
#         else:
#             print(f"✅ Modelo encontrado: {llama_models[0]['name']}")

#     except Exception as e:
#         print(f"⚠️  Não foi possível verificar modelos disponíveis: {e}")

#     try:
#         # Inicializar sistema
#         rag = RAGSystem()

#         # Indexar documentos
#         rag.index_documents()

#         # Iniciar chat interativo
#         rag.interactive_chat()

#     except Exception as e:
#         print(f"❌ Erro ao inicializar sistema: {e}")
#         print("   Verifique se todas as dependências estão instaladas corretamente.")
#         print("   Execute: python install_dependencies.py")


def main():
    """Função principal"""
    print("🚀 Iniciando Sistema RAG com LM Studio...")
    print(f"Python: {sys.version}")

    try:
        # Inicializar sistema
        rag = RAGSystem()

        # Indexar documentos
        rag.index_documents()

        # Iniciar chat interativo
        rag.interactive_chat()

    except Exception as e:
        print(f"❌ Erro ao inicializar sistema: {e}")
        print("   Verifique se todas as dependências estão instaladas corretamente.")


if __name__ == "__main__":
    main()
