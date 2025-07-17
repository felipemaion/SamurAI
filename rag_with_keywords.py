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
from collections import Counter

# Bibliotecas para processamento de PDF
import pypdf
from sentence_transformers import SentenceTransformer
import faiss
import pickle

import lmstudio as lms

import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams

# garantir que stopwords estão baixadas
import nltk

try:
    stopwords.words("portuguese")
except LookupError:
    nltk.download("stopwords")

PORTUGUESE_STOPWORDS = set(stopwords.words("portuguese"))
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
            "max_tokens": 3000,
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
            self.embedding_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )

        self.index = None
        self.documents = []
        self.chunk_to_doc_map = []

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            return self.embedding_model.encode(texts, show_progress_bar=True)
        except Exception:
            return self.embedding_model.encode(texts)

    # def extract_keywords(self, text: str, top_n=5) -> List[str]:
    #     words = [w.lower() for w in text.split() if len(w) > 3]
    #     counter = Counter(words)
    #     keywords = [w for w, _ in counter.most_common(top_n)]
    #     logger.info(f"📌 Keywords extraídas: {keywords}")
    #     return keywords

    def extract_keywords(self, text: str, top_n=200) -> List[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        filtered = [w for w in tokens if len(w) > 3 and w not in PORTUGUESE_STOPWORDS]
        counter = Counter(filtered)

        # palavras mais comuns
        keywords = [w for w, _ in counter.most_common(top_n)]

        # menos comuns
        least_common = [w for w, _ in counter.most_common()[-top_n:]]
        keywords.extend(least_common)

        # todas únicas
        unique_words = list(counter.keys())
        keywords.extend(unique_words)

        # bigrams e trigrams
        bigrams = [" ".join(bg) for bg in ngrams(filtered, 2)]
        trigrams = [" ".join(tg) for tg in ngrams(filtered, 3)]

        bigram_counter = Counter(bigrams)
        trigram_counter = Counter(trigrams)

        keywords.extend([w for w, _ in bigram_counter.most_common(top_n // 2)])
        keywords.extend([w for w, _ in trigram_counter.most_common(top_n // 4)])

        logger.info(f"📌 Keywords + ngrams + únicas extraídas: {keywords}")
        return list(set(keywords))

    def add_documents(self, documents: List[Document]):
        all_chunks = []

        for doc in documents:
            keywords = self.extract_keywords(doc.content)
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["keywords"] = keywords

            for chunk in doc.chunks:
                all_chunks.append(chunk)
                self.chunk_to_doc_map.append(doc.id)

        if not all_chunks:
            logger.warning("Nenhum chunk para processar")
            return

        embeddings = self.create_embeddings(all_chunks)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings.astype("float32"))
        self.documents.extend(documents)

        logger.info(
            f"Adicionados {len(all_chunks)} chunks de {len(documents)} documentos ao índice"
        )

    def search(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        if self.index is None:
            logger.warning("❌ Índice vetorial não inicializado.")
            return []

        query_words = [
            w
            for w in re.findall(r"\b\w+\b", query.lower())
            if len(w) > 3 and w not in PORTUGUESE_STOPWORDS
        ]
        logger.info(f"🔍 Palavras-chave da query: {query_words}")

        # Conta quantas vezes as keywords da query aparecem nas keywords do documento
        doc_scores = []
        for doc in self.documents:
            doc_keywords = doc.metadata.get("keywords", [])
            matches = sum(
                1
                for kw in doc_keywords
                if any(qw in kw or kw in qw for qw in query_words)
            )
            if matches > 0:
                doc_scores.append((doc, matches))

        if not doc_scores:
            logger.info("🚨 Nenhum documento com keywords compatíveis encontrado.")
            return []

        # Ordena documentos por maior número de matches
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in doc_scores[:5]]  # pega os 5 melhores documentos

        logger.info(f"✅ Documentos candidatos: {[doc.filename for doc in top_docs]}")

        # Chunks candidatos com prioridade para correspondência literal
        literal_hits = []
        candidate_chunks = []
        candidate_chunk_to_doc = []

        for doc in top_docs:
            for chunk in doc.chunks:
                if any(qw in chunk.lower() for qw in query_words):
                    literal_hits.append(
                        {
                            "document": doc.filename,
                            "content": chunk,
                            "score": 1.0,
                            "metadata": doc.metadata,
                        }
                    )
                candidate_chunks.append(chunk)
                candidate_chunk_to_doc.append(doc.id)

        # Se houver correspondências literais, devolve no máximo k balanceando documentos
        if literal_hits:
            logger.info(
                f"🔷 {len(literal_hits)} resultados com correspondência literal."
            )
            per_doc = max(1, k // len(top_docs))
            results = []
            seen = {}
            for hit in literal_hits:
                doc = hit["document"]
                seen.setdefault(doc, 0)
                if seen[doc] < per_doc:
                    results.append(hit)
                    seen[doc] += 1
                if len(results) >= k:
                    break
            return results

        # Fallback: busca vetorial
        logger.info(
            f"🧲 Nenhum literal encontrado, buscando por similaridade vetorial."
        )
        query_embedding = self.create_embeddings([query])
        candidate_embeddings = self.create_embeddings(candidate_chunks)

        dim = candidate_embeddings.shape[1]
        temp_index = faiss.IndexFlatIP(dim)
        temp_index.add(candidate_embeddings.astype("float32"))

        scores, indices = temp_index.search(
            query_embedding.astype("float32"), min(k * 5, len(candidate_chunks))
        )

        # Balanceia chunks por documento
        results = []
        seen_docs = {}
        for score, idx in zip(scores[0], indices[0]):
            doc_id = candidate_chunk_to_doc[idx]
            doc = next(d for d in self.documents if d.id == doc_id)
            seen_docs.setdefault(doc.filename, 0)
            if seen_docs[doc.filename] < 2:  # no máximo 2 chunks por doc
                results.append(
                    {
                        "document": doc.filename,
                        "content": candidate_chunks[idx],
                        "score": float(score),
                        "metadata": doc.metadata,
                    }
                )
                seen_docs[doc.filename] += 1
            if len(results) >= k:
                break

        logger.info(f"🔷 {len(results)} resultados finais balanceados retornados.")
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

        self.index_path = "./rag_index_v2"

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
