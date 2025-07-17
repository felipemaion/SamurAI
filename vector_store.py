import pickle
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
from logger_config import get_logger

logger = get_logger(__name__)


def extract_keywords(question: str, nlp) -> List[str]:
    """
    Extrai palavras-chave de uma pergunta em português.
    Retorna uma lista de lemas em minúsculas, sem stopwords.
    Prioriza substantivos e nomes próprios.
    """
    doc = nlp(question)

    keywords = [
        token.lemma_.lower()
        for token in doc
        if (
            token.pos_ in {"NOUN", "PROPN"}  # substantivo ou nome próprio
            and not token.is_stop  # não é stopword
            and not token.is_punct  # não é pontuação
            and token.is_alpha  # apenas letras
        )
    ]

    # fallback: se nada foi pego, devolve as palavras sem POS filter
    if not keywords:
        keywords = [
            token.lemma_.lower()
            for token in doc
            if (not token.is_stop and not token.is_punct and token.is_alpha)
        ]

    return keywords


def extract_identifier(question: str) -> Optional[str]:
    """
    Extrai identificador do tipo PDL 3-2023, PL 45/2022, PEC 10 de 2021, etc. e devolve normalizado com .pdf no final.
    """
    # Aceita hífen, barra ou "de" como separador
    pattern = re.compile(
        r"\b(PDL|PL|PEC)[\s\-]*(\d{1,4})[\s\-\/]*(?:de\s*)?(\d{4})\b", re.IGNORECASE
    )

    match = pattern.search(question)
    if match:
        sigla, numero, ano = match.groups()
        return f"{sigla.upper()} {numero}-{ano}.pdf"
    return None


class VectorStore:
    def __init__(
        self, nlp, embedding_model: str = "distiluse-base-multilingual-cased-v2"
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.chunk_to_doc_map = []  # lista de (doc_id, chunk_idx)
        self.nlp = nlp

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model.encode(texts, show_progress_bar=True)

    def add_documents(self, documents: List[Any]):
        all_chunks = []

        for doc in documents:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["filename"] = doc.filename
            doc.metadata["path"] = doc.metadata.get("path", "")
            doc.metadata["folder"] = os.path.dirname(doc.metadata["path"])
            doc.metadata["processed_at"] = datetime.now().isoformat()

            for i, chunk in enumerate(doc.chunks):
                all_chunks.append(chunk)
                self.chunk_to_doc_map.append(
                    (doc.id, i)
                )  # ✅ CORRETO: sempre uma tupla

        if not all_chunks:
            logger.warning("Nenhum chunk para processar")
            return

        embeddings = self.create_embeddings(all_chunks)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings.astype("float32"))
        self.documents.extend(documents)

        logger.info(
            f"✅ Adicionados {len(all_chunks)} chunks de {len(documents)} documentos ao índice."
        )

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            logger.warning("Índice FAISS não inicializado.")
            return []

        identifier = extract_identifier(query)
        matching_docs = []

        if identifier:
            matching_docs = [
                doc
                for doc in self.documents
                if identifier.lower() in doc.filename.lower()
            ]
            if matching_docs:
                logger.info(
                    f"✅ Pré-filtro encontrou {len(matching_docs)} documento(s) com identificador exato: {identifier}"
                )
                for doc in matching_docs:
                    logger.info(f"    - {doc.filename}")
                if len(matching_docs) > 5:
                    logger.info(
                        f"⚠️ Mais de 5 documentos encontrados no pré-filtro. Ignorando pré-filtro e partindo para busca vetorial."
                    )
                    matching_docs = []

        if not matching_docs:
            keywords = extract_keywords(query, self.nlp)
            logger.info(f"🔷 Palavras-chave extraídas da pergunta: {keywords}")

            keyword_matching_docs = []
            for doc in self.documents:
                content_lower = " ".join(doc.chunks).lower()
                if any(kw in content_lower for kw in keywords):
                    keyword_matching_docs.append(doc)

            if keyword_matching_docs:
                logger.info(
                    f"✅ Pré-filtro por palavras-chave encontrou {len(keyword_matching_docs)} documento(s)."
                )
                for doc in keyword_matching_docs:
                    logger.info(f"    - {doc.filename}")
                if len(keyword_matching_docs) > 5:
                    logger.info(
                        f"⚠️ Mais de 5 documentos encontrados por palavras-chave. Ignorando pré-filtro e partindo para busca vetorial."
                    )
                    keyword_matching_docs = []

                if keyword_matching_docs:
                    return [
                        {
                            "document": doc.filename,
                            "content": doc.chunks[0],
                            "score": 0.8,
                            "metadata": doc.metadata,
                        }
                        for doc in keyword_matching_docs
                    ]

        if matching_docs and len(matching_docs) <= 5:
            return [
                {
                    "document": doc.filename,
                    "content": doc.chunks[0],
                    "score": 1.0,
                    "metadata": doc.metadata,
                }
                for doc in matching_docs
            ]

        logger.info(
            "ℹ️ Nenhum documento encontrado por pré-filtros (ou mais de 5). Fazendo busca vetorial."
        )

        query_embedding = self.create_embeddings([query])
        scores, indices = self.index.search(query_embedding.astype("float32"), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_to_doc_map):
                doc_id, chunk_idx = self.chunk_to_doc_map[idx]  # ✅ agora SEM ERRO
                doc = next((d for d in self.documents if d.id == doc_id), None)
                if doc:
                    results.append(
                        {
                            "document": doc.filename,
                            "content": doc.chunks[chunk_idx],
                            "score": float(score),
                            "metadata": doc.metadata,
                        }
                    )

        logger.info(f"🔷 Busca vetorial retornou {len(results)} resultados.")
        return results

    def save_index(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.index is None:
            logger.warning("Nenhum índice para salvar")
            return

        faiss.write_index(self.index, os.path.join(path, "faiss_index.idx"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "chunk_to_doc_map": self.chunk_to_doc_map,
                },
                f,
            )
        logger.info("✅ Índice e metadados salvos com sucesso.")

    def load_index(self, path: str) -> bool:
        try:
            idx_path = os.path.join(path, "faiss_index.idx")
            meta_path = os.path.join(path, "metadata.pkl")
            if not os.path.exists(idx_path) or not os.path.exists(meta_path):
                logger.info(
                    "Arquivos de índice não encontrados, será necessário reindexar."
                )
                return False

            self.index = faiss.read_index(idx_path)
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.chunk_to_doc_map = data["chunk_to_doc_map"]

            logger.info(
                f"✅ Índice carregado com {len(self.documents)} documentos e {len(self.chunk_to_doc_map)} chunks."
            )
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao carregar índice: {e}")
            return False
