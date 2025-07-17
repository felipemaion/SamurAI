import os
import re

import logging
from datetime import datetime
from typing import List

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    filename: str
    content: str
    chunks: List[str]
    metadata: dict


def normalize_whitespace(text: str) -> str:
    """
    Normaliza espaços em branco em um texto:
    - Remove espaços duplicados
    - Remove tabs e newlines extras
    """
    # troca tabs e quebras por espaço
    text = re.sub(r"[\t\n\r]+", " ", text)
    # remove múltiplos espaços
    text = re.sub(r" {2,}", " ", text)
    # remove espaços no início/fim
    return text.strip()


class DocumentProcessor:
    """Classe para processar documentos PDF e gerar chunks por sentença com spaCy"""

    def __init__(self, nlp, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Carrega modelo spaCy para português
        try:
            self.nlp = nlp
            logger.info("✅ spaCy carregado com modelo pt_core_news_sm")
        except OSError:
            logger.error(
                "⚠️ Modelo spaCy pt_core_news_sm não encontrado. "
                "Execute: python -m spacy download pt_core_news_sm"
            )
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrai texto de um arquivo PDF"""
        import pypdf

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
        """
        Divide o texto em chunks de no máximo `chunk_size` palavras,
        respeitando sentenças com spaCy.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            sent_len = len(sent.split())

            # Se a próxima sentença cabe no chunk atual
            if current_length + sent_len <= self.chunk_size:
                current_chunk.append(sent)
                current_length += sent_len
            else:
                # Salva chunk atual
                chunks.append(normalize_whitespace(" ".join(current_chunk)))
                # Inicia próximo chunk com overlap opcional
                if self.chunk_overlap > 0 and len(current_chunk) > 0:
                    overlap_words = []
                    # Pega últimas palavras do chunk para overlap
                    for s in reversed(current_chunk):
                        overlap_words = s.split() + overlap_words
                        if len(overlap_words) >= self.chunk_overlap:
                            break
                    current_chunk = [" ".join(overlap_words)]
                    current_length = len(overlap_words)
                else:
                    current_chunk = []
                    current_length = 0
                # adiciona a sentença que não coube
                current_chunk.append(sent)
                current_length += sent_len

        # adiciona o último chunk
        if current_chunk:
            chunks.append(normalize_whitespace(" ".join(current_chunk)))

        logger.info(f"📄 Texto dividido em {len(chunks)} chunks")
        return chunks

    def process_document(self, pdf_path: str) -> Document:
        """Processa um documento PDF completo"""
        filename = os.path.basename(pdf_path)
        content = self.extract_text_from_pdf(pdf_path)
        if not content:
            logger.warning(f"Nenhum conteúdo extraído de {filename}")
            return None

        # Normaliza conteúdo
        content = normalize_whitespace(content)

        # Inclui título + categoria
        categoria = os.path.basename(os.path.dirname(pdf_path))
        content_with_title = f"Título do documento: {filename.replace('.pdf', '')}, categorias:{categoria}\n\n{content}"

        # Normaliza novamente (caso concatenar título tenha adicionado espaços extras)
        content_with_title = normalize_whitespace(content_with_title)

        chunks = self.chunk_text(content_with_title)

        document = Document(
            id=filename,
            filename=filename,
            content=content_with_title,
            chunks=chunks,
            metadata={
                "path": pdf_path,
                "processed_at": datetime.now().isoformat(),
                "num_chunks": len(chunks),
                "categoria": categoria,
            },
        )

        return document
