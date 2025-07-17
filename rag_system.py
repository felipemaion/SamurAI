import os
from pathlib import Path
import spacy


from document_processor_spacy import DocumentProcessor  # usa spaCy para chunks
from vector_store import VectorStore
from lmstudio_client import LMStudioClient
from logger_config import get_logger

logger = get_logger(__name__)


# carrega uma vez o modelo spaCy para português
nlp = spacy.load("pt_core_news_sm")


class RAGSystem:
    def __init__(self, nlp, content_dir: str = "./content"):
        self.content_dir = Path(content_dir)
        self.processor = DocumentProcessor(nlp=nlp)
        self.nlp = nlp
        self.vector_store = VectorStore(nlp=nlp)
        self.llm_client = LMStudioClient()
        self.index_path = "./rag_index_v1"
        self.content_dir.mkdir(exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)

    def list_indexed_files(self, verbose=False):
        if not self.vector_store.documents:
            print("📂 Nenhum documento indexado ainda.")
            return

        print("📄 Documentos indexados:")
        for doc in self.vector_store.documents:
            print(
                f"- {doc.filename} ({len(doc.chunks)} chunks, processado em {doc.metadata.get('processed_at', 'N/A')})"
            )
            if verbose:
                for i, chunk in enumerate(doc.chunks):
                    print(f"    Chunk {i+1}: {chunk[:200].replace('\n', ' ')}...")
                    if i == 0:  # só mostra o primeiro chunk,
                        # remova para listar todos
                        break

    def index_documents(self, force_reindex=False):
        if not force_reindex and self.vector_store.load_index(self.index_path):
            logger.info("✅ Índice carregado do disco.")
            return
        pdfs = list(self.content_dir.rglob("*.pdf"))
        if not pdfs:
            logger.warning(f"⚠️ Nenhum PDF encontrado em {self.content_dir}")
            return
        documents = []
        for pdf in pdfs:
            logger.info(f"Processando {pdf}")
            doc = self.processor.process_document(str(pdf))
            if doc:
                documents.append(doc)
        self.vector_store.add_documents(documents)
        self.vector_store.save_index(self.index_path)
        logger.info(f"Indexação concluída: {len(documents)} documentos")

    def query(self, question: str, k: int = 5) -> str:
        """Responde uma pergunta usando RAG com pré-filtro + busca vetorial."""

        results = self.vector_store.search(question, k=k)

        if not results:
            return "Não sei. Não encontrei informações relevantes nos documentos indexados."

        # DEBUG dos resultados
        context_parts = []
        sources = set()

        for result in results:
            logger.info(
                f"[DEBUG] Score: {result['score']:.4f} | Doc: {result['document']} | Chunk: {result['content'][:100]}"
            )
            context_parts.append(result["content"])
            sources.add(result["document"])

        # monta contexto e lista de fontes
        context = "\n\n".join(context_parts)
        source_list = ", ".join(sources)

        system_prompt = """
            Você é um assistente que responde perguntas baseado exclusivamente no contexto fornecido.
            Regras importantes:
            1. Responda APENAS com base no contexto fornecido.
            2. Se a informação não estiver no contexto, responda "Não sei".
            3. Sempre mencione os documentos fonte no início da resposta.
            4. Responda em português brasileiro.
            5. Seja preciso e objetivo.
            6. Não invente respostas, apenas responda com base no contexto fornecido.
        """

        prompt = f"""Contexto dos documentos: {context}

        Pergunta: {question}

        Com base no contexto acima, responda a pergunta mencionando os documentos fonte no início da resposta.
        Se a informação não estiver disponível no contexto, responda apenas "Não sei".
        """

        response = self.llm_client.generate(prompt, system_prompt)

        if "não sei" not in response.lower():
            response = f"De acordo com o(s) documento(s) {source_list}:\n\n{response}"

        return response

    def interactive_chat(self):
        print("🤖 Sistema RAG inicializado!")
        print(
            "Digite 'sair' para encerrar, 'reindex' para reindexar documentos, 'list' para listar documentos."
        )
        print("-" * 50)
        try:
            while True:
                question = input("\n📝 Sua pergunta: ").strip()
                if question.lower() in ["sair", "quit", "exit"]:
                    print("Tchau! 👋")
                    break
                if question.lower() == "list":
                    self.list_indexed_files()
                    continue
                if question.lower() == "list verbose":
                    self.list_indexed_files(verbose=True)
                    continue
                if question.lower() == "reindex":
                    self.index_documents(force_reindex=True)
                    continue
                if not question:
                    continue
                print("🔍 Buscando...")
                print(f"\n🤖 Resposta:\n{self.query(question)}")
        except KeyboardInterrupt:
            print("\nTchau! 👋")


def main():
    print("🚀 Iniciando Sistema RAG com LM Studio...")
    rag = RAGSystem(nlp=nlp)
    rag.index_documents()
    rag.interactive_chat()


if __name__ == "__main__":
    main()
