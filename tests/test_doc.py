from pathlib import Path
from document_processor_spacy import DocumentProcessor


def test_processor():
    # Caminho para um PDF de teste (ajuste para o caminho correto do seu PDF)
    pdf_path = Path("./content/PEC 3-2023.pdf")

    if not pdf_path.exists():
        print(f"❌ Arquivo PDF não encontrado: {pdf_path}")
        return

    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    document = processor.process_document(str(pdf_path))

    if not document:
        print("❌ Falha ao processar documento")
        return

    print(f"✅ Documento processado: {document.filename}")
    print(f"📄 Total de chunks: {len(document.chunks)}")
    print("-" * 40)

    for i, chunk in enumerate(document.chunks[:5], 1):  # Mostra só os 5 primeiros
        print(f"Chunk {i}:")
        print(chunk)
        print("-" * 40)


if __name__ == "__main__":
    test_processor()
