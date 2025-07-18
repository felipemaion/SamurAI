# 📚 SamurAI - RAG System com LMStudio

> Sistema RAG (Retrieval-Augmented Generation) para responder perguntas com base em documentos PDF, usando embeddings vetoriais, pré-processamento com palavras-chave e LMStudio como modelo LLM.

---

## 🚀 Visão Geral (Objetivo)

Este projeto implementa um sistema RAG para responder perguntas em português com precisão e referências aos documentos utilizados.  
Ele indexa documentos PDF, divide-os em chunks, cria embeddings vetoriais e utiliza LMStudio para gerar respostas baseadas no contexto recuperado.

### Componentes principais:

✅ **Pré-processamento de PDFs**: Extração de texto, divisão em chunks, geração de embeddings.  
✅ **Indexação**: Armazenamento dos vetores em FAISS, junto com metadados e palavras-chave.  
✅ **Busca**:

- `rag_system.py` — Sistema Principal.

✅ **LMStudio**: gera a resposta baseada no contexto recuperado.

---

## 📁 Estrutura do projeto

```
.
├── baixarTeorPL.py            # script auxiliar para baixar PDFs
├── install_script.py          # script para criar venv e instalar dependências
├── rag_system.py              # RAG com busca puramente vetorial
├── .venv/                     # será criado automaticamente
├── content/                   # coloque seus PDFs aqui
├── rag_index_v1/              # índice para rag_system.py
```

---

## 🧰 Pré-requisitos

- Python **3.8+** (recomendado 3.13)
- LMStudio instalado e rodando localmente ([https://lmstudio.ai](https://lmstudio.ai))

---

## 👨‍💻 Instalação

1️⃣ Clone este repositório:

```bash
git clone https://github.com/felipemaion/SamurAI.git
cd SamurAI
```

2️⃣ Rode o instalador:

```bash
python3 install_script.py
```

Ele irá:

- Criar um ambiente virtual `.venv/`.
- Instalar todas as dependências dentro da `.venv/`.
- Baixar os recursos necessários do NLTK.

---

## 🔗 Ativar ambiente virtual

Após a instalação, ative a venv criada:

✅ Linux/Mac:

```bash
source .venv/bin/activate
```

✅ Windows (cmd):

```cmd
.venv\Scripts\activate
```

✅ Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

---

### ⬇ Baixar documentos (Lista fixa, precisaremos melhorar para automatizar)

```bash
python baixarTeorPL.py
```

Ele irá baixar os arquivos de autoria e coautoria e colocar nas respectivas pastas em `content/`

## 📝 Como usar

### 🔍 Indexar documentos e rodar chat

Após os arquivos estarem na pasta `content/` você pode rodar o sistema da RAG:

### 1️⃣ **Modo básico — vetorial puro**

```bash
python rag_system.py
```

- Indexa documentos (ou usa índice salvo em `rag_index_v1/`).
- Busca puramente semântica vetorial.
- Pode retornar vários chunks de um mesmo documento.

## 🧪 Comandos no chat

- Pergunte algo: digite sua pergunta e pressione `Enter`.
- Reindexar: digite `reindex` para reprocessar todos os PDFs.
- Listar documentos indexados: digite `list` ou `list verbose`.
- Sair: digite `sair`, `quit` ou `exit`.

---

## 🌟 Exemplos de perguntas

✅ _O que o Kim fala sobre criptoativos?_  
✅ _Onde nasceu Kim Kataguiri?_  
✅ _Qual é a definição de responsabilidade fiscal no documento X?_

O sistema responderá (pelo menos esse é o objetivo...) com trechos dos documentos e a referência à fonte.

---

## 🤝 Contribuindo

Pull requests são bem-vindos! Para grandes mudanças, por favor abra uma issue primeiro para discutir o que você gostaria de mudar.

---

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

## 📞 Suporte

Se tiver problemas, abra uma [issue no GitHub](https://github.com/felipemaion/SamurAI.git/issues).
