#!/usr/bin/env python3
"""
Script para criar venv e instalar dependências para o RAG System com LMStudio
compatível com Python 3.13
"""

import subprocess
import sys
import os
import venv
from pathlib import Path

VENV_DIR = Path("./.venv")


def run_command(command, env=None):
    """Executa um comando (list[str]) e retorna o resultado"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"❌ Erro ao executar: {' '.join(command)}")
            print(f"   Saída: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False


def create_venv():
    """Cria virtual environment"""
    print(f"🐍 Criando virtual environment em: {VENV_DIR}")
    venv.create(VENV_DIR, with_pip=True)
    print("✅ venv criada com sucesso!")


def get_venv_python():
    """Retorna o caminho para o Python dentro da venv"""
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    else:
        return VENV_DIR / "bin" / "python"


def install_dependencies(python_exe):
    """Instala as dependências necessárias"""
    print("🚀 Instalando dependências na venv...")

    # Atualizar pip
    print("🔄 Atualizando pip na venv...")
    run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])

    # Lista de pacotes principais
    packages = [
        "pypdf>=4.0.1",
        "sentence-transformers>=2.2.2",
        "requests>=2.31.0",
        "numpy>=1.24.3",
        "torch>=2.0.0",
        "transformers>=4.21.0",
        "nltk>=3.8.1",
        "lmstudio>=0.0.2",
        "bs4>=0.0.2",
    ]

    # Instalar FAISS separadamente
    print("📦 Instalando FAISS...")
    faiss_success = run_command(
        [str(python_exe), "-m", "pip", "install", "faiss-cpu>=1.7.4"]
    )

    if not faiss_success:
        print("⚠️  Tentando instalar FAISS alternativo...")
        faiss_success = run_command(
            [str(python_exe), "-m", "pip", "install", "faiss-cpu==1.8.0"]
        )

        if not faiss_success:
            print(
                "❌ Não foi possível instalar FAISS. Tentando sem versão específica..."
            )
            run_command([str(python_exe), "-m", "pip", "install", "faiss-cpu"])

    # Instalar outros pacotes
    for package in packages:
        print(f"📦 Instalando {package}...")
        success = run_command([str(python_exe), "-m", "pip", "install", package])
        if not success:
            print(f"⚠️  Erro ao instalar {package}, continuando...")

    # Baixar dados do nltk
    print("📚 Baixando stopwords para NLTK...")
    run_command([str(python_exe), "-m", "nltk.downloader", "stopwords"])

    print("✅ Instalação concluída!")


def main():
    """Função principal"""
    print("🔧 Configurador de Dependências - RAG System (LMStudio)")
    print("=" * 50)

    # Verificar versão do Python
    python_version = sys.version_info
    print(
        f"Python detectado: {python_version.major}.{python_version.minor}.{python_version.micro}"
    )

    if python_version.major < 3 or (
        python_version.major == 3 and python_version.minor < 8
    ):
        print("❌ Este sistema requer Python 3.8 ou superior")
        return False

    # Criar venv
    if VENV_DIR.exists():
        print(f"ℹ️  venv já existe em: {VENV_DIR}")
    else:
        create_venv()

    python_exe = get_venv_python()
    print(f"🐍 Python da venv: {python_exe}")

    install_dependencies(python_exe)

    print("\n🎉 Configuração concluída com sucesso!")
    print("\nPróximos passos:")
    if os.name == "nt":
        print(rf"1. Ative a venv: .\.venv\Scripts\activate")
    else:
        print(rf"1. Ative a venv: source .venv/bin/activate")
    print(f"2. Execute: python baixarTeorPL.py")
    print(f"3. Execute: 'python rag_system.py' ou 'python rag_with_keywords.py'")
    return True


if __name__ == "__main__":
    main()
