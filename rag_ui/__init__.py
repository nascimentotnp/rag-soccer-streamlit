"""Interface web (Streamlit) para o RAG de regras do futebol.

Ordem de leitura: app.py -> rag_ui/servico.py -> rag_ui/componentes.py -> pages/1_Avaliacao.py

O .env DESTE repositório é lido aqui, no __init__ do pacote, porque ele roda ANTES de qualquer
módulo da interface importar o `rag`. O rag.config usa o RAG_HOME (se definido) no momento do
import para achar o .env com as chaves, os PDFs e o índice.
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
