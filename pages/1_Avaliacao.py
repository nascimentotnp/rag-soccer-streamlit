"""TELA DE AVALIAÇÃO - mostra os relatórios gerados pelo RAG (rag avaliar ...).

A interface não recalcula nada: só LÊ os relatórios do repositório do RAG. Avaliar custa
chamadas ao LLM e minutos de execução, por isso fica na linha de comando.

⬅️ Passo anterior: rag_ui/componentes.py
➡️ Fim do roteiro da interface. Desafio: tests/test_app.py mostra como testar telas sem navegador.
"""
import streamlit as st

from rag_ui import servico

st.set_page_config(page_title="Avaliação do RAG", page_icon="📊", layout="wide")
st.title("📊 Como este RAG foi avaliado")
st.caption("Gabarito com 15 perguntas (diretas, com gíria e fora do escopo). "
           "Hit@4 = o trecho certo está entre os 4 primeiros; MRR = quão perto do topo ele está.")

RELATORIOS = {
    "Reranking: antes × depois": ("rerank_ab.md", "rag avaliar --ab-rerank"),
    "Recuperação adaptativa": ("roteamento.md", "rag avaliar --roteamento"),
    "Técnicas de busca": ("resultados.md", "rag avaliar"),
}

pasta = servico.pasta_de_relatorios()
abas = st.tabs(list(RELATORIOS))
for aba, (arquivo, comando) in zip(abas, RELATORIOS.values()):
    with aba:
        caminho = pasta / arquivo
        if caminho.exists():
            st.markdown(caminho.read_text(encoding="utf-8"))
        else:
            st.info(f"Relatório ainda não gerado. No repositório do RAG, rode: `{comando}`")
