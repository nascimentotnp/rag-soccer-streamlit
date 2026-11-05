"""COMPONENTES - pedaços visuais reaproveitados pelas telas.

Separar "como desenhar" (aqui) de "o que perguntar ao RAG" (servico.py) deixa as telas curtas
e facilita mudar o visual sem tocar na lógica.

⬅️ Passo anterior: rag_ui/servico.py
➡️ Próximo passo: pages/1_Avaliacao.py (os relatórios de avaliação do RAG)
"""
import streamlit as st
from rag.pipeline import Answer

from rag_ui import servico

ROTULO_ROTA = {
    "confiante": ("✅ confiante", "a busca simples bastou"),
    "incerto": ("🤔 incerto", "confiança baixa: usou a técnica de 2ª chance"),
    "fora_do_escopo": ("⛔ fora do escopo", "confiança muito baixa: recusou sem chamar o LLM"),
}


def painel_da_resposta(resposta: Answer) -> None:
    """Mostra, abaixo da resposta, COMO ela foi produzida: rota, confiança, custo e fontes."""
    nivel, cor = servico.nivel_de_confianca(resposta.confidence)
    nota = f" ({resposta.confidence:.2f})" if resposta.confidence is not None else ""
    partes = [f"Confiança: :{cor}[**{nivel}**{nota}]"]
    if resposta.route:
        rotulo, explicacao = ROTULO_ROTA.get(resposta.route, (resposta.route, ""))
        partes.append(f"Rota: **{rotulo}**, {explicacao}")
    partes.append(f"Chamadas ao LLM: **{resposta.llm_calls}**")
    partes.append(f"Busca {resposta.retrieval_seconds:.1f}s · geração {resposta.generation_seconds:.1f}s")
    st.caption(" · ".join(partes))

    if resposta.sources:
        with st.expander(f"📚 Fontes ({len(resposta.sources)} trechos enviados ao LLM)"):
            for i, fonte in enumerate(resposta.sources, start=1):
                nota_fonte = f" · nota {fonte.score:.2f}" if fonte.score is not None else ""
                st.markdown(f"**[{i}] {fonte.file_name}, p. {fonte.page_number}**{nota_fonte}")
                st.caption(fonte.excerpt.replace("\n", " ") + " …")


def mensagem(papel: str, texto: str, resposta: Answer | None = None) -> None:
    with st.chat_message(papel, avatar="⚽" if papel == "assistant" else None):
        st.markdown(texto)
        if resposta is not None:
            painel_da_resposta(resposta)
