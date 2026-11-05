import streamlit as st

from rag_ui import componentes, servico

st.set_page_config(page_title="RAG Regras do Futebol", page_icon="⚽", layout="centered")

# ------------------------------------------------------------------ barra lateral: configuração
with st.sidebar:
    st.header("⚙️ Configuração")
    tecnica = st.selectbox(
        "Técnica de busca",
        options=list(servico.TECNICAS),
        format_func=lambda t: servico.TECNICAS[t][0],
    )
    st.caption(servico.TECNICAS[tecnica][1])

    adaptativa = tecnica == "adaptive"
    rerank = st.toggle(
        "Reranking (cross-encoder)", value=True, disabled=adaptativa,
        help="Reordena os trechos com um modelo mais preciso. A técnica adaptativa exige o reranking.",
    )

    st.divider()
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.mensagens = []
        st.rerun()

    cfg = servico.configuracao()
    with st.expander("Sobre este RAG"):
        st.markdown(
            f"- **Documentos:** {cfg.docs_domain}\n"
            f"- **LLM:** `{cfg.llm_provider} / {cfg.llm_model}`\n"
            f"- **Embeddings:** `{cfg.hf_embeddings_model}`\n"
            f"- **Reranker:** `{cfg.rerank_model}`\n"
            f"- **Dados em:** `{servico.rag_home()}`"
        )

# ------------------------------------------------------------------ cabeçalho
st.title("⚽ Pergunte sobre as Regras do Futebol")
st.caption("Respostas geradas **somente** a partir das Regras do Jogo 2026/27 (IFAB/CBF), com a página "
           "de cada informação. Se não estiver nos documentos, o assistente diz que não encontrou.")

if not servico.indice_pronto():
    st.error("O índice ainda não foi criado. No repositório do RAG, rode `rag indexar` e recarregue a página.")
    st.stop()

# ------------------------------------------------------------------ histórico (re-desenhado a cada execução)
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for m in st.session_state.mensagens:
    componentes.mensagem(m["papel"], m["texto"], m.get("resposta"))

# ------------------------------------------------------------------ perguntas de exemplo (só com o chat vazio)
pergunta_exemplo = None
if not st.session_state.mensagens:
    st.markdown("**Experimente:**")
    colunas = st.columns(2)
    for i, exemplo in enumerate(servico.PERGUNTAS_EXEMPLO):
        if colunas[i % 2].button(exemplo, key=f"exemplo_{i}", use_container_width=True):
            pergunta_exemplo = exemplo

# ------------------------------------------------------------------ nova pergunta
pergunta = st.chat_input("Faça uma pergunta sobre as regras...") or pergunta_exemplo
if pergunta:
    st.session_state.mensagens.append({"papel": "user", "texto": pergunta})
    componentes.mensagem("user", pergunta)

    with st.chat_message("assistant", avatar="⚽"):
        try:
            with st.spinner("Buscando nos documentos e gerando a resposta..."):
                resposta = servico.perguntar(pergunta, tecnica, rerank)
        except Exception as erro:  # erro de provedor não pode derrubar a tela
            texto = servico.mensagem_de_erro(erro)
            st.error(texto)
            print(f"[rag-ui] {type(erro).__name__}: {erro}")  # detalhe só no terminal
            st.session_state.mensagens.append({"papel": "assistant", "texto": f"⚠️ {texto}"})
        else:
            st.markdown(resposta.text)
            componentes.painel_da_resposta(resposta)
            st.session_state.mensagens.append(
                {"papel": "assistant", "texto": resposta.text, "resposta": resposta})
