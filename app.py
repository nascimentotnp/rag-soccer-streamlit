"""TELA DE CHAT - ponto de entrada:  streamlit run app.py

Como o Streamlit funciona (o conceito mais importante para ler este arquivo):
o script RODA DE NOVO, de cima a baixo, a cada interação (clique, mensagem enviada...).
Por isso:
  - a conversa aberta fica em st.session_state (sobrevive às re-execuções, 1 por usuário);
  - as conversas ficam SALVAS em SQLite (rag_ui/historico.py): recarregar a página não apaga nada;
  - os modelos ficam em @st.cache_resource no servico.py (carregados 1 vez por servidor).

Conversa: a cada mensagem, as trocas ANTERIORES vão para o RAG (servico.historico_da_conversa).
É assim que ele entende "e se ele passar disso?" e o "sim" de uma confirmação
("Você está perguntando sobre impedimento?"). A interface só guarda e envia; quem entende é o RAG.

Fluxo de uma pergunta (o que dá a "cara de IA"):
  st.status "Pensando..."  -> mostra as etapas REAIS do rastro (segurança, contexto, busca)
  st.write_stream          -> a resposta aparece sendo escrita
  chips + fontes + rastro  -> transparência: confiança, rota, modelo, custo, de onde veio
  👍/👎                     -> feedback que vira pergunta nova no gabarito do RAG

📖 Ordem de leitura: app.py -> rag_ui/servico.py -> rag_ui/historico.py -> rag_ui/componentes.py
   -> pages/1_Avaliacao.py
➡️ Próximo passo: rag_ui/servico.py (a ponte com o RAG)
"""
import streamlit as st

from rag_ui import componentes, historico, servico

st.set_page_config(page_title="Árbitro IA · Regras do Futebol", page_icon="⚽", layout="centered")
componentes.aplicar_estilo()

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
    st.session_state.conversa_id = None


def _abrir_conversa(conversa_id: str | None) -> None:
    st.session_state.conversa_id = conversa_id
    st.session_state.mensagens = historico.carregar_mensagens(conversa_id) if conversa_id else []


def _ao_avaliar(mensagem: dict, chave: str) -> None:
    """Callback do 👍/👎: salva na conversa e envia ao RAG (feedback -> candidatas ao gabarito)."""
    valor = st.session_state.get(chave)
    if valor is None:
        return
    util = valor == 1
    mensagem["feedback"] = 1 if util else 0
    if mensagem.get("id"):
        historico.salvar_feedback(mensagem["id"], util)
    servico.registrar_feedback(mensagem["resposta"], util)


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.markdown("### ⚽ Árbitro IA")
    if st.button("＋ Nova conversa", use_container_width=True, type="primary"):
        _abrir_conversa(None)
        st.rerun()

    conversas = historico.listar_conversas()
    if conversas:
        st.caption("CONVERSAS")
        for c in conversas:
            atual = c["id"] == st.session_state.conversa_id
            if st.button(("▸ " if atual else "") + c["titulo"], key=f"conv_{c['id']}", use_container_width=True):
                _abrir_conversa(c["id"])
                st.rerun()

    st.divider()
    with st.expander("⚙️ Configurações"):
        tecnica = st.selectbox("Técnica de busca", options=list(servico.TECNICAS),
                               format_func=lambda t: servico.TECNICAS[t][0])
        st.caption(servico.TECNICAS[tecnica][1])
        rerank = st.toggle("Reranking (cross-encoder)", value=True, disabled=tecnica == "adaptive",
                           help="Reordena os trechos com um modelo mais preciso. A técnica adaptativa exige.")
        if st.session_state.conversa_id and st.button("🗑️ Apagar esta conversa", use_container_width=True):
            historico.apagar_conversa(st.session_state.conversa_id)
            _abrir_conversa(None)
            st.rerun()

    cfg = servico.configuracao()
    with st.expander("ℹ️ Sobre este RAG"):
        st.markdown(
            f"- **Documentos:** {cfg.docs_domain}\n"
            f"- **LLM:** `{cfg.llm_model}` (reserva: `{cfg.llm_fallback_model or '—'}`)\n"
            f"- **Embeddings:** `{cfg.hf_embeddings_model}`\n"
            f"- **Reranker:** `{cfg.rerank_model}`\n"
            f"- **Segurança:** heurística + `{cfg.guard_model or 'sem classificador'}`\n"
            f"- **Dados em:** `{servico.rag_home()}`"
        )

# ------------------------------------------------------------------ índice
if not servico.indice_pronto():
    st.error("O índice ainda não foi criado. No repositório do RAG, rode `rag indexar` e recarregue a página.")
    st.stop()

# ------------------------------------------------------------------ tela inicial (sem mensagens)
pergunta_exemplo = None
if not st.session_state.mensagens:
    componentes.cabecalho()
    st.write("")
    colunas = st.columns(2)
    for i, exemplo in enumerate(servico.PERGUNTAS_EXEMPLO):
        if colunas[i % 2].button(f"💬 {exemplo}", key=f"exemplo_{i}", use_container_width=True):
            pergunta_exemplo = exemplo

# ------------------------------------------------------------------ conversa (redesenhada a cada execução)
for i, m in enumerate(st.session_state.mensagens):
    componentes.mensagem(m["papel"], m["texto"], m.get("resposta"))
    if m["papel"] == "assistant" and m.get("resposta") is not None:
        if m.get("feedback") is None:
            chave = f"fb_{m.get('id', i)}"
            st.feedback("thumbs", key=chave, on_change=_ao_avaliar, args=(m, chave))
        else:
            st.caption("Obrigado pelo feedback! " + ("👍" if m["feedback"] else "👎"))

# ------------------------------------------------------------------ nova pergunta
pergunta = st.chat_input("Pergunte sobre as regras do futebol...") or pergunta_exemplo
if pergunta:
    turnos = servico.historico_da_conversa(st.session_state.mensagens)  # trocas ANTERIORES (memória)
    if st.session_state.conversa_id is None:
        st.session_state.conversa_id = historico.nova_conversa(pergunta)
    conversa_id = st.session_state.conversa_id

    msg_id = historico.salvar_mensagem(conversa_id, "user", pergunta)
    st.session_state.mensagens.append({"papel": "user", "texto": pergunta, "id": msg_id})
    componentes.mensagem("user", pergunta)

    with st.chat_message("assistant", avatar="⚽"):
        try:
            with st.status("Pensando...", expanded=False) as status:
                meta, pedacos = servico.perguntar_stream(pergunta, tecnica, rerank, turnos)
                componentes.linha_do_tempo(meta)  # etapas reais: segurança, contexto, busca
                status.update(label=f"Pronto: busca em {meta.retrieval_seconds:.1f}s", state="complete")
            componentes.entendi_como(meta)
            escrito = st.write_stream(pedacos)  # a resposta aparece sendo escrita
            if meta.text and meta.text != (escrito or "").strip():
                # a guarda de SAÍDA trocou o texto no fim do stream (vazamento do prompt): mostra o certo
                st.warning(meta.text)
            componentes.painel_da_resposta(meta)
        except Exception as erro:  # erro de provedor não pode derrubar a tela
            texto = servico.mensagem_de_erro(erro)
            st.error(texto)
            print(f"[rag-ui] {type(erro).__name__}: {erro}")  # detalhe só no terminal
            msg_id = historico.salvar_mensagem(conversa_id, "assistant", f"⚠️ {texto}")
            st.session_state.mensagens.append({"papel": "assistant", "texto": f"⚠️ {texto}", "id": msg_id})
        else:
            msg_id = historico.salvar_mensagem(conversa_id, "assistant", meta.text, meta)
            st.session_state.mensagens.append(
                {"papel": "assistant", "texto": meta.text, "resposta": meta, "id": msg_id, "feedback": None})
            st.rerun()  # redesenha com o 👍/👎 e a conversa na barra lateral
