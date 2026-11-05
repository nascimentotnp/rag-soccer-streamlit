"""COMPONENTES - pedaços visuais reaproveitados pelas telas.

Separar "como desenhar" (aqui) de "o que perguntar ao RAG" (servico.py) deixa as telas curtas
e facilita mudar o visual sem tocar na lógica.

A "cara de IA" vem de 4 coisas (nenhuma muda a resposta, todas mudam a EXPERIÊNCIA):
  1. a resposta aparece sendo ESCRITA (streaming, no app.py);
  2. o "pensando..." mostra as ETAPAS reais (guarda, memória, busca) com os tempos do rastro;
  3. chips de transparência: confiança, rota, modelo (avisa quando foi o RESERVA), custo;
  4. fontes como cartões, com página, versão e nota: o usuário pode CONFERIR.

Todo texto que vem do RAG passa por html.escape antes de entrar em HTML (um trecho de PDF
ou uma pergunta não pode injetar HTML/JS na página).

⬅️ Passo anterior: rag_ui/historico.py
➡️ Próximo passo: pages/1_Avaliacao.py (os relatórios de avaliação do RAG)
"""
import html

import streamlit as st
from rag.pipeline import Answer

from rag_ui import servico

ROTULO_ROTA = {
    "confiante": ("✅", "confiante"),
    "incerto": ("🤔", "incerto"),
    "fora_do_escopo": ("⛔", "fora do escopo"),
    "bloqueado": ("🛡️", "bloqueado por segurança"),
}
ICONE_ETAPA = {"guarda": "🛡️", "memoria": "🔁", "busca": "🔎", "geracao": "✍️"}
NOME_ETAPA = {"guarda": "Verificação de segurança", "memoria": "Entendendo o contexto da conversa",
              "busca": "Busca nos documentos", "geracao": "Escrevendo a resposta"}
COR_CONFIANCA = {"green": "#22c55e", "blue": "#38bdf8", "orange": "#f59e0b", "red": "#ef4444", "gray": "#94a3b8"}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 820px;}
.hero {text-align: center; padding: 1.2rem 0 0.4rem;}
.hero h1 {font-size: 2.3rem; margin-bottom: .2rem;
  background: linear-gradient(90deg, #22c55e, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero p {opacity: .75; margin: 0;}
.chips {display: flex; flex-wrap: wrap; gap: .35rem; margin: .5rem 0 .2rem;}
.chip {font-size: .74rem; padding: .15rem .55rem; border-radius: 999px; border: 1px solid rgba(148,163,184,.35);
  background: rgba(148,163,184,.08); white-space: nowrap;}
.chip.alerta {border-color: #f59e0b; background: rgba(245,158,11,.12);}
.fonte {border: 1px solid rgba(148,163,184,.25); border-left: 3px solid #22c55e; border-radius: 8px;
  padding: .5rem .7rem; margin-bottom: .45rem; font-size: .86rem;}
.fonte .meta {opacity: .7; font-size: .75rem;}
.entendi {font-size: .8rem; opacity: .75; margin-bottom: .3rem;}
.etapa {font-size: .82rem; margin: .1rem 0;}
div[data-testid="stSidebar"] button {text-align: left;}
</style>
"""


def aplicar_estilo() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def cabecalho() -> None:
    st.markdown(
        '<div class="hero"><h1>⚽ Árbitro IA</h1>'
        "<p>Pergunte sobre as Regras do Jogo 2026/27. Cada resposta cita a página de onde veio.</p></div>",
        unsafe_allow_html=True,
    )


def _chip(texto: str, alerta: bool = False, cor: str | None = None) -> str:
    estilo = f' style="border-color:{cor}; color:{cor}"' if cor else ""
    return f'<span class="chip{" alerta" if alerta else ""}"{estilo}>{html.escape(texto)}</span>'


def chips(resposta: Answer) -> str:
    nivel, cor = servico.nivel_de_confianca(resposta.confidence)
    partes = []
    if resposta.confidence is not None:
        partes.append(_chip(f"confiança {nivel} ({resposta.confidence:.2f})", cor=COR_CONFIANCA[cor]))
    if resposta.needs_clarification:
        partes.append(_chip("🤔 pediu confirmação"))
    elif resposta.route:
        icone, nome = ROTULO_ROTA.get(resposta.route, ("", resposta.route))
        partes.append(_chip(f"{icone} {nome}", alerta=resposta.route == "bloqueado"))
    if resposta.model:
        modelo = resposta.model.split("/")[-1]
        partes.append(_chip(f"⚠️ modelo reserva: {modelo}" if resposta.used_fallback_model else f"🧠 {modelo}",
                            alerta=resposta.used_fallback_model))
    partes.append(_chip(f"💸 {resposta.llm_calls} chamada{'s' if resposta.llm_calls != 1 else ''} ao LLM"))
    total = resposta.retrieval_seconds + resposta.generation_seconds
    partes.append(_chip(f"⏱️ {total:.1f}s"))
    return f'<div class="chips">{"".join(partes)}</div>'


def linha_do_tempo(resposta: Answer) -> None:
    """As etapas reais da pergunta (o rastro do RAG), com tempos."""
    for etapa in resposta.trace:
        icone = ICONE_ETAPA.get(etapa["name"], "•")
        nome = NOME_ETAPA.get(etapa["name"], etapa["name"])
        detalhe = f" — {html.escape(etapa['detail'])}" if etapa.get("detail") else ""
        st.markdown(f'<div class="etapa">{icone} {nome}{detalhe} <b>({etapa["seconds"]:.2f}s)</b></div>',
                    unsafe_allow_html=True)


def painel_da_resposta(resposta: Answer) -> None:
    """Abaixo da resposta: como foi entendida, chips de transparência, fontes e rastro."""
    if resposta.needs_clarification:
        st.caption("Responda **sim** para confirmar, ou diga o que você quis perguntar.")
    st.markdown(chips(resposta), unsafe_allow_html=True)

    if resposta.sources and resposta.route != "bloqueado":
        titulo = "trechos mais próximos" if resposta.needs_clarification else "fontes usadas"
        with st.expander(f"📚 {len(resposta.sources)} {titulo}"):
            for i, f in enumerate(resposta.sources, start=1):
                nota = f" · nota {f.score:.2f}" if f.score is not None else ""
                versao = f" · versão {html.escape(f.version)}" if f.version else ""
                st.markdown(
                    f'<div class="fonte"><b>[{i}] {html.escape(f.file_name)}, p. {f.page_number}</b>'
                    f'<div class="meta">{versao.lstrip(" ·")}{nota}</div>'
                    f"{html.escape(f.excerpt.replace(chr(10), ' '))}…</div>",
                    unsafe_allow_html=True)
    if resposta.trace:
        with st.expander(f"🔍 Como cheguei nesta resposta (rastro {resposta.trace_id})"):
            linha_do_tempo(resposta)


def entendi_como(resposta: Answer) -> None:
    if resposta.standalone_question:
        st.markdown(f'<div class="entendi">🔁 Entendi como: <i>{html.escape(resposta.standalone_question)}</i></div>',
                    unsafe_allow_html=True)


def mensagem(papel: str, texto: str, resposta: Answer | None = None) -> None:
    with st.chat_message(papel, avatar="⚽" if papel == "assistant" else "🧑"):
        if resposta is not None:
            entendi_como(resposta)
        st.markdown(texto)
        if resposta is not None:
            painel_da_resposta(resposta)
