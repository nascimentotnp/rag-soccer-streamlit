"""SERVIÇO - o único arquivo da interface que conhece o pacote `rag`.

Regra (a mesma do repositório do RAG): a interface só fala com o RAGPipeline.
Telas (app.py, pages/) -> servico.py -> rag.pipeline.RAGPipeline

Por que um arquivo só para isso:
- se o RAG mudar de API, só este arquivo muda;
- os testes trocam `obter_pipeline` por um pipeline falso (sem modelos, sem internet).

O que mora aqui:
- cache dos modelos (@st.cache_resource): carregados 1 vez por servidor, compartilhados entre usuários;
- perguntar_stream(): a resposta em pedaços (streaming), para aparecer sendo escrita;
- historico_da_conversa(): converte o chat da tela em [(pergunta, resposta)] para a memória do RAG;
- registrar_feedback(): 👍/👎 vão para o RAG e viram candidatas ao gabarito;
- mensagem_de_erro(): erros do provedor viram texto amigável (sem expor detalhes internos).

⬅️ Passo anterior: app.py (a tela de chat)
➡️ Próximo passo: rag_ui/historico.py (as conversas salvas, que não somem ao recarregar)
"""
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import streamlit as st
from rag.config import Settings, settings  # o .env deste repo já foi lido em rag_ui/__init__.py
from rag.pipeline import ADAPTIVE, Answer, RAGPipeline
from rag.query_transform import TECHNIQUES
from rag.vectorstore import index_exists

# técnica -> (nome amigável, quando usar). A ordem é a do menu.
TECNICAS: dict[str, tuple[str, str]] = {
    ADAPTIVE: ("Adaptativa (recomendada)",
               "Busca simples + reranker; chama o LLM extra só quando a confiança é baixa, "
               "e recusa perguntas fora do escopo sem gastar LLM."),
    "none": ("Busca híbrida", "Vetor + BM25 (+ reranker). Sem chamadas extras ao LLM."),
    "multi_query": ("Multi-query (RAG-Fusion)", "O LLM gera 4 versões da pergunta; bom para gírias."),
    "rewrite": ("Reescrita", "O LLM reescreve a pergunta com termos oficiais."),
    "step_back": ("Step-back", "Busca também por uma versão mais genérica da pergunta."),
    "hyde": ("HyDE", "O LLM escreve uma resposta hipotética e busca por ela."),
}
assert set(TECNICAS) == {*TECHNIQUES, ADAPTIVE}, "menu desatualizado em relação às técnicas do RAG"

PERGUNTAS_EXEMPLO = [
    "Quantas substituições cada equipe pode fazer?",
    "Por quanto tempo o goleiro pode segurar a bola?",
    "o atacante tava na banheira, vale o gol?",
    "Quem ganhou a Copa do Mundo de 2022?",
]


def configuracao() -> Settings:
    return settings


def indice_pronto() -> bool:
    return index_exists(settings.index_dir)


@st.cache_resource(show_spinner="Carregando modelos e índice (só na 1ª vez, ~30 s)...")
def _pipeline_base() -> RAGPipeline:
    """Carregado UMA vez por servidor (cache_resource) e compartilhado entre os usuários."""
    p = RAGPipeline(replace(settings, rerank=True))
    p.warmup()
    return p


@st.cache_resource(show_spinner=False)
def obter_pipeline(rerank: bool) -> RAGPipeline:
    """Um pipeline por configuração, reaproveitando os modelos já carregados (nada é baixado 2x)."""
    base = _pipeline_base()
    if rerank:
        return base
    return base.variant(rerank=False)  # reaproveita os modelos e o índice já carregados


def historico_da_conversa(mensagens: list[dict]) -> list[tuple[str, str]]:
    """Converte o chat da tela em [(pergunta, resposta), ...] para a memória do RAG.

    Só entram trocas completas: mensagens de erro (sem "resposta") ficam de fora.
    """
    turnos, pergunta = [], None
    for m in mensagens:
        if m["papel"] == "user":
            pergunta = m["texto"]
        elif pergunta is not None and "resposta" in m:
            turnos.append((pergunta, m["texto"]))
            pergunta = None
    return turnos


def perguntar(pergunta: str, tecnica: str, rerank: bool,
              historico: list[tuple[str, str]] | None = None) -> Answer:
    if tecnica == ADAPTIVE:
        rerank = True  # a rota adaptativa usa a nota do reranker
    return obter_pipeline(rerank).ask(pergunta, tecnica, history=historico)


def perguntar_stream(pergunta: str, tecnica: str, rerank: bool,
                     historico: list[tuple[str, str]] | None = None) -> tuple[Answer, Iterator[str]]:
    """Streaming: (metadados já prontos - fontes, rota, rastro da busca -, pedaços do texto).

    É o que dá a "cara de IA": a resposta aparece sendo escrita, em vez de surgir de uma vez
    depois de alguns segundos de tela parada.
    """
    if tecnica == ADAPTIVE:
        rerank = True
    return obter_pipeline(rerank).ask_stream(pergunta, tecnica, history=historico)


def registrar_feedback(resposta: Answer, util: bool, comentario: str = "") -> None:
    """👍/👎 vão para o RAG (data/feedback/feedback.jsonl): os 👎 viram candidatas ao gabarito."""
    obter_pipeline(True).record_feedback(resposta, useful=util, comment=comentario)


def nivel_de_confianca(nota: float | None) -> tuple[str, str]:
    """Traduz a nota do reranker para o usuário, alinhada aos limites da rota adaptativa."""
    if nota is None:
        return "—", "gray"
    if nota >= 0:
        return "alta", "green"
    if nota >= settings.adaptive_uncertain_below:
        return "média", "blue"
    if nota >= settings.adaptive_out_of_scope_below:
        return "baixa", "orange"
    return "muito baixa", "red"


def mensagem_de_erro(erro: Exception) -> str:
    """Erros comuns em linguagem de gente (sem expor detalhes internos ou chaves)."""
    texto = str(erro)
    if "429" in texto or "rate limit" in texto.lower():
        return ("O provedor do LLM atingiu o limite de uso (plano gratuito do Groq). "
                "Tente de novo em alguns minutos. Perguntas fora do escopo continuam funcionando "
                "na técnica adaptativa, porque não chamam o LLM.")
    if "api_key" in texto.lower() or "401" in texto:
        return "Chave do LLM ausente ou inválida: confira o GROQ_API_KEY no .env do repositório do RAG."
    return f"Erro inesperado ({type(erro).__name__}). Veja o terminal para detalhes."


def pasta_de_relatorios() -> Path:
    return settings.eval_file.parent


def rag_home() -> str:
    return os.getenv("RAG_HOME") or str(Path(settings.docs_dir).parents[1])
