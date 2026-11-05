"""SERVIÇO - o único arquivo da interface que conhece o pacote `rag`.

Regra (a mesma do repositório do RAG): a interface só fala com o RAGPipeline.
Telas (app.py, pages/) -> servico.py -> rag.pipeline.RAGPipeline

Por que um arquivo só para isso:
- se o RAG mudar de API, só este arquivo muda;
- os testes trocam `obter_pipeline` por um pipeline falso (sem modelos, sem internet).

⬅️ Passo anterior: app.py (a tela de chat)
➡️ Próximo passo: rag_ui/componentes.py (como a resposta é desenhada)
"""
import os
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
    # embeddings são o componente pesado (PyTorch): compartilha. O cliente do LLM é leve: cria outro.
    return RAGPipeline(replace(base.cfg, rerank=False), embeddings=base.embeddings)


def perguntar(pergunta: str, tecnica: str, rerank: bool) -> Answer:
    if tecnica == ADAPTIVE:
        rerank = True  # a rota adaptativa usa a nota do reranker
    return obter_pipeline(rerank).ask(pergunta, tecnica)


def nivel_de_confianca(nota: float | None) -> tuple[str, str]:
    """Traduz a nota do reranker para o usuário, com os MESMOS limites da rota adaptativa."""
    if nota is None:
        return "—", "gray"
    if nota >= settings.adaptive_uncertain_below:
        return "alta", "green"
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
