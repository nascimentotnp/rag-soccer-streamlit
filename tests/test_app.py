"""Testes das telas com o AppTest do Streamlit: rodam o app SEM navegador.

O RAG de verdade (modelos, índice, Groq) é trocado por um pipeline FALSO: os testes rodam
em segundos, sem internet e sem gastar cota do LLM.
"""
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from rag.config import settings
from rag.generation import NOT_FOUND_MESSAGE
from rag.pipeline import RAGPipeline
from streamlit.testing.v1 import AppTest

from rag_ui import servico

RAIZ = Path(__file__).resolve().parents[1]  # o AppTest resolve caminhos a partir deste arquivo

PAGINAS = [
    Document(page_content="Regra 12. O goleiro não pode controlar a bola com as mãos por mais de oito segundos; "
                 "se isso acontecer, o árbitro concede um tiro de canto à equipe adversária.",
             metadata={"source": "data/regras.pdf", "page": 125}),
    Document(page_content="Regra 3. Cada equipe pode fazer no máximo cinco substituições em competições oficiais, "
                 "em até três oportunidades durante a partida, além do intervalo.",
             metadata={"source": "data/regras.pdf", "page": 60}),
]


class ScorerFixo:
    """Reranker falso: a mesma nota para todos os trechos (controla a rota adaptativa)."""

    def __init__(self, nota):
        self.nota = nota

    def predict(self, pairs, **kwargs):
        return [self.nota] * len(pairs)


@pytest.fixture
def pipeline_falso(tmp_path, monkeypatch):
    """Instala no servico um pipeline falso; devolve uma função para escolher a nota do reranker."""

    def _instalar(nota=5.0):
        cfg = replace(settings, index_dir=tmp_path / "index", rerank=True, retriever_k=2,
                      rerank_candidates=2, max_context_docs=2,
                      adaptive_uncertain_below=0, adaptive_out_of_scope_below=-3)
        p = RAGPipeline(cfg, embeddings=DeterministicFakeEmbedding(size=16),
                        llm=FakeListChatModel(responses=["O goleiro tem oito segundos [1]."]),
                        reranker=ScorerFixo(nota))
        p.build(documents=PAGINAS)
        monkeypatch.setattr(servico, "obter_pipeline", lambda rerank: p)
        monkeypatch.setattr(servico, "indice_pronto", lambda: True)
        return p

    return _instalar


def _app() -> AppTest:
    return AppTest.from_file(str(RAIZ / "app.py"), default_timeout=30)


def test_tela_inicial(pipeline_falso):
    pipeline_falso()
    at = _app().run()
    assert not at.exception
    assert "Regras do Futebol" in at.title[0].value
    assert len(at.button) >= len(servico.PERGUNTAS_EXEMPLO)  # botões de exemplo


def test_pergunta_mostra_resposta_fontes_e_custo(pipeline_falso):
    pipeline_falso(nota=5.0)
    at = _app().run()
    at.chat_input[0].set_value("Quanto tempo o goleiro pode segurar a bola?").run()
    assert not at.exception
    assert any("oito segundos" in md.value for md in at.markdown)
    legenda = " ".join(c.value for c in at.caption)
    assert "Chamadas ao LLM" in legenda and "confiante" in legenda
    assert len(at.session_state.mensagens) == 2  # pergunta + resposta guardadas no histórico


def test_fora_do_escopo_nao_chama_llm(pipeline_falso):
    pipeline_falso(nota=-8.0)  # nota muito baixa -> rota "fora do escopo"
    at = _app().run()
    at.chat_input[0].set_value("Como fazer um bolo?").run()
    assert any(NOT_FOUND_MESSAGE in md.value for md in at.markdown)
    legenda = " ".join(c.value for c in at.caption)
    assert "fora do escopo" in legenda and "Chamadas ao LLM: **0**" in legenda


def test_botao_de_exemplo_faz_a_pergunta(pipeline_falso):
    pipeline_falso()
    at = _app().run()
    at.button(key="exemplo_0").click().run()
    assert at.session_state.mensagens[0]["texto"] == servico.PERGUNTAS_EXEMPLO[0]


def test_erro_do_provedor_vira_mensagem_amigavel(pipeline_falso, monkeypatch):
    pipeline_falso()

    def _estourou(*args, **kwargs):
        raise RuntimeError("Error code: 429 - rate limit")

    monkeypatch.setattr(servico, "perguntar", _estourou)
    at = _app().run()
    at.chat_input[0].set_value("qualquer").run()
    assert not at.exception  # a tela não quebra
    assert "limite de uso" in at.error[0].value


def test_sem_indice_orienta_o_usuario(monkeypatch):
    monkeypatch.setattr(servico, "indice_pronto", lambda: False)
    at = _app().run()
    assert "rag indexar" in at.error[0].value


def test_pagina_de_avaliacao():
    at = AppTest.from_file(str(RAIZ / "pages" / "1_Avaliacao.py"), default_timeout=30).run()
    assert not at.exception
    assert len(at.tabs) == 3


@pytest.mark.parametrize("nota, nivel", [(5, "alta"), (-1, "baixa"), (-8, "muito baixa"), (None, "—")])
def test_nivel_de_confianca(nota, nivel):
    assert servico.nivel_de_confianca(nota)[0] == nivel
