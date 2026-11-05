"""HISTÓRICO PERSISTENTE - as conversas não somem quando a página recarrega.

Dois tipos de "memória", que é bom não confundir numa entrevista:
- memória de CURTO prazo (no RAG): o "e se ele passar disso?" entende as últimas trocas;
- PERSISTÊNCIA (aqui): a conversa fica salva e pode ser reaberta depois, como no ChatGPT.

Antes, o chat vivia só em st.session_state: recarregou a página, perdeu tudo.
Agora cada mensagem vai para um SQLite local (dados/conversas.db). SQLite porque é um arquivo só,
vem com o Python e aguenta bem um usuário. Com muitos usuários simultâneos, troque por
PostgreSQL e separe as conversas POR USUÁRIO (hoje todas ficam juntas: é uma interface local).

Privacidade: o banco guarda as perguntas dos usuários -> está no .gitignore.

⬅️ Passo anterior: rag_ui/servico.py
➡️ Próximo passo: rag_ui/componentes.py (como a resposta é desenhada)
"""
import json
import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rag.pipeline import Answer, Source

PADRAO = Path(__file__).resolve().parents[1] / "dados" / "conversas.db"


def _caminho() -> Path:
    return Path(os.getenv("RAG_UI_DB") or PADRAO)  # RAG_UI_DB: os testes usam um banco temporário


def _conectar() -> sqlite3.Connection:
    caminho = _caminho()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(caminho)
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE IF NOT EXISTS conversas (
            id TEXT PRIMARY KEY, titulo TEXT NOT NULL, criada_em TEXT NOT NULL, atualizada_em TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS mensagens (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conversa_id TEXT NOT NULL REFERENCES conversas(id),
            papel TEXT NOT NULL, texto TEXT NOT NULL, resposta_json TEXT, feedback INTEGER, criada_em TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS ix_mensagens_conversa ON mensagens(conversa_id, id);
    """)
    return con


def _agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def nova_conversa(primeira_pergunta: str) -> str:
    conversa_id = uuid.uuid4().hex[:12]
    titulo = primeira_pergunta.strip()[:60] + ("…" if len(primeira_pergunta.strip()) > 60 else "")
    with closing(_conectar()) as con, con:
        con.execute("INSERT INTO conversas VALUES (?, ?, ?, ?)", (conversa_id, titulo, _agora(), _agora()))
    return conversa_id


def salvar_mensagem(conversa_id: str, papel: str, texto: str, resposta: Answer | None = None) -> int:
    resposta_json = json.dumps(asdict(resposta), ensure_ascii=False) if resposta is not None else None
    with closing(_conectar()) as con, con:
        cur = con.execute(
            "INSERT INTO mensagens (conversa_id, papel, texto, resposta_json, criada_em) VALUES (?, ?, ?, ?, ?)",
            (conversa_id, papel, texto, resposta_json, _agora()))
        con.execute("UPDATE conversas SET atualizada_em = ? WHERE id = ?", (_agora(), conversa_id))
        return cur.lastrowid


def salvar_feedback(mensagem_id: int, util: bool) -> None:
    with closing(_conectar()) as con, con:
        con.execute("UPDATE mensagens SET feedback = ? WHERE id = ?", (1 if util else 0, mensagem_id))


def _resposta_de(texto_json: str | None) -> Answer | None:
    if not texto_json:
        return None
    d = json.loads(texto_json)
    d["sources"] = [Source(**s) for s in d.get("sources", [])]
    return Answer(**d)


def carregar_mensagens(conversa_id: str) -> list[dict]:
    """No mesmo formato do st.session_state.mensagens (o app desenha igual, venha de onde vier)."""
    with closing(_conectar()) as con:
        linhas = con.execute("SELECT * FROM mensagens WHERE conversa_id = ? ORDER BY id", (conversa_id,)).fetchall()
    mensagens = []
    for linha in linhas:
        m = {"papel": linha["papel"], "texto": linha["texto"], "id": linha["id"], "feedback": linha["feedback"]}
        resposta = _resposta_de(linha["resposta_json"])
        if resposta is not None:
            m["resposta"] = resposta
        mensagens.append(m)
    return mensagens


def listar_conversas(limite: int = 30) -> list[dict]:
    with closing(_conectar()) as con:
        linhas = con.execute("SELECT id, titulo, atualizada_em FROM conversas ORDER BY atualizada_em DESC LIMIT ?",
                             (limite,)).fetchall()
    return [dict(linha) for linha in linhas]


def apagar_conversa(conversa_id: str) -> None:
    with closing(_conectar()) as con, con:
        con.execute("DELETE FROM mensagens WHERE conversa_id = ?", (conversa_id,))
        con.execute("DELETE FROM conversas WHERE id = ?", (conversa_id,))
