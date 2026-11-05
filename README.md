# ⚽ RAG Regras do Futebol: interface web

Interface em **Streamlit** para o [LangChainadvancedRAG](https://github.com/nascimentotnp/LangChainadvancedRAG): um chat que responde perguntas sobre as Regras do Jogo 2026/27 **citando a página de cada informação**. Ele também mostra **como** cada resposta foi produzida: rota, confiança, chamadas ao LLM e fontes.

## Por que um repositório separado?

O RAG é o **motor**: busca, reranking, roteamento, geração e avaliação. Esta interface é só uma **casca fina** que o usa como biblioteca:

```
rag-soccer-streamlit (este repo)              LangChainadvancedRAG (outro repo)
┌───────────────────────────────┐              ┌──────────────────────────────────┐
│ app.py, pages/   (telas)      │              │ RAGPipeline.ask()                │
│        │                      │   pip        │   busca híbrida + reranker       │
│        ▼                      │  install     │   recuperação adaptativa         │
│ rag_ui/servico.py ────────────┼─────────────►│   LLM + citação de fontes        │
│ (único arquivo que conhece    │              │ .env (chaves), PDFs, índice      │
│  o pacote `rag`)              │              │ rag avaliar -> relatórios        │
└───────────────────────────────┘              └──────────────────────────────────┘
```

- **Cada repositório tem uma responsabilidade.** Dá para trocar a interface (por uma API com FastAPI, um bot ou um app) sem tocar no RAG.
- **As chaves, os PDFs e o índice ficam em um lugar só** (o repositório do RAG). Nada é duplicado aqui.
- **A interface não acessa os módulos internos do RAG**, só o `RAGPipeline`. Se o RAG mudar por dentro, esta interface não quebra.

## O que a tela mostra

Uma experiência de assistente de IA, com tudo que acontece por trás **visível**:

- **Resposta sendo escrita** (streaming), em vez de aparecer de uma vez depois de segundos de tela parada.
- **"Pensando…" com as etapas reais**: 🛡️ verificação de segurança → 🔁 contexto da conversa → 🔎 busca nos documentos, com o tempo de cada uma (vem do *rastro* do RAG).
- **Chips de transparência** em cada resposta:
  - confiança (alta, média, baixa ou muito baixa);
  - rota (✅ confiante, 🤔 pediu confirmação, ⛔ fora do escopo, 🛡️ bloqueado);
  - modelo, com aviso **⚠️ quando o modelo reserva respondeu**;
  - custo em chamadas ao LLM e tempo total.
- **Fontes em cartões**: arquivo, página, **versão da regra** e nota do reranker. O usuário pode conferir.
- **"🔍 Como cheguei nesta resposta"**: a linha do tempo completa da pergunta.
- **Conversa com memória**: "e se ele passar disso?" é entendido pelo contexto ("🔁 Entendi como: …"). Na dúvida, o assistente **pergunta** em vez de chutar.
- **Histórico de conversas salvo** (barra lateral, como no ChatGPT). Recarregar a página **não apaga nada** (SQLite local em `dados/`, fora do git).
- **👍/👎 em cada resposta**: o feedback vai para o RAG, e os 👎 viram candidatas a perguntas novas no gabarito (`rag feedback --exportar`).
- **Página "Avaliação"**: os relatórios do RAG (juiz, segurança, busca com intervalo de confiança, rodadas, roteamento, reranking).
- **Tema escuro**, configurado em `.streamlit/config.toml`.

**Dois tipos de "memória" (não confunda numa entrevista):**
- **curto prazo**, no RAG: entender a pergunta de acompanhamento;
- **persistência**, aqui: guardar a conversa para reabrir depois.

## Como rodar

**Pré-requisito:** o repositório do RAG **na pasta ao lado**, já configurado (`.env` com a `GROQ_API_KEY`) e indexado (`rag indexar`).

```
langchain/
├── LangChainadvancedRAG/     ← o motor (já configurado e indexado)
└── rag-soccer-streamlit/    ← este repo
```

```powershell
cd rag-soccer-streamlit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt pytest     # instala o Streamlit + o RAG (modo editável, da pasta ao lado)
streamlit run app.py                        # abre em http://localhost:8501
```

A 1ª pergunta demora cerca de 30 s, porque os modelos de embeddings e o reranker são carregados. As seguintes levam de 1 a 3 s. Os modelos ficam em `@st.cache_resource`: são carregados **uma vez por servidor** e compartilhados entre os usuários.

**O RAG está em outro lugar ou foi instalado pelo git?** Copie `.env.example` para `.env` e preencha `RAG_HOME` com a pasta que tem o `.env` com as chaves, `data/documents` e `data/indexes`.

## Testes

```powershell
pytest -v
```

Os testes usam o `AppTest` do Streamlit, que **roda as telas sem navegador**, com um **pipeline falso** (embeddings, LLM e reranker de mentira). Eles levam segundos, não precisam de internet e não gastam cota do LLM. Cobrem:
- a tela inicial e os botões de exemplo;
- pergunta → resposta em streaming + chips + fontes;
- fora do escopo com 0 chamadas ao LLM, e ataque de prompt injection bloqueado;
- memória ("e se ele…?") e pedido de confirmação na dúvida;
- **a conversa sobrevive a recarregar a página** (histórico salvo e reaberto pela barra lateral);
- o histórico guarda e recupera a resposta completa (fontes, versão, rastro);
- o feedback chega ao RAG;
- erro do provedor (429) virando mensagem amigável, falta de índice e a página de avaliação.

O histórico dos testes vai para um SQLite **temporário** (`RAG_UI_DB`), e o pipeline falso **não chama** o Prompt Guard real.

## Estudar o código

Ordem de leitura (cada arquivo indica o **passo anterior** e o **próximo**):

1. `app.py`: a tela de chat e **o modelo de execução do Streamlit** (o script roda de novo a cada interação).
2. `rag_ui/servico.py`: a ponte com o RAG, o cache dos modelos, o streaming e os erros amigáveis.
3. `rag_ui/historico.py`: as conversas salvas (SQLite) e por que persistência ≠ memória.
4. `rag_ui/componentes.py`: a "cara de IA": chips, fontes em cartões e a linha do tempo do rastro.
5. `pages/1_Avaliacao.py`: os relatórios de avaliação.
6. `tests/test_app.py`: como testar telas sem navegador.

## Próximos passos
- [x] **Memória de conversa** e **confirmação na dúvida** ("Você está perguntando sobre impedimento?").
- [x] **Streaming**, etapas do "pensando…", histórico persistente e **feedback 👍/👎**.
- [ ] **Conversas por usuário** (login) e banco compartilhado (PostgreSQL) para vários usuários.
- [ ] **Chamar a API** do RAG em vez de importar o pacote: a interface fica leve, sem PyTorch.
- [ ] **Deploy** (Streamlit Community Cloud ou Docker). Exige resolver onde ficam o índice e os PDFs.

## Stack
Python · Streamlit · [rag](https://github.com/nascimentotnp/LangChainadvancedRAG) (LangChain, FAISS, BM25, cross-encoder, Groq) · pytest (AppTest)
