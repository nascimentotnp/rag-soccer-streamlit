# ⚽ RAG Regras do Futebol: interface web

Interface em **Streamlit** para o [LangChainadvancedRAG](https://github.com/nascimentotnp/LangChainadvancedRAG): um chat que responde perguntas sobre as Regras do Jogo 2026/27 **citando a página de cada informação**. Ele também mostra **como** cada resposta foi produzida: rota, confiança, chamadas ao LLM e fontes.

## Por que um repositório separado?

O RAG é o **motor**: busca, reranking, roteamento, geração e avaliação. Esta interface é só uma **casca fina** que o usa como biblioteca:

```
rag-futebol-streamlit (este repo)              LangChainadvancedRAG (outro repo)
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

- **Chat** com a resposta e as **fontes** (arquivo, página, trecho e nota do reranker).
- **Painel de transparência** abaixo de cada resposta:
  - **Confiança:** alta, baixa ou muito baixa, pela nota do reranker.
  - **Rota** (técnica adaptativa): ✅ confiante, 🤔 incerto ou ⛔ fora do escopo.
  - **Chamadas ao LLM:** o custo daquela pergunta. Fora do escopo = **0**.
  - **Tempo** de busca e de geração.
- **Barra lateral:** escolha da técnica de busca e liga/desliga o reranking, para comparar ao vivo.
- **Página "Avaliação":** os relatórios do RAG (antes × depois do reranking, roteamento, técnicas).

## Como rodar

**Pré-requisito:** o repositório do RAG **na pasta ao lado**, já configurado (`.env` com a `GROQ_API_KEY`) e indexado (`rag indexar`).

```
langchain/
├── LangChainadvancedRAG/     ← o motor (já configurado e indexado)
└── rag-futebol-streamlit/    ← este repo
```

```powershell
cd rag-futebol-streamlit
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
- a tela inicial;
- pergunta → resposta + fontes + custo;
- fora do escopo com 0 chamadas ao LLM;
- os botões de exemplo;
- erro do provedor (429) virando mensagem amigável;
- a falta de índice;
- a página de avaliação.

## Estudar o código

Ordem de leitura (cada arquivo indica o **passo anterior** e o **próximo**):

1. `app.py`: a tela de chat e **o modelo de execução do Streamlit** (o script roda de novo a cada interação).
2. `rag_ui/servico.py`: a ponte com o RAG, o cache dos modelos e os erros amigáveis.
3. `rag_ui/componentes.py`: o painel de transparência e as fontes.
4. `pages/1_Avaliacao.py`: os relatórios de avaliação.
5. `tests/test_app.py`: como testar telas sem navegador.

## Próximos passos
- [ ] **Memória de conversa:** o histórico já aparece na tela, mas cada pergunta ainda é independente no RAG.
- [ ] **Streaming:** mostrar a resposta palavra por palavra.
- [ ] **Feedback 👍/👎** por resposta, salvo para virar pergunta nova no gabarito.
- [ ] **Deploy** (Streamlit Community Cloud ou Docker). Exige resolver onde ficam o índice e os PDFs.

## Stack
Python · Streamlit · [rag](https://github.com/nascimentotnp/LangChainadvancedRAG) (LangChain, FAISS, BM25, cross-encoder, Groq) · pytest (AppTest)
