# Startup Idea Validator

**Agentic RAG system that checks a startup idea against 6,000+ Y Combinator companies and the live web, running on a self-hosted local LLM.**

Describe an idea in one or two sentences. A LangGraph agent decides which tools to call, retrieves semantically similar YC startups from a vector database, searches the web for current competitors, and returns a structured, source-linked competitor report.

---

## What it does

- **Semantic search over Y Combinator:** dense retrieval over the public yc-oss dataset (6,286 companies), followed by cross-encoder reranking and a relevance threshold.
- **Live web search:** current competitors, reviews and product pages via Tavily, with automatic DuckDuckGo fallback.
- **Agentic tool use:** a local LLM decides which tools to call and how to phrase each query.
- **Grounded report:** tables, links and competitor lists are built deterministically from tool outputs; every competitor named by the model is verified against retrieved data.
- **Fully self-hosted inference:** Qwen 2.5 3B served by Ollama on a CPU-only VPS, no paid LLM APIs.

## How it works

```
User idea
   |
   v
LangGraph agent (Qwen 2.5 3B via Ollama)
   |  decides which tools to call
   +--> database_search: Qdrant dense retrieval (top-15)
   |        -> cross-encoder rerank -> top-5, min relevance 10%
   +--> web_search: Tavily (5 results) -> DuckDuckGo fallback
   |
   v
LLM analysis (no tools bound, task reminder at end of context)
   |
   v
Deterministic report assembly + competitor grounding (Gradio UI)
```

1. **Idea input.** Free text or one of 10 built-in examples.
2. **Tool selection.** The agent node (LLM with tools bound) emits tool calls. Duplicate calls are filtered in code, so each tool runs exactly once per run.
3. **Database search.** The query is embedded with `bge-small-en-v1.5` (with the model's recommended query instruction prefix), 15 nearest neighbours are retrieved from Qdrant, rescored by a cross-encoder, mapped to 0-100% with a sigmoid, filtered at 10% and truncated to the top 5.
4. **Web search.** Tavily `basic` search, 5 results, page content truncated to 500 characters per result to fit the small model's context.
5. **Analysis.** Once both tools have returned, a tool-free model instance writes the analysis: main competitors, three differentiation points, verdict. The idea text and output format are repeated at the end of the context, where small models attend best.
6. **Report assembly.** Code builds the final Markdown report from structured tool artifacts and the model's analysis.

## Report structure

| Section | Content | Source |
|---|---|---|
| 🔎 Found | Number of similar YC startups by status (active / acquired / inactive) and number of web competitors | Code |
| 🏢 Similar Y Combinator startups | Name (YC link), one-liner, Match %, status, batch | Code |
| 🌐 From the web | 5 linked web results | Code |
| ⚔️ Main competitors | Grouped by source: YC (match % + description) and web (link to the article where the name was found) | Code + verified model output |
| 💡 How to stand out | 3 differentiation points for the new idea | LLM |
| ✅ Verdict | One-sentence summary | LLM |
| 🔎 Agent steps | Tool calls, queries and result sizes | Code |

While the agent runs, the UI streams progress (`Step 1/3` tool selection, `Step 2/3` search with the tools the agent actually chose, `Step 3/3` analysis).

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (`StateGraph`, `MessagesState`, `ToolNode`, `tools_condition`), LangChain tools |
| LLM | Qwen 2.5 3B via Ollama (Docker), `temperature=0`, `num_ctx=8192` |
| Vector database | Qdrant Cloud (EU), collection of 6,236 vectors |
| Indexing / retrieval | LlamaIndex (`VectorStoreIndex`, `QdrantVectorStore`) |
| Embeddings | FastEmbed, `BAAI/bge-small-en-v1.5` (384 dimensions), CPU |
| Reranking | FastEmbed `TextCrossEncoder`, `Xenova/ms-marco-MiniLM-L-6-v2` (~80 MB), CPU |
| Web search | Tavily API, DuckDuckGo (`ddgs`) as fallback |
| UI | Gradio Blocks, generator-based streaming, basic auth |
| Observability | Langfuse (EU cloud), LangChain `CallbackHandler` |
| Infrastructure | Ubuntu 24.04 VPS (7.8 GB RAM, CPU only), Python 3.12 venv |

## Data pipeline

- **Source:** public yc-oss dataset of Y Combinator companies (6,286 records).
- **Cleaning:** 54 companies without any description are skipped, leaving 6,232 documents.
- **Embedded text:** name, one-liner, long description, industry and tags.
- **Metadata (stored, not embedded):** name, one-liner, website, industry, batch, status, regions, YC URL.
- **Ingestion:** LlamaIndex chunks and embeds documents with FastEmbed and upserts them into Qdrant; a few long descriptions are split into multiple chunks, giving 6,236 vectors. Re-ingestion recreates the collection to avoid duplicates.

## Key design decisions

**1. Cross-encoder reranking on top of dense retrieval.**
Bi-encoder similarity scores were nearly flat (70-76%) and let unrelated companies into the top 5. Retrieving 15 candidates and rescoring each (idea, company) pair with a cross-encoder separates direct competitors from noise; a 10% threshold removes irrelevant results before they reach the model.

**2. Structured tool artifacts, deterministic report.**
Tools use `response_format="content_and_artifact"`: the LLM receives compact text, while the app receives structured data (lists and dicts). Tables and links are rendered from artifacts, never from generated text, so statuses, batches and URLs cannot be hallucinated.

**3. Competitor grounding.**
Every competitor name produced by the model is checked against the retrieved corpus (YC names, web titles and snippets); unverified names are dropped. YC matches at or above 50% are always included, even if the model omits them. Product names are also extracted from web result titles (e.g. `Loora - Your Personal AI English Tutor` -> `Loora`), so web competitors appear even when the model mentions them elsewhere.

**4. Format-robust parsing.**
Small models do not reliably follow output formats. The parser removes repeated sections and recovers the competitor list when the model writes it outside the expected line.

**5. End-of-context task reminder.**
With ~1,500 words of tool results in context, a 3B model tended to describe competitors instead of advising on the user's idea. Appending the idea and expected format right before generation keeps the analysis focused on the new idea. The reminder is sent to the model only and is not stored in graph state.

**6. No generated score.**
An LLM-generated 1-10 score did not discriminate between ideas (6-7/10 for almost every input, including one with five 99-100% matches). It was replaced by a factual summary line computed in code.

**7. Local model sized to the server.**
The VPS also runs other services, so a 3B model (~2-3 GB RAM at inference) was chosen over 7-8B models (~5 GB). The context window was raised from Ollama's default 2,048 to 8,192 tokens so all tool results fit.

## Results

**Retrieval quality**, example idea: *"AI travel planner that builds day-by-day trip itineraries"*

| Rank | Dense retrieval only | Dense retrieval + reranker |
|---|---|---|
| 1 | Stardrift 76% | Stardrift 98% |
| 2 | Waypoint Transit 71% (urban planning, irrelevant) | SkyLink 22% |
| 3 | Wanderlog 70% | Wanderlog 18% |
| 4 | Veeza AI 70% (visas, irrelevant) | filtered out (<10%) |
| 5 | PLANO AI 70% (construction, irrelevant) | filtered out (<10%) |

**Latency** (Langfuse trace, CPU-only inference): ~2 minutes per run.

| Step | Time |
|---|---|
| Tool selection (LLM) | ~1 min |
| Qdrant retrieval + reranking + web search | ~2 s |
| Analysis (LLM) | ~1 min |

Retrieval and reranking add negligible latency; total time is dominated by CPU inference of the local model.

## Security

- All credentials are kept in a server-side `.env` file with owner-only permissions.
- The app and Ollama are bound to `127.0.0.1` only; access goes through an SSH tunnel, and the UI is password-protected.

---

**Author:** Lilia Avagyan · [LinkedIn](https://www.linkedin.com/in/lil-ai-travel)

**Source code is private, available on request.**

