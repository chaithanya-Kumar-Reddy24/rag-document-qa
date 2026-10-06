# Interview Preparation

## 30-second explanation (for HR / non-technical)

"I built an app that lets you upload your own documents — PDFs, Word files,
spreadsheets — and ask questions about them in plain English, like chatting
with an assistant that has actually read your files. Instead of guessing, it
looks up the exact parts of your documents that answer your question and
always shows you where the answer came from, so you can verify it yourself."

## 1-minute explanation (technical workflow)

"It's a Retrieval-Augmented Generation, or RAG, system. When you upload a
document, I extract its text, clean it, and split it into small overlapping
chunks so each piece is self-contained but not too big. Each chunk gets turned
into a vector embedding — a numeric fingerprint of its meaning — using OpenAI's
embedding model, and I store those vectors in ChromaDB, a local vector
database, along with metadata like the filename and page number.

When you ask a question, I embed the question the same way and run a cosine
similarity search against the stored vectors to find the most relevant chunks.
Only chunks above a relevance threshold get passed to the LLM, as numbered
context blocks, with instructions to answer only from that context. The LLM's
answer, plus the exact sources it used, are shown back in the UI."

## 3-minute explanation (full architecture)

"The pipeline is orchestrated with LangGraph as an explicit state graph rather
than one long function, which makes each step inspectable and testable on its
own.

Ingestion: a file is validated (extension, size), loaded with a
format-specific loader — pypdf for PDFs, python-docx for Word, plain text for
TXT/CSV — producing one LangChain `Document` per page or logical section, each
tagged with `source`, `file_type`, and `doc_id`. The text is cleaned (Unicode
normalization, de-hyphenation across line breaks, whitespace collapsing) and
split with `RecursiveCharacterTextSplitter`, which tries paragraph boundaries
first, then lines, then words, so chunks stay coherent. Each chunk gets a
`chunk_id` and is embedded and stored in ChromaDB, using the document's content
hash as its ID for duplicate detection, so re-uploading the same file is a
no-op.

Query time: the LangGraph workflow has five nodes. `rewrite_query` uses the LLM
to turn a follow-up question like 'What are its advantages?' into a standalone
query by resolving pronouns against recent chat history — but that history is
only ever used for this rewriting step, never as a source of facts.
`retrieve_documents` calls the `HybridRetriever`: it runs vector (cosine
similarity) search and BM25 keyword search in parallel, fuses the two ranked
lists with Reciprocal Rank Fusion, re-scores the fused candidates with the LLM
acting as a re-ranker, and only then applies the relevance threshold and
Top-K — all optionally filtered to specific documents. `validate_context`
deduplicates chunks and builds a numbered context string, capped to a
character budget. A conditional edge then routes to `generate_answer` only if
that context is non-empty — otherwise it routes straight to a `no_context`
node that returns a fixed message without calling the LLM at all. That's the
main hallucination guard: no relevant chunks, no LLM call, no chance of the
model making something up. Finally `format_sources` turns the retrieved
chunks' metadata into a deduplicated source list for the UI.

The Streamlit UI streams the answer token by token, shows sources per answer,
and lets you expand the exact retrieved text with its relevance score —
nothing is hidden or asserted without a way to check it. A separate
Evaluation Dashboard runs a question set through the same pipeline and scores
retrieval and generation quality with an LLM judge, so retrieval and
generation quality claims are backed by numbers, not just a demo."

---

### What is RAG?
Retrieval-Augmented Generation: instead of relying only on what an LLM
memorized during training, you retrieve relevant external text at query time
and feed it to the model as context, so it can answer using up-to-date or
private information it never saw during training.

### Why RAG instead of directly asking an LLM?
The LLM has no knowledge of your private documents, its training data has a
cutoff, and pasting whole documents into every prompt doesn't scale (cost,
latency, context-window limits) and doesn't tell you *where* an answer came
from. RAG retrieves only the relevant few chunks and can cite them.

### What are embeddings?
Numeric vectors that represent the *meaning* of text, produced by a model
trained so that semantically similar text ends up close together in vector
space, letting you compare meaning with simple math like cosine similarity
instead of exact keyword matches.

### How does semantic search work?
The query is embedded with the same model used for the documents, then
compared to every stored chunk vector (here, via cosine similarity, since the
Chroma collection is configured with `hnsw:space = cosine`). The chunks with
the highest similarity are the most likely to be relevant.

### Why ChromaDB?
It's an embedded, persistent vector database with no separate server to run —
good for a self-contained portfolio project — while still supporting metadata
filtering and per-collection distance metrics, which this project needs for
document-scoped search.

### Why LangChain?
It standardizes the interfaces for chat models, embedding models, prompts, and
text splitters, so swapping the LLM or embedding provider is a config change,
not a rewrite, and it provides battle-tested utilities like
`RecursiveCharacterTextSplitter`.

### Why LangGraph?
The RAG workflow isn't a single straight line — it needs a conditional branch
(context found vs. not found) and explicit, typed state that's easy to test
node-by-node. LangGraph models that directly as a graph instead of nested
if/else logic buried in one function.

### How does chunking work?
`RecursiveCharacterTextSplitter` tries to split on paragraph breaks first, then
line breaks, then sentences/words, backing off only as needed to respect
`chunk_size`, and keeps `chunk_overlap` characters of context between
consecutive chunks so ideas that span a chunk boundary aren't lost.

### How are hallucinations reduced?
Three layers: (1) the relevance threshold filters out weakly related chunks
before they ever reach the LLM; (2) if nothing clears the threshold, the graph
routes to a fixed "not found" response and skips the LLM call entirely; (3) the
system prompt explicitly instructs the model to answer only from the numbered
context and to say the fixed message if the answer isn't there.

### How are sources generated?
Each retrieved chunk already carries `source`, `page`/`rows`, and `chunk_id` in
its metadata from ingestion. `format_sources` reads that metadata off the
chunks that were actually used for the answer and deduplicates it — sources
are never generated or guessed by the LLM.

### How does multi-document retrieval work?
All chunks from all documents live in one Chroma collection, distinguished by
`doc_id` metadata. A query with no `doc_ids` filter searches every document;
selecting specific documents in the UI passes their `doc_id`s as a filter to
both the vector search and the BM25 index, so the search space is narrowed
before ranking happens in either retriever.

### Why isn't vector search alone always sufficient?
Embeddings capture meaning well but are weaker on exact terms without a
strong semantic "neighborhood" — product codes, IDs, version numbers, precise
names. Keyword/BM25 retrieval complements that: it matches exact or
near-exact terms regardless of surrounding meaning. This project runs both
and fuses them with Reciprocal Rank Fusion, then re-ranks the combined
candidates with the LLM before applying Top-K and the relevance threshold.

### How does conversational follow-up work?
A short sliding window of recent chat turns is kept in memory. Before
retrieval, the LLM rewrites the latest question into a standalone query using
that history (resolving "it", "its", etc.). The rewritten query is used only
for retrieval and for the LLM's final grounding — history itself never
substitutes for retrieved evidence.

### What happens when no relevant document context is found?
The `validate_context` node produces an empty context string, the conditional
edge routes to `no_context` instead of `generate_answer`, and the app returns
"I could not find the answer in the uploaded documents." without ever calling
the LLM for that turn.

### What are the limitations of this system?
Single-machine ChromaDB doesn't scale to huge document sets or concurrent
writers well; the in-memory BM25 index is rebuilt from scratch (fine for a
portfolio-scale document set, not for a very large one); the LLM re-ranker
adds an extra API call and some latency per question compared to a dedicated
cross-encoder; very large or scanned/image PDFs without OCR won't extract
text; the relevance threshold is a single global value rather than adaptive
per query; and there's no authentication or per-user data isolation.

### How could this system be deployed at scale?
Swap ChromaDB for a managed/cloud vector database (e.g., a hosted Chroma,
pgvector, or a managed vector DB service) for durability and concurrent
access; move the BM25 index to an incrementally-updated search service (e.g.
OpenSearch/Elasticsearch) instead of an in-memory rebuild; move ingestion to
an async job queue so uploads don't block the UI; add caching for repeated
queries and consider a faster dedicated cross-encoder for re-ranking under
load; and add authentication and per-tenant collections.
