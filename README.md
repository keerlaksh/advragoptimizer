# advrag — RAG Pipeline Optimizer

advrag evaluates chunking, embedding, and retrieval choices against a labelled
question set for an uploaded document collection. The Streamlit app is a
development/evaluation prototype; its manually entered tenant is not an
authentication boundary.

## Run the baseline

From this directory in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

Enter a Groq API key in the password-masked sidebar for every experiment:
answer generation and response evaluation are mandatory after retrieval.
Dense, hybrid, reranking, embedding, ChromaDB, and mathematical retrieval
metrics do not use Groq; HyDE and query expansion call Groq automatically
when selected. The frontend keeps the key in the current Streamlit session
and does not write it to reports or `.env`. For non-frontend scripts,
`GROQ_API_KEY` may be configured in `.env`; never commit `.env` or put
credentials in Python source files.

The default experiment selects all four chunking strategies, all three
embedding models, and Dense plus Hybrid retrieval, producing the 12 base
chunking/embedding configurations. The UI shows progress for dataset
validation, ingestion, chunking, embedding/storage, retrieval, retrieval
metrics, answer generation, response evaluation, and recommendation.

Upload source documents and a JSON question set. For standard retrieval
metrics, add a verified source quote as `evidence` or one or more exact source
quotes as `evidence_passages` per answerable question. Users never supply chunk
IDs: the application creates IDs during chunking, stores them as Chroma record
IDs, and maps each evidence quote to matching chunk text separately for each
chunking strategy. The evaluator then calculates Hit Rate@K, Precision@K,
Recall@K, MRR, and nDCG@K mathematically from those internal relevant IDs and
the ranked retrieved IDs.

If evidence is absent or does not match all supplied quotes in the generated
chunks, retrieval metrics are marked unavailable with a reason; the system
does not infer relevance from the reference answer or fabricate labels. Add
all known supporting quotes to `evidence_passages` for questions supported by
multiple passages. Unanswerable questions have no relevant evidence and their standard relevance
metrics are unavailable. Answer generation and response evaluation always run
after retrieval and use Groq. The selected retrieval method controls only
whether an additional LLM call is used during retrieval: HyDE and query
expansion use Groq, while Dense and Hybrid do not.
Response metrics include an explicit hallucination rate, surfaced from the
judge's unsupported-claim rate (0 means no unsupported claims; higher is worse).
Retrieval coverage is summarized across questions, so a partially available
pipeline clearly reports how many questions had usable evidence labels.

Groq calls retry up to twice when its suggested rate-limit wait is ten minutes
or less. Longer waits are reported. Each question in each pipeline requires
two Groq calls for answer generation and response evaluation; HyDE and query
expansion add another retrieval call. For example, 4 chunkers × 3 embedding
models × 2 retrieval methods × 3 questions can require at least 144 calls.
If limits persist, wait for the provider's limit window to reset and reduce
questions or selected pipeline variants. The app writes Chroma data under
`chroma_data` and reports under `experiments/results`.

## Project modules

| Module | Responsibility |
| --- | --- |
| `ingestion/document_loader.py` | Extract text and attach document/source metadata |
| `chunking/` | Fixed-size, semantic, hierarchical, and structure-aware chunking |
| `embeddings/embedding_manager.py` | MiniLM, E5-small, and BGE-base embeddings |
| `vectorstore/chroma_manager.py` | Per-configuration Chroma collections and tenant-filtered search |
| `retrieval/` | Dense, BM25+dense hybrid, HyDE, query expansion, filters, and reranking |
| `generation/answer_generator.py` | Grounded answer generation with retrieved evidence |
| `evaluation/` | Evidence-to-chunk mapping, mathematical retrieval metrics, and response metrics |
| `experiments/` | Experiment orchestration and quality-based recommendations |
| `app.py` | Streamlit workflow and result presentation |

After validating the baseline, select additional chunking strategies, embedding
models, and retrieval methods in the sidebar to expand the experiment matrix.
HyDE and query expansion add Groq retrieval calls; all methods require Groq for
answer generation and response evaluation. The report records measured
retrieval and end-to-end latency. API cost
is not currently estimated, and recommendation quality scores are not
latency- or cost-weighted.

## Evaluation expectations

Each dataset entry requires a non-empty `question`. Add a verified
`reference_answer` for response evaluation. To calculate retrieval metrics,
include verified evidence copied from the uploaded source:

```json
{
  "question": "How many days of annual leave do employees receive?",
  "reference_answer": "Employees receive 20 days of annual paid leave.",
  "evidence": "Employees are entitled to 20 days of annual paid leave."
}
```

Use `evidence_passages` with a list of source quotes when multiple passages
support a question. Evidence quotes must match text in the uploaded document
(case and whitespace differences are normalized). No chunk IDs are needed in the JSON.
Without matching evidence, the retrieval metrics are explicitly unavailable;
the reference answer alone is not treated as ground-truth evidence.

## Sample evaluation datasets

The sample JSON files are paired with synthetic source documents. Upload the
matching Markdown document and select the corresponding JSON dataset in the
app:

| Sample collection | Source document | Evaluation questions |
| --- | --- | --- |
| Employee handbook | `evaluation/sample_data/employee_handbook.md` | `evaluation/sample_datasets/employee_handbook.json` |
| IT support guide | `evaluation/sample_data/it_support_guide.md` | `evaluation/sample_datasets/it_support_guide.json` |

Each set includes answerable, multi-passage, and unanswerable questions.
Answerable examples include evidence quotes copied from their paired synthetic
documents, allowing the app to map quotes to each chunking strategy's generated
IDs automatically. Unanswerable examples correctly have no evidence and report
retrieval relevance metrics as unavailable.

Before production, replace the development tenant field with authenticated
authorization context and enforce document-level access controls, upload limits,
and data-retention policies.
