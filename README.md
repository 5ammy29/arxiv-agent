# arXiv Paper Digest & QA Agent

An end-to-end research agent that acts as an end-to-end workflow taking in either an **arXiv ID, arXiv URL, or natural-language research topic**, fetching an appropriate paper, parsing the PDF, indexing it as vectors for querying, producing an executive summary, and answering questions on the paper chosen.

The system is implemented as a **LangGraph stateful workflow** that utilizes local or open-source tools for retrieval and LLMs for summarization and question answering.

---

## Features

* **Flexible input**

  * arXiv ID
  * arXiv URL
  * Natural-language research topic

* **Query understanding**

  * Converts a broad research request into a useful search query.

* **arXiv retrieval**

  * Searches arXiv and selects a relevant paper for processing.

* **PDF processing**

  * Downloads and parses the selected paper locally.

* **Document chunking**

  * Splits the paper into searchable chunks.

* **Semantic retrieval**

  * Generates embeddings using `sentence-transformers`.
  * Stores vectors locally using FAISS.

* **Reranking**

  * Reranks retrieved chunks to improve the relevance of the evidence supplied to the LLM.

* **Executive briefing**

  * Produces a concise research briefing from the selected paper.

* **Grounded QA**

  * Allows follow-up questions about the paper.
  * Answers are grounded in retrieved paper content rather than relying only on the model's prior knowledge.

* **LangGraph orchestration**

  * Uses an explicit state graph to coordinate retrieval, processing, indexing, briefing, and QA.

* **Failure handling**

  * Handles invalid inputs and failures during retrieval, downloading, parsing, and processing.

---

# Architecture

The agent follows a stateful pipeline:

```text
                         ┌──────────────────────┐
                         │      User Input      │
                         │ arXiv ID / URL /     │
                         │     Research Topic   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Query Understanding │
                         │                      │
                         │ Normalize / rewrite  │
                         │ the research query   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   arXiv Retrieval    │
                         │                      │
                         │ Search arXiv API     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Paper Selection    │
                         │                      │
                         │ Select paper to      │
                         │ process              │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   PDF Fetch / Parse  │
                         │                      │
                         │ Download PDF and     │
                         │ extract text         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Chunk + Embed       │
                         │                      │
                         │ Split paper into     │
                         │ chunks and generate  │
                         │ embeddings           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Vector Store      │
                         │                      │
                         │ Local FAISS index    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Executive Briefing   │
                         │                      │
                         │ Generate concise     │
                         │ paper briefing       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     QA Loop          │
                         │                      │
                         │ Retrieve relevant    │
                         │ chunks → rerank →    │
                         │ generate grounded    │
                         │ answer               │
                         └──────────────────────┘
```

## LangGraph State

The graph passes a shared state between nodes.

Conceptually, the state contains:

```python
{
    "query": ...,
    "processed_query": ...,
    "papers": ...,
    "selected_paper": ...,
    "pdf_path": ...,
    "chunks": ...,
    "results": ...,
    "answer": ...
}
```

### Main state fields

| Field             | Purpose                                       |
| ----------------- | --------------------------------------------- |
| `query`           | Original user input                           |
| `processed_query` | Normalized/rewritten query used for retrieval |
| `papers`          | Papers returned by arXiv search               |
| `selected_paper`  | Paper selected for processing                 |
| `pdf_path`        | Local path to the downloaded PDF              |
| `chunks`          | Processed document chunks                     |
| `results`         | Retrieved/reranked evidence for a query       |
| `answer`          | Generated briefing or QA response             |

The graph separates **research acquisition** from **question answering**: once the paper has been processed and indexed, follow-up questions can use the existing searchable representation rather than downloading and processing the paper again.

---

# End-to-End Flow

For a new research request:

```text
User Query
    ↓
Query Understanding
    ↓
arXiv Search
    ↓
Paper Selection
    ↓
PDF Download
    ↓
PDF Parsing
    ↓
Chunking
    ↓
Embedding
    ↓
FAISS Index
    ↓
Executive Briefing
    ↓
Interactive QA
```

For each follow-up question:

```text
Question
    ↓
Semantic Retrieval
    ↓
Reranking
    ↓
Relevant Paper Context
    ↓
LLM
    ↓
Grounded Answer
```

---

# Tech Stack

| Component           | Technology            |
| ------------------- | --------------------- |
| Language            | Python 3.12           |
| Agent orchestration | LangGraph             |
| Paper source        | arXiv API             |
| PDF parsing         | PyMuPDF               |
| Embeddings          | Sentence Transformers |
| Vector search       | FAISS                 |
| LLM                 | Ollama (`qwen2.5:3b`) |
| Data validation     | Pydantic              |
| Configuration       | python-dotenv         |
| Testing             | pytest                |

The system is designed to run locally without requiring a paid vector database or hosted retrieval service.

---

# Setup

## 1. Clone the repository

```bash
git clone https://github.com/5ammy29/arxiv-agent.git
cd arxiv-agent
```

## 2. Create a Python environment

Python **3.12** is recommended.

Using Conda:

```bash
conda create -n arxiv-agent python=3.12
conda activate arxiv-agent
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project uses:

* `langgraph`
* `arxiv`
* `pymupdf`
* `sentence-transformers`
* `faiss-cpu`
* `pydantic`
* `python-dotenv`

---

## LLM Setup

The agent uses [Ollama](https://ollama.com/) to run the LLM locally.

Install Ollama:

```powershell
irm https://ollama.com/install.ps1 | iex
````

The default model is:

```text
qwen2.5:3b
```

To download the model, run:

```bash
ollama pull qwen2.5:3b
```

---

# Running the Agent

Run the CLI from the repository root.

```bash
python -m cli
```

The CLI accepts three types of input:

### arXiv ID

```text
Enter an arXiv ID, URL, or topic: 1706.03762
```

### arXiv URL

```text
Enter an arXiv ID, URL, or topic: https://arxiv.org/abs/1706.03762
```

### Natural-language topic

```text
Enter an arXiv ID, URL, or topic: transformer architecture
```

The agent then retrieves a relevant paper, processes it, generates a briefing, and allows follow-up questions.

---

# Testing

Run the complete test suite with:

```bash
PYTHONPATH=. pytest
```

The test suite covers the core components of the application, including:

* query processing
* arXiv retrieval
* document processing
* chunking
* embeddings
* retrieval
* reranking
* graph behavior
* error handling
* CLI-related behavior

The final implementation was tested end-to-end using arXiv IDs, URLs, and broad natural-language research topics.

---

# Example Run

### Input

```text
Enter an arXiv ID, URL, or topic: transformer architecture
```

The agent searches arXiv, selects a relevant Transformer paper, downloads and processes the PDF, builds the local retrieval index, and generates an executive briefing.

### Example briefing

```text
EXECUTIVE BRIEFING

Paper: Attention Is All You Need

Overview
The paper describes the Transformer, an architecture which uses only
the attention mechanism and does not rely on either recurrence or convolution.

Main Contribution
The authors suggest to use self-attention and multi-head attention as a
replacement of recurrent sequence modeling that allows processing of tokens in
parallel and modeling relationships between positions.

Architecture
The architecture suggested by the authors consists of two stacks - encoder and decoder.
Each layer of the architecture utilizes multi-head self-attention, position-wise
feed-forward networks, residual connections, and layer normalization.

Key Result
The Transformer architecture reaches very high translation accuracy while improving
the training parallelization substantially as compared to the recurrent architecture.

Significance
The architecture offers a scalable alternative to recurrent sequence modeling and was the
basis for subsequent language-model architectures.
```

The exact briefing depends on the paper retrieved and the generated model response.

---

# Example QA

After the briefing, the agent can answer follow-up questions using retrieved evidence from the processed paper.

### Q1

```text
Question: What is the key contribution of the paper?
```

```text
Answer:
The key contribution is the Transformer model, which uses
attention-based sequence modeling instead of recurrence and
convolution. It enables highly efficient parallel training while
maintaining the capability to capture token dependencies.
```

### Q2

```text
Question: Why does the architecture use multi-head attention?
```

```text
Answer:
Multi-head attention allows the model to attend to information from
different representation subspaces and positions simultaneously.
Different attention heads can therefore learn different relationships
within the sequence.
```

### Q3

```text
Question: How does the paper address the lack of recurrence?
```

```text
Answer:
The Transformer introduces positional encodings so that information
about token order is available to the model. Self-attention can then
operate over the sequence without requiring recurrent computation.
```

> The examples above illustrate the interaction format. Actual answers are generated from the retrieved paper context at runtime.

---

# Design Decisions & Tradeoffs

## Why LangGraph?

LangGraph is used since the application is intrinsically modeled as a stateful workflow as opposed to a workflow of one LLM call. Retrieval, document processing, indexing, brief writing, and QA are different steps with data flowing between them.

An ability to maintain state in the graph also helps understand the workflow better and serves as a good basis to add other branches to the agent or implement its recovery.

However, the cost is additional complexity of the architecture as opposed to writing all the logic inside a single Python function. In the case of a very small application, using LangGraph may become an overhead. In the current project, it does not.

## Why local FAISS?

FAISS was used as the vector store because it is lightweight, open source, and runs locally. This avoids introducing a hosted vector database or external infrastructure requirement.

The downside is that the current version is built to handle one paper at a time, or one local use case. It is not designed for a persistent, multi-user retrieval engine setup.

## Why semantic retrieval + reranking?

Using embedding for retrieval ensures an effective method of locating semantically relevant parts of the paper regardless of differences in how the question is phrased compared to the text in the document.

Subsequent reranking is performed to enhance the order of selected evidence before being presented to the LLM.

The downside of such an approach is increased computational costs. An alternative approach would be to base everything on vector similarity alone.

## Why process one selected paper?

The current agent focuses on producing a high-quality briefing and QA experience around a selected paper rather than attempting to synthesize an entire literature search.

This keeps the retrieval and grounding problem manageable and makes the resulting answers easier to associate with a specific source.

With more time, the system could be extended to process multiple papers and provide cross-paper comparison, citation-aware synthesis, and disagreement detection.

## What I would improve with more time

Several areas could be improved:

1. **Better paper selection**

   * Using better ranking criteria rather than the first few papers from the arXiv search results.

2. **Multi-paper research**

   * Retrieving and processing more papers for answering questions on the level of literature.

3. **Persistent indexing**

   * Caching processed papers and their embeddings to avoid reprocessing when asking the same question.

4. **Better document structure**

   * Preserving section headings, tables, equations, and page information during PDF extraction..

5. **Evaluation**

   * Adding a benchmark of retrieval and answers' quality with questions and answers being manually verified.

6. **More robust retrieval**

   * Researching hybrid retrieval based on both lexical and semantic criteria and better reranking.

---

# Known Limitations

* Paper retrieval relies on the quality of the search results on arXiv.
* Selecting just one paper may not be enough to answer general literature-review questions.
* The quality of PDF extraction is dependent on the structure of the input PDF.
* Tables, formulas, figures, and other complex structures may not be accurately reflected in text extraction.
* The current approach relies on a local FAISS index and is not a production-ready multi-user vector database.
* LLM-generated briefings and answers can still contain errors.
* Grounding helps to avoid ungrounded responses but does not guarantee that the responses are correct.
* Processing and embedding a paper locally introduces additional latency before the first briefing/QA interaction.
* The system currently focuses on one selected paper rather than performing multi-paper synthesis.

---

# Project Structure

```text
arxiv-agent/
│
├── project/
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   └── nodes.py
│   │
│   ├── nodes/
│   │   ├── arxiv.py
│   │   └── llm.py
│   │
│   ├── retrieval/
│   │   └── ...
│   │
│   └── cli.py
│
├── tests/
│   └── ...
│
├── data/
│   ├── papers/
│   └── indexes/
│
├── .gitignore
├── requirements.txt
└── README.md
```

Local papers, indexes, and other generated artifacts are kept out of version control.

---

# Future Work

Potential extensions include:

* Multi-paper retrieval and synthesis
* Citation-aware answers
* Persistent paper/index caching
* Hybrid BM25 + vector retrieval
* Improved reranking
* Section-aware chunking
* Figure/table extraction
* Retrieval evaluation benchmarks
* Streaming responses
* Web interface
* Conversation persistence
* More sophisticated query planning
