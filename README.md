# Retrieval‑Augmented Generation (RAG)

> Summary of the system – Upload one or more PDFs, ask a question in natural language, get an answer grounded in the uploaded documents together with the supporting passages (sources).

---

## Purpose

This mini‑service demonstrates a **retrieval‑augmented generation (RAG)** workflow:

* Flask API 
* MistralAI OCR/text extraction → chunking → OpenAI embeddings  
* ChromaDB vector store with on‑disk persistence in `data/chroma_db/`  
* A single `/documents` endpoint to _ingest_ PDFs and a `/question` endpoint to _query_ them and obtain answers.

---

## 📂 Repository layout

```bash
project-root/
├── app/                        # Flask application package
│   ├── main.py                 # Dev entry‑point  `python -m app.main`
│   ├── routers/                # Blueprints
│   │   ├── documents.py        #  POST /documents
│   │   └── question.py         #  POST /question
│   ├── services/               # RAG pipeline building blocks
│   │   ├── extractor.py        #  PDF/OCR → raw text
│   │   ├── chunker.py          #  text → chunks
│   │   ├── embedder.py         #  chunks → vectors
│   │   └── retriever.py        #  similarity search + LLM answer
│   ├── core/                   # Settings
│   │   └── config.py           #  pydantic BaseSettings using `.env`
│   ├── schemas/                # Pydantic request/response models
│   └── static/                 # Simple HTML landing page
│       ├── tractian_logo.svg   #  Tractian logo
│       └── index.html          #  Basic static html page for interaction
├── data/                       # Data folder
│   ├── uploads/                #  Copies of every PDF received
│   └── chroma_db/              #  Persistent ChromaDB collection
├── eval/                       # Evaluation folder
│   ├── eval_samples.pdf        #  Evaluation samples
│   ├── ragas_scores.csv        #  Scores from evaluation
│   └── evaluate_rag.py         #  Evaluation script (Ragas)
├── requirements.txt            # Exact package versions
├── requirements_eval.txt       # Exact package versions (for evaluation)
├── .env.example                # Template for required environment vars
├── challenge.pdf               # PDF file for the Machine Learning Engineering Role
├── machinery.pdf               # PDF file about machines - Artificially generated with ChatGPT
└── README.md                   # You are here 🙂
```

*(Some folders such as `data/` are created lazily at first run.)*

---

## ⚙️ Local setup

### 1. Prerequisites

* Python 3.10+  
* An OpenAI API key (used for embeddings)
* A MistralAI API key (used for OCR)
* A Requesty API key (used for LLM routing)


### 2. Clone & install

```bash
git clone git@github.com:leandromugnaini/RAG.git
cd RAG

# Create and activate an isolated env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Environment variables

Copy the template and fill in your secrets/paths:

```bash
cp .env.example .env
```

| Variable            | ⚠️ Required | Example value                 | Notes                                               |
|---------------------|-------------|-------------------------------|-----------------------------------------------------|
| `OPENAI_API_KEY`    | yes         | `sk‑...`                      | Uses the `text-embedding-3-small` embedding model   |
| `MISTRAL_API_KEY`   | yes         | `mistral‑...`                 | Uses the `mistral-ocr-latest` OCR model             |
| `UPLOAD_DIR`        | yes         | `data/uploads`                | Where incoming PDFs are stored                      |
| `ROUTER_API_KEY`    | yes         | `sk-...`                      | For LLM Routing (RAG Fallback policy)               |   


#### Requesty Router API key

As we are routing the LLMs traffic through **[Requesty](https://requesty.ai)**, we only need to set **one** credential in the `.env` file to use the retriever (and hence, ask questions to the system):

```bash
ROUTER_API_KEY=sk-...
```

The API key is configured with a `fallback_policy` that tries up to **four** models *in order*:

| Order | Model slug (provider)     | Retries |
| ----- | ------------------------- | ------- |
| 1     | `openai/gpt-4.1-mini`     | 1       |
| 2     | `google/gemini-1.5-flash` | 1       |
| 3     | `alibaba/qwen-turbo`      | 1       |
| 4     | `deepset/deepseek-chat`   | 1       |

If the first model fails (rate‑limit, timeout, 5xx) Requesty automatically bumps the request to the next one, ensuring you always get a response without having to change code. In this case, all models in the fallback chain are low-cost and with a large context window, allowing proper answers to the final user.



### 4. Run the dev server

```bash
# still inside the venv
python -m app.main
# 👉 The API is now listening on http://127.0.0.1:8000
```

You should see Flask logs like:

```
 * Running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## 🚀 Usage

### 1. Ingest PDFs – `POST /documents`

```bash
curl -X POST http://localhost:8000/documents \
     -F "files=@machinery.pdf" \
     -F "files=@challenge.pdf"
```

**Response `201 Created`**

```jsonc
{
  "message": "Documents processed successfully",
  "documents_indexed": 2,
  "total_chunks": 187
}
```

Behind the scenes the pipeline does:

1. Saves the raw PDF in `data/uploads/…`
2. OCR/extracts text  
3. Splits it into chunks  
4. Embeds each chunk with `OpenAIEmbeddings`  
5. Upserts vectors into the persistent Chroma collection.

### 2. Ask a question – `POST /question`

```bash
curl -X POST http://localhost:8000/question \
     -H "Content-Type: application/json" \
     -d '{"question":"What are the main components of Hydraulic Circuits?"}'
```

**Response `200 OK`**

```jsonc
{
  "answer": "The main components of hydraulic circuits are:\n\n1. Reservoir (tank): Volume sized greater than 3 times the pump flow per minute.\n2. Pump: Gear, vane, or axial-piston types; variable-displacement swashplate allows flow control.\n3. Actuators: Double-acting cylinders, rotary vane motors, radial piston motors.\n4. Valves: Directional ...",
  "sources": [
    {
      "chunk_index": 1,
      "filename": "machinery.pdf",
      "page_index": 7,
      "score": 0.6469521522521973,
      "text": "### 5.1 Fundamental Equations\n\n- Pascal's Law: p = F/A uniform in all directions.\n- Continuity: Q = A v; for hydraulic actuators, flow determines speed.\n\n\n### 5.2 Components of Hydraulic Circuits\n\n1. Reservoir (tank): Volume sized $>3 \\times$ pump flow per minute.\n2. Pump: Gear, vane, or axial-piston; variable-displacement swashplate allows flow control.\n3. Actuators: Double-acting cylinder, rotary vane motor, radial piston..."
    },
    {
      "chunk_index": 0,
      "filename": "machinery.pdf",
      "page_index": 7,
      "score": 0.8127862811088562,
      "text": "N- Scheduled oil analysis (TAN, TBN, wear metals).\n- Valve clearance adjustment every $40,000 \\mathrm{~km}$.\n- Compression test: healthy cylinder pressure $\\geq 90 \\%$ of spec.\n\n\n# 5 Hydraulic and Pneumatic Systems \n\nHydraulic systems use incompressible fluids to transmit power; pneumatics employ compressed air.\n\n### 5.1 Fundamental Equations\n\n- Pascal's Law: p = F/A uniform in all directions.\n- Continuity: Q = A v; for hydraulic actuators, flow determines speed.\n\n\n### 5.2 Components of Hydraulic Circuits..."
    }
  ]
}
```

The answer is generated by a `chat.completions` call with a system prompt that
**inserts the retrieved chunks as context** and instructs the model to:

* Answer based on the CONTEXT alone. If the context is insufficient to answer confidently, say so instead of inventing information.  
* Make sure to format the answer properly, but to not change the content of the answer or invent new information.  

---
## 📊 Quality Evaluation with RAGAS

This repository includes a **self-contained evaluation script** designed to measure how effectively the pipeline answers questions using an external benchmark dataset. The evaluation is performed with the [**Ragas**](https://github.com/explodinggradients/ragas) library on the first **50 samples** from the test split of the [`neural-bridge/rag-dataset-1200`](https://huggingface.co/datasets/neural-bridge/rag-dataset-1200) dataset, available on Hugging Face.

To ensure fairness in evaluation, all 50 samples’ contexts are combined into a single PDF file (`eval/eval_samples.pdf`). This file is then processed using the system's `documents/` endpoint. Subsequently, only the questions from the dataset are passed to the system, and the generated answers are saved. Ragas is then used to compute quality metrics by comparing the system’s answers with the ground truth.

### 📂 Evaluation Setup

| **Component**         | **Details**                                                                 |
|-----------------------|------------------------------------------------------------------------------|
| **Dataset**           | [`neural-bridge/rag-dataset-1200`](https://huggingface.co/datasets/neural-bridge/rag-dataset-1200) – test split, first 50 rows |
| **Evaluation Metrics**| `context_precision`, `faithfulness`, `answer_relevancy` via [`ragas`](https://github.com/explodinggradients/ragas) |
| **Script Location**   | `eval/evaluate_rag.py`                                                      |

The table below compiles the score of each metric for our system. As we can see, the scores show that our system is capable of retrieving and answering the questions properly. However, it is important to remember that this is a simple task and more complex data would probably make improvements necessary. 

### 📈 Results

| **Metric**            | **Score** | **Explanation**                                                                                       |
|-----------------------|-----------|--------------------------------------------------------------------------------------------------------|
| `context_precision`   | **1.0000** | Measures how well the answer aligns with the provided context. A perfect score means all facts were drawn from retrieved content. |
| `faithfulness`        | **0.9009** | Measures the truthfulness of the generated answer relative to its sources. Lower scores suggest possible hallucinations or exaggerations. |
| `answer_relevancy`    | **0.9677** | Indicates how directly the answer responds to the input question. Lower values imply the answer might be off-topic or verbose.            |


If you want to run the evaluation by yourself, make sure to install the necessary requirements from the file ```requirements_eval.txt```, turn on the app (```python -m app.main```), ingest the `eval/eval_samples.pdf` file using the `documents/` endpoint and run ```python eval/evaluate_rag.py```. A more detailed output, comparing the ground truth with the generated answer is available at the ```ragas_scores.csv```.


### Improvements

This system serves as a foundational example of a functional RAG implementation. However, several areas could be enhanced to improve the system's robustness, performance, and maintainability:

- RAG Optimization:
  - Implement more sophisticated retrieval strategies (e.g., keyword search, hybrid search, or re-ranking) to handle a larger number or higher complexity of documents. Evaluate the impact of database size on latency and answer quality.
  - Optimize chunk size, chunk overlap, and the amount of context provided to the LLM for optimal performance.
- LLM Interaction:
  - Refine the system prompt for better instruction following and potentially explore prompt engineering techniques for specific query types.
  - Strengthen defenses against prompt injection and other adversarial attacks.
- API Performance and Structure:
  - Migrate from Flask to a modern asynchronous framework like FastAPI to improve the API's performance, especially under concurrent load.
- Data Storage:
  - Consider using a more robust and maintainable database solution, such as PostgreSQL with a vector extension, for long-term storage and scalability compared to on-disk persistence.
- Code Quality:
  - Develop and integrate comprehensive unit and integration tests to improve code reliability and facilitate future development.
