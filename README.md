# 🏢 AI HR Policy Agent

> **Open-source RAG agent** that answers employee questions about company policies using AI-powered document understanding.

[![CI — Lint, Test & Security](https://github.com/arpitkorde/ai-hr-policy-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/arpitkorde/ai-hr-policy-agent/actions/workflows/ci.yml)
[![Deploy to Cloud Run](https://github.com/arpitkorde/ai-hr-policy-agent/actions/workflows/deploy.yml/badge.svg)](https://github.com/arpitkorde/ai-hr-policy-agent/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What It Does

Upload your company's HR policy documents (PDF, DOCX, TXT), and employees can ask questions in natural language:

- _"How many vacation days do I get?"_
- _"What's the parental leave policy?"_
- _"How do I file a grievance?"_

The agent retrieves relevant policy sections and generates accurate, **cited** answers — never hallucinating beyond what's in the documents.

---

## 🏗️ Architecture

```
Employee Question
       │
       ▼
┌──────────────────┐
│  Vector Search   │──── ChromaDB (HuggingFace embeddings)
│  (Top-20 chunks) │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  BERT Reranker   │──── Cross-Encoder (PyTorch)
│  (Top-5 chunks)  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Gemini LLM      │──── Google Gemini API
│  (Answer + Cite) │
└──────────────────┘
```

### Why Reranking?

Vector similarity search is fast but imprecise. The **BERT cross-encoder reranker** jointly encodes the query and each document, producing far more accurate relevance scores. This two-stage approach (retrieve broadly → rerank precisely) is used in production search systems at Google, Bing, and Amazon.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Google Gemini API | Answer generation |
| **RAG Framework** | LangChain | Pipeline orchestration |
| **Vector DB** | ChromaDB | Semantic search |
| **Embeddings** | HuggingFace `sentence-transformers` | Local embedding generation |
| **Reranker** | BERT Cross-Encoder (PyTorch) | Precision reranking |
| **API** | FastAPI | REST endpoints |
| **UI** | Streamlit | Chat + admin interface |
| **Evaluation** | RAGAS | RAG quality metrics |
| **Deployment** | Google Cloud Run | Serverless hosting |
| **CI/CD** | GitHub Actions | DevSecOps pipeline |

---

## 🔒 DevSecOps Pipeline

Every push triggers a comprehensive security pipeline:

| Stage | Tool | What It Checks |
|-------|------|----------------|
| **Lint** | Ruff | Code quality & style |
| **Test** | Pytest | Unit & integration tests |
| **SAST** | Bandit | Python security vulnerabilities |
| **Deps** | Safety | Known CVEs in dependencies |
| **Secrets** | Gitleaks | Hardcoded API keys/passwords |
| **Container** | Trivy | Docker image vulnerabilities |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Google Gemini API key](https://aistudio.google.com/apikey)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/arpitkorde/ai-hr-policy-agent.git
cd ai-hr-policy-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 5. Start the API server
uvicorn src.api.server:app --reload

# 6. (In another terminal) Start the UI
streamlit run src/ui/app.py
```

### Docker

```bash
docker build -t hr-policy-agent .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your-key hr-policy-agent
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload HR policy document (PDF/DOCX/TXT) |
| `POST` | `/query` | Ask a question about policies |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Knowledge base statistics |
| `GET` | `/prompts` | List prompt template versions |

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do I get?"}'
```

**Response:**
```json
{
  "answer": "Employees are entitled to 20 days of paid annual leave per calendar year...",
  "sources": [{"document": "hr_policy.pdf", "page": 3}],
  "metrics": {"latency_ms": 1250, "tokens_used": 580, "chunks_retrieved": 20, "chunks_after_rerank": 5}
}
```

---

## 🧪 Evaluation (RAGAS)

The project includes automated RAG evaluation using [RAGAS](https://docs.ragas.io/):

- **Faithfulness** — Is the answer grounded in retrieved documents?
- **Answer Relevancy** — Does it actually answer the question?
- **Context Precision** — Are the right chunks being retrieved?

```python
from src.rag.evaluation import RAGEvaluator

evaluator = RAGEvaluator()
result = evaluator.evaluate_single(
    question="How many vacation days?",
    answer="Employees get 20 days of PTO.",
    contexts=["Employees are entitled to 20 days of paid annual leave..."]
)
print(f"Faithfulness: {result.faithfulness_score:.2f}")
```

---

## 🏭 RAG vs Fine-Tuning: When to Use What

This project uses **RAG** (Retrieval-Augmented Generation) rather than fine-tuning. Here's why:

| Factor | RAG ✅ | Fine-Tuning |
|--------|--------|-------------|
| **Data freshness** | Real-time (just upload new docs) | Requires retraining |
| **Hallucination control** | Grounded in retrieved context | Can still hallucinate |
| **Cost** | Minimal (no training compute) | Expensive GPU training |
| **Auditability** | Citations show exactly where | Black box |
| **Best for** | Knowledge Q&A, policy lookup | Style/tone adaptation, domain language |

**When would fine-tuning be better?** When you need the model to learn domain-specific language patterns (e.g., legal jargon), adopt a specific communication style, or perform specialized reasoning that generic models struggle with. In practice, most enterprise knowledge Q&A use cases are best served by RAG with a strong reranker — which is exactly what this project implements.

---

## ☁️ Cloud Deployment

### Google Cloud Run (Serverless)

The project includes a complete CI/CD pipeline for automated deployment:

1. **Set up GCP**: Create a project, enable Cloud Run & Artifact Registry APIs.
2. **Configure Secrets**: Add `GCP_PROJECT_ID`, `GOOGLE_API_KEY`, `WIF_PROVIDER`, and `WIF_SERVICE_ACCOUNT` to GitHub repository secrets.
3. **Push to main**: The deploy pipeline builds, scans, and deploys automatically.

See [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) for the full pipeline.

---

## 📂 Project Structure

```
ai-hr-policy-agent/
├── src/
│   ├── rag/
│   │   ├── ingest.py          # Document loading & chunking
│   │   ├── vector_store.py    # ChromaDB operations
│   │   ├── reranker.py        # BERT cross-encoder reranker
│   │   ├── chain.py           # Gemini RAG chain
│   │   ├── prompts.py         # Versioned prompt templates
│   │   └── evaluation.py      # RAGAS evaluation
│   ├── api/
│   │   └── server.py          # FastAPI endpoints
│   ├── ui/
│   │   └── app.py             # Streamlit interface
│   └── config.py              # Pydantic settings
├── tests/
│   └── test_rag.py
├── data/
│   └── sample_hr_policy.txt
├── .github/workflows/
│   ├── ci.yml                 # Lint, test, SAST, dep scan
│   └── deploy.yml             # Build, scan, deploy
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── .gitleaks.toml
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

All PRs must pass the CI pipeline (lint, tests, security scans).

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⭐ Star History

If this project helped you, please give it a ⭐ on GitHub!
