# HR Assist – AI‑Powered HR Policy Chatbot

## Overview
HR Assist is a conversational AI assistant that helps employees and HR staff retrieve HR‑policy information, staff recruitment guidelines, and related resources. It supports **role‑based access control (RBAC)**, dual AI back‑ends (Groq cloud model & Ollama local model), and an optional web‑search tool.

## Features
- **Chat UI** – Glassmorphism design, markdown rendering, citations, and timestamps.
- **RBAC** – Three roles (HR, Staff, General) with scoped access to documents and internet search.
- **Dual‑Mode AI** – 
  - **Online**: Groq (`ChatGroq`) for fast, cloud‑based responses.
  - **Offline**: Ollama (`ChatOllama`) for local inference when internet is unavailable.
- **RAG** – PDF ingestion (`hr_policy.pdf`, `staffrecruitment.pdf`) into a Chroma vector store.
- **Web Search** – DuckDuckGo integration for real‑time internet results (HR role only).
- **Docker Support** – Ready for deployment to Hugging Face Spaces or other containers.
- **Git‑Backed** – Repository linked to `https://github.com/Rbcuv/hr-chatbot`.

## Quick Start (Local Development)
```bash
# Clone the repo (if not already)
git clone https://github.com/Rbcuv/hr-chatbot.git
cd hr-chatbot

# Create a virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # macOS/Linux

pip install -r requirements.txt

# Add environment variables (create .env)
echo "GROQ_API_KEY=your_key_here" >> .env
echo "OLLAMA_HOST=http://localhost:11434" >> .env

# Run the app
python app.py
# Open http://127.0.0.1:7860 in a browser
```

## Docker Deployment
```bash
docker build -t hr-assist .
docker run -p 7860:7860 --env-file .env hr-assist
```

## Project Structure
```
├─ app.py                # Flask API, model routing, web‑search
├─ data_handler.py       # Helper functions for PDF loading & filtering
├─ extract_pdf.py        # PDF extraction utilities
├─ hrchatbot.py / hrchatbot1.py  # Core chatbot logic (legacy & updated)
├─ chroma_db_v2/         # Vector store (persisted)
├─ hr_policy.pdf
├─ staffrecruitment.pdf
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

## Configuration
| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | API key for Groq cloud model |
| `OLLAMA_HOST` | URL of local Ollama server |
| `PERSIST_DIR` | Directory for Chroma DB persistence (`./chroma_db_v2`) |
| `PDF_PATHS` | List of PDFs to ingest (`["hr_policy.pdf","staffrecruitment.pdf"]`) |

## Contributing
1. Fork the repository.
2. Create a feature branch.
3. Submit a Pull Request with clear description and tests (if any).

## License
MIT – see `LICENSE` file.
