# HR Assist — Complete Project Walkthrough (Learn Everything From Scratch)

---

## Table of Contents
1. [The Big Picture — What Does This Project Do?](#1-the-big-picture)
2. [Key Concepts You Need to Understand First](#2-key-concepts)
3. [How the Files Fit Together](#3-how-files-fit-together)
4. [File-by-File Deep Dive](#4-file-by-file-deep-dive)
5. [The Complete Request Lifecycle — What Happens When a User Asks a Question?](#5-request-lifecycle)
6. [Deployment & Infrastructure](#6-deployment)
7. [Glossary](#7-glossary)

---

## 1. The Big Picture — What Does This Project Do? <a name="1-the-big-picture"></a>

Imagine you're a new employee at a company. You have questions like:
- "How many sick leaves do I get?"
- "What's the dress code policy?"
- "Who got shortlisted for the new position?"

Instead of reading a 200-page HR policy PDF, you open a chat window, type your question, and get an instant answer — with the exact page reference from the PDF.

**That's what HR Assist does.** It's a chatbot that:
1. **Reads** your company's HR policy PDFs
2. **Understands** them by converting the text into mathematical vectors (numbers)
3. **Finds** the most relevant sections when you ask a question
4. **Sends** those sections to an AI model (like ChatGPT, but different ones)
5. **Returns** a friendly, accurate answer

---

## 2. Key Concepts You Need to Understand First <a name="2-key-concepts"></a>

### 2.1 What is an LLM (Large Language Model)?

An LLM is an AI that can understand and generate human text. Think of it like a super-smart autocomplete. Examples: ChatGPT, LLaMA, Phi3.

In your project, you use **two** LLMs:
- **Groq (Cloud)** — Runs LLaMA 3.1 on Groq's servers over the internet. Fast (1-3 seconds). Requires an API key.
- **Ollama (Local)** — Runs Phi3 directly on YOUR computer. Slow (2-3 minutes on CPU) but works offline.

### 2.2 What is RAG (Retrieval-Augmented Generation)?

This is THE core technique of your entire project. Here's the problem RAG solves:

> **Problem:** If you just ask an LLM "What is the leave policy?", it will make something up because it was never trained on YOUR company's specific HR policy.

> **Solution (RAG):** Before asking the LLM, you first SEARCH your PDF for relevant sections, then you PASTE those sections into the question. Now the LLM has the real information to work with.

RAG has two steps:
1. **Retrieval** — Search the PDF and find the 5 most relevant chunks of text
2. **Generation** — Give those chunks + the user's question to the LLM and let it generate an answer

### 2.3 What are Embeddings?

Embeddings are the magic that makes the "search" part of RAG work.

An embedding converts text into a list of numbers (a "vector"). For example:
- "sick leave policy" → `[0.12, -0.45, 0.78, ...]` (384 numbers)
- "annual leave rules" → `[0.11, -0.43, 0.76, ...]` (384 numbers)
- "company dress code" → `[0.89, 0.12, -0.33, ...]` (384 numbers)

Notice how "sick leave policy" and "annual leave rules" have similar numbers? That's because they're semantically similar! The embedding model (`all-MiniLM-L6-v2` in your project) learned to place similar meanings close together in number-space.

### 2.4 What is a Vector Database (ChromaDB)?

A vector database stores all those number-lists (embeddings) and lets you do **similarity search**.

When a user asks "How many sick leaves do I get?", the system:
1. Converts the question into numbers using the same embedding model
2. Searches ChromaDB for the 5 stored chunks whose numbers are closest
3. Returns those chunks as "context" for the LLM

Your project uses **ChromaDB**, stored in the `chroma_db_v2/` folder.

### 2.5 What is Flask?

Flask is a Python web framework. It turns your Python code into a web server that can:
- Serve web pages (your `index.html`)
- Accept HTTP requests (like when the chat UI sends a question)
- Return JSON responses (the AI's answer)

Think of Flask as the "waiter" in a restaurant — it takes orders (requests) from the customer (browser), passes them to the kitchen (your AI code), and brings back the food (response).

### 2.6 What is LangChain?

LangChain is a Python library that simplifies working with LLMs. Instead of writing complex code to:
- Load PDFs
- Split text into chunks
- Create embeddings
- Search vector databases
- Format prompts
- Call AI models

...LangChain gives you ready-made building blocks that you can chain together (hence the name "LangChain").

---

## 3. How the Files Fit Together <a name="3-how-files-fit-together"></a>

```
┌──────────────────────────────────────────────────────────────┐
│                       USER'S BROWSER                         │
│                        (index.html)                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Glassmorphism Chat UI                                 │  │
│  │  - Login modal (HR / Staff / General)                  │  │
│  │  - Chat input box                                      │  │
│  │  - Message bubbles with timestamps                     │  │
│  │  - Mode toggle (Groq / Local)                          │  │
│  └────────────────────────────────────────────────────────┘  │
│         │ HTTP POST /chat  (sends JSON)                      │
│         │ HTTP POST /login (sends credentials)               │
│         ▼                                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  FLASK SERVER  (app.py)                                │  │
│  │                                                        │  │
│  │  1. Validate the incoming request                      │  │
│  │  2. Determine user role (HR/Staff/General)             │  │
│  │  3. Search ChromaDB for relevant PDF chunks            │  │
│  │  4. (If HR) Also search the web via DuckDuckGo         │  │
│  │  5. Build a prompt with context + question             │  │
│  │  6. Send to LLM (Groq or Ollama)                       │  │
│  │  7. Return AI answer as JSON                           │  │
│  └────────────────────────────────────────────────────────┘  │
│         │                              │                     │
│    ┌────▼─────┐              ┌─────────▼──────────┐          │
│    │ ChromaDB │              │  LLM (AI Model)    │          │
│    │ (Vector  │              │  - Groq (cloud)    │          │
│    │  Store)  │              │  - Ollama (local)  │          │
│    └──────────┘              └────────────────────┘          │
│                                                              │
│  SUPPORT FILES:                                              │
│  - data_handler.py   → reads staffrecruitment.xlsx           │
│  - extract_pdf.py    → utility to preview PDF content        │
│  - hrchatbot.py      → early prototype v1 (terminal only)    │
│  - hrchatbot1.py     → improved prototype v2 (terminal only) │
│  - Dockerfile        → packages everything for cloud deploy  │
│  - requirements.txt  → list of Python packages needed        │
│  - .env              → secret API keys (never pushed to Git) │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. File-by-File Deep Dive <a name="4-file-by-file-deep-dive"></a>

### 4.1 `app.py` — The Brain of the Project (290 lines)

This is the main file. Everything runs from here. Let's walk through it section by section.

#### Section A: Imports & Setup (Lines 1–28)

```python
from dotenv import load_dotenv
load_dotenv()
```
**What this does:** Reads the `.env` file and loads its contents (like `GROQ_API_KEY=gsk_abc123...`) into the system's environment variables. This is how your code accesses the API key without hardcoding it.

```python
app = Flask(__name__)
CORS(app)
```
**What this does:** Creates a Flask web server. `CORS(app)` allows your frontend (index.html) to talk to the backend even if they're on different ports/domains. Without CORS, the browser would block the requests for security reasons.

```python
PERSIST_DIR = "./chroma_db_v2"
PDF_PATHS   = ["hr_policy.pdf", "staffrecruitment.pdf"]
```
**What this does:** Tells the app where to save/load the vector database and which PDFs to ingest.

---

#### Section B: Web Search (Lines 30–45)

```python
def web_search(query: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=4))
```
**What this does:** Uses DuckDuckGo to search the internet. This is ONLY used when an HR user asks a question — it adds real-time web results to the context so the AI can answer questions that go beyond the PDF.

---

#### Section C: Embeddings & Vector DB (Lines 47–86)

This is where the RAG "Retrieval" part is set up.

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```
**What this does:** Loads a small AI model (only ~80MB) that converts text into 384-dimensional vectors. This model was trained by Microsoft and is free to use. It runs locally on your machine — no API key needed.

```python
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
```
**What this does:** Checks if the vector database already exists on disk. If yes, load it (fast). If no, build it from scratch (slow, one-time operation).

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
texts = text_splitter.split_documents(documents)
```
**What this does:** The PDFs might have hundreds of pages. You can't send ALL of that to the AI (it has a token limit). So this splits the entire PDF into small chunks of ~1000 characters each, with 150 characters of overlap between chunks (so no information is lost at the boundaries).

Example:
- Chunk 1: characters 1–1000
- Chunk 2: characters 851–1850 (overlaps with chunk 1 by 150)
- Chunk 3: characters 1701–2700
- ...and so on

```python
db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=PERSIST_DIR)
```
**What this does:** Takes every chunk, converts it to a 384-number vector using the embedding model, and stores it in ChromaDB. This is the one-time "indexing" step.

---

#### Section D: LLM Models (Lines 88–105)

```python
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=1024,
    model_kwargs={"seed": 42},
)
```
**What this does:** Sets up the cloud AI model.
- `model="llama-3.1-8b-instant"` — Uses Meta's LLaMA 3.1 model (8 billion parameters)
- `groq_api_key=os.getenv("GROQ_API_KEY")` — Reads the API key from environment
- `temperature=0` — Makes responses deterministic (same question = same answer every time). Higher temperature = more creative/random.
- `seed=42` — Another way to ensure consistency
- `max_tokens=1024` — Limits the response length

```python
local_llm = ChatOllama(
    model="phi3",
    temperature=0,
    num_ctx=4096,
    seed=42,
)
```
**What this does:** Sets up the local AI model (Microsoft's Phi3, running on your machine via Ollama).
- `num_ctx=4096` — The "context window" — how much text the model can see at once (question + PDF chunks + prompt combined must fit in this)

---

#### Section E: The Prompt Template (Lines 107–144)

```python
prompt = ChatPromptTemplate.from_template("""You are HR Assist, a friendly...
Rules:
- Answer clearly and concisely...
Context:
{context}
Employee Question: {question}
Answer:""")
```
**What this does:** This is the instruction manual you give to the AI every single time it answers a question. It has three placeholders:
- `{context}` — Gets replaced with the relevant PDF chunks found by ChromaDB
- `{question}` — Gets replaced with the user's actual question
- `{role_instructions}` — Gets replaced with role-specific rules (HR/Staff/General)

The rules are critical. Without them, the AI might:
- Make up information
- Mix up candidate data
- Answer questions outside its scope

---

#### Section F: The Chain (Lines 146–157)

```python
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
```
**What this does:** Creates a "retriever" that searches ChromaDB and returns the **5 most similar** chunks to any given question.

```python
def make_chain(llm):
    return prompt | llm | StrOutputParser()
```
**What this does:** This is LangChain's "pipe" syntax. It creates a processing pipeline:
1. `prompt` — Fill in the template with context, question, and role instructions
2. `llm` — Send the filled prompt to the AI model
3. `StrOutputParser()` — Extract just the text string from the AI's response

The `|` operator here works like a Unix pipe — output of one step becomes input to the next.

---

#### Section G: Routes — The API Endpoints (Lines 161–283)

**`GET /`** — Serves the frontend
```python
@app.route("/")
def index():
    return send_from_directory(".", "index.html")
```
When you open `http://localhost:7860` in your browser, this sends back `index.html`.

**`POST /login`** — Handles authentication
```python
@app.route("/login", methods=["POST"])
def login():
```
Checks username/password against hardcoded credentials and returns the user's role. In a real production app, you'd use a database and hashed passwords.

**`POST /chat`** — The main endpoint (where all the magic happens)

This is called every time a user sends a message. Here's what happens step by step:

1. **Validate** — Check that the request has a valid question (not empty, not too long)
2. **Determine role** — Check if the user is HR, Staff, or General
3. **Retrieve** — Search ChromaDB for the 5 most relevant chunks: `docs = retriever.invoke(question)`
4. **Build context** — Join those chunks into one big string
5. **Web search** (HR only) — Append DuckDuckGo results to the context
6. **Invoke the chain** — Send everything to the AI model
7. **Retry logic** — If the AI fails, try again (up to 2 attempts)
8. **Return** — Send the answer back as JSON

**`GET /health`** — A simple endpoint that returns "ok" — useful for monitoring if the server is alive.

---

### 4.2 `data_handler.py` — Excel Data Helper (11 lines)

```python
import pandas as pd
df = pd.read_excel("staffrecruitment.xlsx")

def get_shortlisted():
    shortlisted = df[df["Status"] == "Shortlisted"]
    return shortlisted["Full Name"].tolist()
```
**What this does:** Reads the `staffrecruitment.xlsx` Excel file and provides functions to filter candidates by status. This is imported by `app.py` (line 5) but the actual filtering is currently handled through the RAG pipeline rather than these direct functions.

---

### 4.3 `extract_pdf.py` — PDF Preview Utility (22 lines)

**What this does:** A standalone script (not used by the main app) that extracts the first 5 pages of `hr_policy.pdf` into a text file. It was used during development to verify that the PDF could be read correctly. It has a fallback: if LangChain's loader fails, it tries PyPDF2 instead.

---

### 4.4 `hrchatbot.py` — The Original Prototype (34 lines)

**What this does:** This was the FIRST version of the chatbot — a simple terminal-based Q&A tool. No web UI, no Flask, no Groq. Just:
1. Load PDF → 2. Split into chunks → 3. Create vector DB → 4. Ask Ollama → 5. Print answer

It uses the older `RetrievalQA` chain (now deprecated in LangChain) and runs in a `while True` loop in the terminal.

---

### 4.5 `hrchatbot1.py` — Improved Prototype (73 lines)

**What this does:** The second iteration. Improvements over v1:
- Uses `RecursiveCharacterTextSplitter` (smarter splitting)
- Uses `ChatOllama` instead of plain `Ollama` (newer API)
- Uses `OllamaEmbeddings` (local embeddings instead of default)
- Persists the vector DB to disk (so you don't re-embed every time)
- Uses the modern LangChain "pipe" syntax (`prompt | llm | parser`)

Still terminal-only though. `app.py` is the final evolution that adds Flask, Groq, RBAC, and the web UI.

---

### 4.6 `index.html` — The Frontend UI

A single HTML file containing all the CSS and JavaScript for the chat interface. Features:
- **Glassmorphism design** — Semi-transparent cards with blur effects
- **Login modal** — Lets users authenticate as HR or Staff
- **Chat bubbles** — Messages with timestamps
- **Mode toggle** — Switch between Groq (fast) and Local (offline)
- **Markdown rendering** — AI responses can include bold, lists, etc.

The JavaScript in this file uses `fetch()` to make HTTP requests to your Flask backend (`/chat`, `/login`).

---

### 4.7 `Dockerfile` — Container Recipe

```dockerfile
FROM python:3.10-slim          # Start with a lightweight Python image
RUN apt-get update && ...      # Install system-level build tools
COPY requirements.txt .        # Copy dependency list
RUN pip install ... -r requirements.txt  # Install Python packages
COPY . .                       # Copy all project files
EXPOSE 7860                    # Tell Docker which port the app uses
CMD ["gunicorn", ...]          # Start the app with gunicorn (production server)
```
**What this does:** A Dockerfile is like a recipe for creating a portable "box" (container) that has everything your app needs to run. Anyone with Docker can build and run your app without installing Python, dependencies, etc.

**Gunicorn** is used instead of Flask's built-in server because Flask's server is meant for development only — it can only handle one request at a time. Gunicorn can handle many simultaneous users.

---

### 4.8 `.env` — Secrets File

```
GROQ_API_KEY=gsk_your_key_here
```
Contains your API keys. Listed in `.gitignore` so it's NEVER pushed to GitHub.

---

## 5. The Complete Request Lifecycle <a name="5-request-lifecycle"></a>

Here is exactly what happens when a Staff user types "What is the leave policy?" and hits send:

```
Step 1: BROWSER
   User types "What is the leave policy?" and clicks Send
   JavaScript creates: { question: "What is the leave policy?", mode: "groq", role: "staff" }
   fetch("http://localhost:7860/chat", { method: "POST", body: JSON.stringify(...) })

Step 2: FLASK receives the HTTP POST request at /chat

Step 3: VALIDATION
   ✓ JSON body exists
   ✓ "question" field exists
   ✓ Question is not empty
   ✓ Question is under 1000 characters
   ✓ Mode is "groq" (valid)
   ✓ Role is "staff"

Step 4: ROLE INSTRUCTIONS
   role_instructions = "You are answering a staff member. Answer based
   entirely on the provided document context."

Step 5: RETRIEVAL (RAG — the "R" part)
   a) Embed the question: "What is the leave policy?" → [0.23, -0.11, 0.67, ...]
   b) Search ChromaDB for the 5 nearest vectors
   c) Return 5 chunks of text from the PDF, e.g.:
      - Chunk from page 45: "Annual leave entitlement is 21 days..."
      - Chunk from page 46: "Sick leave is 12 days per year..."
      - Chunk from page 47: "Leave must be applied 7 days in advance..."
      - Chunk from page 12: "Probationary employees receive 10 days..."
      - Chunk from page 48: "Unused leave cannot be carried forward..."

Step 6: BUILD CONTEXT
   context = all 5 chunks joined with "\n\n"

Step 7: FILL THE PROMPT TEMPLATE
   "You are HR Assist, a friendly and helpful HR policy assistant.
   Your job is to answer questions ONLY using the context provided below.
   ...
   Role Instructions: You are answering a staff member...
   Context: Annual leave entitlement is 21 days...  [all 5 chunks]
   Employee Question: What is the leave policy?
   Answer:"

Step 8: GENERATION (RAG — the "G" part)
   Send the filled prompt to Groq's API (LLaMA 3.1)
   Groq processes it in ~1-2 seconds
   Returns: "The leave policy includes: • Annual leave: 21 days per year
   • Sick leave: 12 days per year • Leave must be applied 7 days in
   advance • Probationary employees receive 10 days..."

Step 9: PARSE & RETURN
   Flask sends back JSON: {
     "answer": "The leave policy includes: • Annual leave...",
     "mode": "groq",
     "time_taken": 1.47
   }

Step 10: BROWSER
   JavaScript receives the JSON
   Creates a chat bubble with the answer
   Adds a timestamp
   Renders any markdown formatting
```

---

## 6. Deployment & Infrastructure <a name="6-deployment"></a>

### Local Development
```
Your Computer
├── python app.py          → Flask dev server on port 7860
├── ngrok http 7860        → Creates a public URL that tunnels to your local server
└── Browser opens http://localhost:7860 or the ngrok URL
```

### Cloud Deployment (Hugging Face Spaces)
```
Hugging Face Server
├── Reads your Dockerfile
├── Builds a container with all dependencies
├── Runs gunicorn (production server) on port 7860
├── Reads GROQ_API_KEY from the "Secrets" settings (not .env)
└── Serves your app at https://huggingface.co/spaces/gebsxh/hr-chatbot
```

---

## 7. Glossary <a name="7-glossary"></a>

| Term | Meaning |
|------|---------|
| **LLM** | Large Language Model — AI that understands/generates text |
| **RAG** | Retrieval-Augmented Generation — search first, then answer |
| **Embedding** | Converting text into a list of numbers for similarity search |
| **Vector DB** | Database optimized for searching by number-similarity |
| **ChromaDB** | The specific vector database library used in this project |
| **LangChain** | Python framework for building LLM applications |
| **Flask** | Python web framework (handles HTTP requests) |
| **CORS** | Cross-Origin Resource Sharing — allows frontend to talk to backend |
| **Groq** | Cloud service that runs LLMs very fast on special hardware |
| **Ollama** | Software that runs LLMs locally on your own computer |
| **Gunicorn** | Production-grade Python web server (replaces Flask's dev server) |
| **Docker** | Tool for packaging apps into portable containers |
| **ngrok** | Tool that exposes your local server to the internet |
| **API Key** | A secret password that authenticates you with a cloud service |
| **RBAC** | Role-Based Access Control — different permissions for different users |
| **Token** | A unit of text for LLMs (~4 characters = 1 token). Models have token limits. |
| **Temperature** | Controls AI randomness. 0 = deterministic, 1 = creative/random |
| **Seed** | A fixed number ensuring reproducible AI outputs |
| **Context Window** | Maximum amount of text an LLM can process at once |
