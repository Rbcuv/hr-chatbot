from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
import time

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

PERSIST_DIR = "./chroma_db"
PDF_PATH    = "hr_policy.pdf"

# ── Embeddings & Vector DB ─────────────────────────────────────────────────────
def load_or_build_db():
    """Load existing ChromaDB or build it fresh from the PDF."""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        raise RuntimeError("Ollama embeddings could not be initialized. Is Ollama running?") from e

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        logger.info("Loading existing vector DB...")
        try:
            db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            logger.info("Vector DB loaded successfully.")
            return db
        except Exception as e:
            logger.warning(f"Failed to load existing DB ({e}), rebuilding from PDF...")

    # Build fresh
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"HR policy PDF not found at '{PDF_PATH}'")

    logger.info("Building vector DB from PDF (one-time operation)...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    if not documents:
        raise ValueError("PDF loaded but contains no pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split PDF into {len(texts)} chunks.")

    db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=PERSIST_DIR)
    logger.info("Vector DB built and saved.")
    return db

db = load_or_build_db()

# ── LLM Models ─────────────────────────────────────────────────────────────────

# Groq Cloud — fast (1-3 seconds), seed+temperature=0 = consistent answers
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=1024,
    model_kwargs={"seed": 42},
)

# Local Ollama — fully offline (2-3 minutes on CPU)
local_llm = ChatOllama(
    model="phi3",
    temperature=0,
    num_ctx=2048,
    seed=42,
)

# ── Prompt Template ────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""You are HR Buddy, a friendly and helpful HR policy assistant.
Your job is to answer questions ONLY using the context provided below from the company's HR policy document.

Rules:
- Answer clearly and concisely in a friendly tone.
- If the answer is not in the context, politely state that the current HR Policy Manual (2025) does not seem to contain information on that specific topic. 
- You can suggest related topics if they are mentioned in the context (e.g., if they ask about remote work and it's not there, you could mention the official office hours).
- Do NOT make up or guess any information. Use ONLY the provided context.
- Format your answer neatly. Use bullet points if listing multiple items.

Context from HR Policy:
{context}

Employee Question: {question}

Answer:""")

# ── Retriever ──────────────────────────────────────────────────────────────────
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6},   # fetch 6 chunks for better coverage
)

# ── QA Chains ──────────────────────────────────────────────────────────────────
def make_chain(llm):
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

groq_chain  = make_chain(groq_llm)
local_chain = make_chain(local_llm)

logger.info("✅ Flask server ready! Groq and Local chains initialized.")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)  # silent=True prevents crash on bad JSON

    # ── Validate input ──────────────────────────────────────────────────────────
    if not data:
        return jsonify({"error": "Invalid or missing JSON body."}), 400
    if "question" not in data:
        return jsonify({"error": "Missing 'question' field in request."}), 400

    question = str(data["question"]).strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400
    if len(question) > 1000:
        return jsonify({"error": "Question is too long (max 1000 characters)."}), 400

    mode = data.get("mode", "groq").strip().lower()
    if mode not in ("groq", "local"):
        mode = "groq"  # fallback to groq for unknown modes

    # ── Invoke chain with retry logic ───────────────────────────────────────────
    max_retries = 2
    last_error  = None

    for attempt in range(1, max_retries + 1):
        try:
            start = time.time()
            if mode == "local":
                logger.info(f"[LOCAL] Attempt {attempt} | Q: {question[:80]}...")
                answer = local_chain.invoke(question)
            else:
                logger.info(f"[GROQ] Attempt {attempt} | Q: {question[:80]}...")
                answer = groq_chain.invoke(question)

            elapsed = round(time.time() - start, 2)
            logger.info(f"[{mode.upper()}] Answered in {elapsed}s")

            # Sanity check — make sure we actually got a string response
            if not answer or not isinstance(answer, str) or answer.strip() == "":
                raise ValueError("Empty or invalid response from AI model.")

            return jsonify({"answer": answer.strip(), "mode": mode, "time_taken": elapsed})

        except Exception as e:
            last_error = e
            logger.warning(f"[{mode.upper()}] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(1)  # brief pause before retry

    # All retries exhausted
    logger.error(f"[{mode.upper()}] All {max_retries} attempts failed. Last error: {last_error}")
    return jsonify({
        "error": "The AI could not generate a response. Please try again in a moment.",
        "detail": str(last_error)
    }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint — useful to verify server is up."""
    return jsonify({
        "status":  "ok",
        "modes":   ["groq", "local"],
        "db_path": PERSIST_DIR,
        "pdf":     PDF_PATH,
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)