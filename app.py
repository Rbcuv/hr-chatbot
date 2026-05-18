from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.document_loaders import PyPDFLoader
from data_handler import get_shortlisted, get_selected
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
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

PERSIST_DIR = "./chroma_db_v2"
PDF_PATHS   = ["hr_policy.pdf", "staffrecruitment.pdf"]

# Initialize web search tool
from duckduckgo_search import DDGS

def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No web results found."
        return "\n\n".join(
            f"Source: {r.get('href','')}\nTitle: {r.get('title','')}\n{r.get('body','')}"
            for r in results
        )
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return "Web search unavailable."

# ── Embeddings & Vector DB ─────────────────────────────────────────────────────
def load_or_build_db():
    """Load existing ChromaDB or build it fresh from the PDF."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        raise RuntimeError("Hugging Face embeddings could not be initialized.") from e

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        logger.info("Loading existing vector DB...")
        try:
            db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            logger.info("Vector DB loaded successfully.")
            return db
        except Exception as e:
            logger.warning(f"Failed to load existing DB ({e}), rebuilding from PDF...")

    documents = []
    for pdf_path in PDF_PATHS:
        if os.path.exists(pdf_path):
            logger.info(f"Loading {pdf_path}...")
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        else:
            logger.warning(f"PDF not found at '{pdf_path}'")
            
    if not documents:
        raise ValueError("No PDFs loaded or they contain no pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split PDFs into {len(texts)} chunks.")

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
    num_ctx=4096,
    seed=42,
)

# ── Prompt Template ────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""You are HR Assist, a friendly and helpful HR policy assistant.
Your job is to answer questions ONLY using the context provided below.

Rules:
- Answer clearly and concisely in a friendly tone.
- When reading applicant/candidate data, each line represents one person.
- You MUST strictly match the status requested in the question.

- If the question asks for "Shortlisted":
  ONLY include candidates whose status is EXACTLY "Shortlisted".
  DO NOT include:
  - Selected
  - Rejected
  - On Hold
  - Under Review

- Do NOT mix data between different candidates.
- Do NOT assume or guess.
- If no exact match is found, say:
  "No shortlisted candidates found in the document."

- Use ONLY the provided context.
- Format answers using bullet points when listing names.
- For simple greetings (e.g., "hi", "hello", "how are you"), respond naturally and casually without checking the policy or stating that the information is missing.
- If the answer to a policy question is not in the context, politely state that the current documents do not seem to contain information on that specific topic. 
- You can suggest related topics if they are mentioned in the context.
- Do NOT make up or guess any information. Use ONLY the provided context.
- Format your answer neatly. Use bullet points if listing multiple items.

Role Instructions: {role_instructions}

Context:
{context}

Employee Question: {question}

Answer:""")

# ── Retriever ──────────────────────────────────────────────────────────────────
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},   # fetch 5 chunks for better table coverage
)

# ── QA Chains ──────────────────────────────────────────────────────────────────
def make_chain(llm):
    return prompt | llm | StrOutputParser()

groq_chain  = make_chain(groq_llm)
local_chain = make_chain(local_llm)

logger.info("✅ Flask server ready! Groq and Local chains initialized.")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request"}), 400
    
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()

    if username == "HR1" and password == "hr@2025":
        return jsonify({"role": "hr", "token": "HR1_TOKEN"})
    elif username == "HR2" and password == "hrm@2025":
        return jsonify({"role": "hr", "token": "HR2_TOKEN"})
    elif username == "STAFF1" and password == "staff1@2025":
        return jsonify({"role": "staff", "token": "STAFF1_TOKEN"})
    elif username == "STAFF2" and password == "staff2@2025":
        return jsonify({"role": "staff", "token": "STAFF2_TOKEN"})
    else:
        return jsonify({"error": "Invalid username or password"}), 401

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

    role = str(data.get("role", "general")).strip().lower()

    # Determine role-specific instructions
    if role == "hr":
        role_instructions = "You are answering an HR representative. Use both the provided document context and the external web search context to give a comprehensive answer."
    elif role == "staff":
        role_instructions = "You are answering a staff member. Answer based entirely on the provided document context."
    else:
        # General access is restricted
        role_instructions = "You are answering a general user. ONLY answer questions related to staff recruitment, company policy, recruitment policy, and leave policy. Refuse any other questions politely, stating that general access is limited to these topics."

    # ── Invoke chain with retry logic ───────────────────────────────────────────
    max_retries = 2
    last_error  = None

    for attempt in range(1, max_retries + 1):
        try:
            start = time.time()
            
            # Retrieve from vector DB
            docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # If HR, append web search results
            if role == "hr":
                try:
                    logger.info(f"Executing web search for HR query: {question}")
                    web_results = web_search(question)
                    context += f"\n\nExternal Web Search Results:\n{web_results}"
                except Exception as search_err:
                    logger.warning(f"Web search failed: {search_err}")

            if mode == "local":
                logger.info(f"[LOCAL] Attempt {attempt} | Q: {question[:80]}...")
                answer = local_chain.invoke({
                    "context": context,
                    "question": question,
                    "role_instructions": role_instructions
                })
            else:
                logger.info(f"[GROQ] Attempt {attempt} | Q: {question[:80]}...")
                answer = groq_chain.invoke({
                    "context": context,
                    "question": question,
                    "role_instructions": role_instructions
                })

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
        "pdf":     PDF_PATHS,
    })



if __name__ == "__main__":
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host="0.0.0.0", port=port)
