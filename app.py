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
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import logging
import time

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

import json

PERSIST_DIR = "./chroma_db_v2"
PDF_METADATA_FILE = "metadata_pdfs.json"

TOKEN_TO_USER = {}
USERS_FILE = "users.json"
users_db = []

def load_users():
    global users_db, TOKEN_TO_USER
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                users_db = json.load(f)
            logger.info("Loaded users DB successfully.")
        except Exception as e:
            logger.error(f"Failed to load users DB: {e}")
            users_db = []
    
    if not users_db:
        users_db = [
            {"username": "admin", "password": "admin@2025", "role": "admin", "token": "ADMIN_TOKEN"},
            {"username": "Hr1", "password": "hr@2025", "role": "hr", "token": "HR1_TOKEN"},
            {"username": "Hrm2", "password": "hrm@2025", "role": "hr", "token": "HR2_TOKEN"},
            {"username": "Staff1", "password": "staff1@2025", "role": "staff", "token": "STAFF1_TOKEN"},
            {"username": "Staff2", "password": "staff2@2025", "role": "staff", "token": "STAFF2_TOKEN"}
        ]
        save_users()

    TOKEN_TO_USER.clear()
    for u in users_db:
        token = u.get("token") or f"{u['username'].upper()}_TOKEN"
        TOKEN_TO_USER[token] = u["username"].upper()

def save_users():
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users_db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save users DB: {e}")

load_users()

SESSION_HISTORY_FILE = "hr_memories.json"
hr_memories = {}

def load_hr_memories():
    global hr_memories
    if os.path.exists(SESSION_HISTORY_FILE):
        try:
            with open(SESSION_HISTORY_FILE, "r", encoding="utf-8") as f:
                hr_memories = json.load(f)
                logger.info("Loaded HR long-term memories successfully.")
        except Exception as e:
            logger.error(f"Failed to load HR memories: {e}")
            hr_memories = {}
    else:
        hr_memories = {}

def save_hr_memories():
    try:
        with open(SESSION_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(hr_memories, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save HR memories: {e}")

load_hr_memories()

# In-memory short-term memories for staff and general
short_term_memories = {}

def get_pdf_paths():
    if os.path.exists(PDF_METADATA_FILE):
        try:
            with open(PDF_METADATA_FILE, "r") as f:
                paths = json.load(f)
                if isinstance(paths, list):
                    # Ensure default templates are always in the list
                    for default in ["hr_policy.pdf", "staffrecruitment.pdf"]:
                        if default not in paths:
                            paths.append(default)
                    return paths
        except Exception as e:
            logger.warning(f"Failed to read {PDF_METADATA_FILE}: {e}")
    return ["hr_policy.pdf", "staffrecruitment.pdf"]

def save_pdf_paths(paths):
    try:
        with open(PDF_METADATA_FILE, "w") as f:
            json.dump(paths, f)
    except Exception as e:
        logger.error(f"Failed to save {PDF_METADATA_FILE}: {e}")

PDF_PATHS = get_pdf_paths()

# Initialize web search tool
from ddgs import DDGS

def _bing_search(query: str, max_results: int = 4) -> list[dict[str, str]]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def _search_with_retries(query: str, max_results: int = 4, attempts: int = 3) -> list[dict[str, str]]:
    for attempt in range(1, attempts + 1):
        try:
            results = _bing_search(query, max_results=max_results)
            logger.info(f"Bing search attempt {attempt} for query='{query}' returned {len(results)} results")
            if results:
                return results
        except Exception as e:
            logger.warning(f"Bing search attempt {attempt} failed for query='{query}': {e}")
    return []


def web_search(query: str) -> str:
    try:
        results = _search_with_retries(query, max_results=4)
        if not results:
            fallback_query = query
            if "latest" not in query.lower():
                fallback_query = f"latest {query}"
            if fallback_query != query:
                logger.info(f"Primary search empty; trying fallback query='{fallback_query}'")
                results = _search_with_retries(fallback_query, max_results=4)
        if not results and "news" not in query.lower():
            fallback_query = f"{query} news"
            logger.info(f"Secondary fallback search for query='{fallback_query}'")
            results = _search_with_retries(fallback_query, max_results=4)

        logger.info(f"Web search returned {len(results)} results for original query: {query}")
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

def rebuild_db():
    global db, retriever, PDF_PATHS
    logger.info("Starting dynamic vector DB rebuild...")
    
    # 1. Close/delete current collection to release locks
    if 'db' in globals() and db is not None:
        try:
            db.delete_collection()
            logger.info("Deleted old Chroma collection.")
        except Exception as e:
            logger.warning(f"Could not delete old Chroma collection: {e}")
            
    # 2. Re-read dynamic PDF paths list
    current_pdfs = get_pdf_paths()
    PDF_PATHS = current_pdfs
    
    # 3. Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Load all active PDFs
    documents = []
    for pdf_path in current_pdfs:
        if os.path.exists(pdf_path):
            logger.info(f"Loading {pdf_path} for rebuild...")
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        else:
            logger.warning(f"PDF not found at '{pdf_path}'")
            
    if not documents:
        raise ValueError("No PDFs loaded or they contain no pages.")
        
    # 5. Split updated texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split updated PDFs into {len(texts)} chunks.")
    
    # 6. Rebuild Chroma DB collection
    db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=PERSIST_DIR)
    
    # 7. Hot-swap active retriever reference
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    logger.info("Dynamic vector DB rebuild completed successfully!")

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
Your job is to answer questions using the context provided below.

The context may include:
- relevant policy documents loaded from the provided PDFs
- external web search results for HR users

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
- If no exact match is found in the context, say:
  "No shortlisted candidates found in the document."

- Use only the provided context, including external web search results when they are available.
- Format answers using bullet points when listing names.
- For simple greetings (e.g., "hi", "hello", "how are you"), respond naturally and casually without checking the policy or stating that the information is missing.
- If the answer to a policy question is not in the context, politely state that the current context does not contain information on that specific topic.
- You can suggest related topics if they are mentioned in the context.
- Do NOT make up or guess any information. Use only the provided context.
- Format your answer neatly. Use bullet points if listing multiple items.

Role Instructions: {role_instructions}

Context:
{context}

Conversation History:
{chat_history}

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
    return send_file("index.html")

@app.route("/background.jpg")
def background():
    return send_file("background.jpg")

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request"}), 400
    
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()

    # Search for user matching credentials
    matched_user = None
    for u in users_db:
        if u["username"].lower() == username.lower() and u["password"] == password:
            matched_user = u
            break

    if matched_user:
        token = matched_user.get("token") or f"{matched_user['username'].upper()}_TOKEN"
        return jsonify({"role": matched_user["role"], "token": token})
    else:
        return jsonify({"error": "Invalid username or password"}), 401

def format_chat_history(history, max_turns=6):
    recent_messages = history[-(max_turns * 2):]
    formatted = []
    for msg in recent_messages:
        role = "Human" if msg["role"] == "user" else "AI"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted) if formatted else "No previous conversation history."

@app.route("/chat_history", methods=["GET"])
def get_chat_history():
    token = request.headers.get("Authorization")
    session_id = request.args.get("session_id", "").strip()
    
    user = None
    if token:
        user = next((u for u in users_db if u.get("token") == token or f"{u['username'].upper()}_TOKEN" == token), None)
        
    if user:
        username_upper = user["username"].upper()
        if user["role"] == "hr":
            history = hr_memories.get(username_upper, [])
            return jsonify({"history": history})
        else:
            history = short_term_memories.get(username_upper, [])
            return jsonify({"history": history})
    elif session_id:
        history = short_term_memories.get(session_id, [])
        return jsonify({"history": history})
        
    return jsonify({"history": []})

@app.route("/clear_history", methods=["POST"])
def clear_history():
    token = request.headers.get("Authorization")
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "").strip()
    
    user = None
    if token:
        user = next((u for u in users_db if u.get("token") == token or f"{u['username'].upper()}_TOKEN" == token), None)
    
    if user:
        username_upper = user["username"].upper()
        if user["role"] == "hr":
            hr_memories[username_upper] = []
            save_hr_memories()
            logger.info(f"Cleared long-term memory for HR user {username_upper}")
        else:
            if username_upper in short_term_memories:
                short_term_memories[username_upper] = []
                logger.info(f"Cleared short-term memory for staff {username_upper}")
    elif session_id:
        if session_id in short_term_memories:
            short_term_memories[session_id] = []
            logger.info(f"Cleared short-term memory for session {session_id}")
            
    return jsonify({"status": "success", "message": "Chat history cleared successfully."})

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
    internet_val = data.get("internet", True)
    internet = internet_val in (True, "true", "True", 1, "1")
    logger.info(f"Role received: {role}, Mode: {mode}, Internet: {internet}")

    token = request.headers.get("Authorization")
    session_id = data.get("session_id", "default_session").strip()
    
    user = None
    if token:
        user = next((u for u in users_db if u.get("token") == token or f"{u['username'].upper()}_TOKEN" == token), None)

    # Retrieve correct conversation history
    if user and user["role"] == "hr":
        username_upper = user["username"].upper()
        if username_upper not in hr_memories:
            hr_memories[username_upper] = []
        history = hr_memories[username_upper]
    elif user and user["role"] == "staff":
        username_upper = user["username"].upper()
        if username_upper not in short_term_memories:
            short_term_memories[username_upper] = []
        history = short_term_memories[username_upper]
    else:
        if session_id not in short_term_memories:
            short_term_memories[session_id] = []
        history = short_term_memories[session_id]

    formatted_history = format_chat_history(history)

    # Determine role-specific instructions
    if role == "hr":
        if internet:
            role_instructions = (
                "You are answering an HR representative. Use both the provided document context and the external web search context to give a comprehensive answer. "
                "If the question asks for recent HR news, updates, or current events, prefer the external web search results and clearly summarize them."
            )
        else:
            role_instructions = (
                "You are answering an HR representative. Answer based entirely on the provided document context."
            )
    elif role == "staff":
        role_instructions = "You are answering a staff member. Answer based entirely on the provided document context."
    else:
        role_instructions = "You are answering a general user. ONLY answer questions related to staff recruitment, company policy, recruitment policy, and leave policy. Refuse any other questions politely, stating that general access is limited to these topics."

    # ── Invoke chain with retry logic ───────────────────────────────────────────
    max_retries = 2
    last_error  = None

    for attempt in range(1, max_retries + 1):
        try:
            start = time.time()
            
            docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # Retrieve candidate data from staffrecruitment.xlsx dynamically
            try:
                from data_handler import get_all_candidates
                candidates = get_all_candidates()
                if candidates:
                    cand_context = "Dynamic Recruitment Candidates List (from Excel):\n"
                    for c in candidates:
                        cand_id = c.get('App. ID') or c.get('Unnamed: 0') or ''
                        name = c.get('Full Name') or c.get('Unnamed: 1') or ''
                        gender = c.get('Gender') or c.get('Unnamed: 3') or ''
                        dept = c.get('Department') or c.get('Unnamed: 4') or ''
                        pos = c.get('Position Applied') or c.get('Unnamed: 5') or ''
                        exp = c.get('Experience (Yrs)') or c.get('Unnamed: 7') or ''
                        status = c.get('Status') or c.get('Unnamed: 10') or ''
                        remarks = c.get('Remarks') or ''
                        # Skip header rows if present
                        if name.lower() in ["full name", "name"] or status.lower() == "status":
                            continue
                        cand_context += (
                            f"- Candidate ID: {cand_id} | Name: {name} | Gender: {gender} | "
                            f"Department: {dept} | Position Applied: {pos} | Experience: {exp} Yrs | "
                            f"Status: {status} | Remarks: {remarks}\n"
                        )
                    context += f"\n\n{cand_context}"
            except Exception as e:
                logger.error(f"Error appending dynamic candidates context: {e}")
            
            if role == "hr" and internet:
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
                    "chat_history": formatted_history,
                    "question": question,
                    "role_instructions": role_instructions
                })
            else:
                logger.info(f"[GROQ] Attempt {attempt} | Q: {question[:80]}...")
                answer = groq_chain.invoke({
                    "context": context,
                    "chat_history": formatted_history,
                    "question": question,
                    "role_instructions": role_instructions
                })

            elapsed = round(time.time() - start, 2)
            logger.info(f"[{mode.upper()}] Answered in {elapsed}s")

            if not answer or not isinstance(answer, str) or answer.strip() == "":
                raise ValueError("Empty or invalid response from AI model.")

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer.strip()})
            if user and user["role"] == "hr":
                save_hr_memories()

            return jsonify({"answer": answer.strip(), "mode": mode, "time_taken": elapsed})

        except Exception as e:
            last_error = e
            logger.warning(f"[{mode.upper()}] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(1)

    logger.error(f"[{mode.upper()}] All {max_retries} attempts failed. Last error: {last_error}")
    return jsonify({
        "error": "The AI could not generate a response. Please try again in a moment.",
        "detail": str(last_error)
    }), 500

# ── Admin API Endpoints ────────────────────────────────────────────────────────
def check_admin_token(req):
    token = req.headers.get("Authorization")
    if not token:
        return False
    user = next((u for u in users_db if (u.get("token") == token or f"{u['username'].upper()}_TOKEN" == token) and u["role"] == "admin"), None)
    return user is not None

@app.route("/api/admin/users", methods=["GET"])
def admin_get_users():
    if not check_admin_token(request):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"users": users_db})

@app.route("/api/admin/users", methods=["POST"])
def admin_add_user():
    if not check_admin_token(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400
        
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()
    role = str(data.get("role", "")).strip().lower()
    
    if not username or not password or not role:
        return jsonify({"error": "Username, password, and role are required."}), 400
        
    if role not in ["admin", "hr", "staff"]:
        return jsonify({"error": "Invalid role. Role must be admin, hr, or staff."}), 400
        
    if any(u["username"].lower() == username.lower() for u in users_db):
        return jsonify({"error": f"User '{username}' already exists."}), 400
        
    token = f"{username.upper()}_TOKEN"
    new_user = {
        "username": username,
        "password": password,
        "role": role,
        "token": token
    }
    users_db.append(new_user)
    save_users()
    load_users()
    
    return jsonify({"status": "success", "message": f"User '{username}' added successfully.", "user": new_user})

@app.route("/api/admin/users/<string:old_username>", methods=["PUT"])
def admin_update_user(old_username):
    if not check_admin_token(request):
        return jsonify({"error": "Unauthorized"}), 401
        
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400
        
    new_username = str(data.get("username", "")).strip()
    new_password = str(data.get("password", "")).strip()
    new_role = str(data.get("role", "")).strip().lower()
    
    user_idx = -1
    for idx, u in enumerate(users_db):
        if u["username"].lower() == old_username.lower():
            user_idx = idx
            break
            
    if user_idx == -1:
        return jsonify({"error": f"User '{old_username}' not found."}), 404
        
    if not new_username or not new_password or not new_role:
        return jsonify({"error": "Username, password, and role are required."}), 400
        
    if new_role not in ["admin", "hr", "staff"]:
        return jsonify({"error": "Invalid role. Role must be admin, hr, or staff."}), 400
        
    if new_username.lower() != old_username.lower():
        if any(u["username"].lower() == new_username.lower() for u in users_db):
            return jsonify({"error": f"Username '{new_username}' is already taken."}), 400
            
    users_db[user_idx]["username"] = new_username
    users_db[user_idx]["password"] = new_password
    users_db[user_idx]["role"] = new_role
    users_db[user_idx]["token"] = f"{new_username.upper()}_TOKEN"
    
    save_users()
    load_users()
    
    return jsonify({"status": "success", "message": "User updated successfully."})

@app.route("/api/admin/users/<string:username>", methods=["DELETE"])
def admin_delete_user(username):
    if not check_admin_token(request):
        return jsonify({"error": "Unauthorized"}), 401
        
    user_idx = -1
    for idx, u in enumerate(users_db):
        if u["username"].lower() == username.lower():
            user_idx = idx
            break
            
    if user_idx == -1:
        return jsonify({"error": f"User '{username}' not found."}), 404
        
    token = request.headers.get("Authorization")
    deleting_user = users_db[user_idx]
    if deleting_user.get("token") == token or f"{deleting_user['username'].upper()}_TOKEN" == token:
        return jsonify({"error": "You cannot delete your own admin account while logged in."}), 400
        
    del users_db[user_idx]
    save_users()
    load_users()
    
    return jsonify({"status": "success", "message": f"User '{username}' deleted successfully."})

@app.route("/api/admin/recruitment", methods=["GET"])
def admin_get_recruitment():
    if not check_admin_token(request):
        return jsonify({"error": "Unauthorized"}), 401
        
    from data_handler import get_all_candidates
    candidates = get_all_candidates()
    return jsonify({"recruitment": candidates})

from werkzeug.utils import secure_filename

@app.route("/active_policies", methods=["GET"])
def active_policies():
    return jsonify({
        "pdf_list": get_pdf_paths()
    })

@app.route("/upload_policy", methods=["POST"])
def upload_policy():
    # 1. Authenticate token
    token = request.headers.get("Authorization")
    if not token or token not in ["HR1_TOKEN", "HR2_TOKEN"]:
        return jsonify({"error": "Unauthorized"}), 401
        
    # 2. Check file
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
        
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400
        
    try:
        # Secure and save filename
        filename = secure_filename(file.filename)
        
        # Save to workspace root
        file.save(filename)
        logger.info(f"Saved uploaded policy to {filename}")
        
        # Add to dynamic PDF paths list
        current_pdfs = get_pdf_paths()
        if filename not in current_pdfs:
            current_pdfs.append(filename)
            save_pdf_paths(current_pdfs)
            logger.info(f"Registered {filename} in dynamic PDF list.")
            
        # Dynamically rebuild vector DB
        rebuild_db()
        
        return jsonify({
            "status": "success",
            "message": f"Successfully uploaded and indexed '{filename}'!",
            "pdf_list": current_pdfs
        })
    except Exception as e:
        logger.error(f"Error uploading policy: {e}")
        return jsonify({"error": f"Failed to upload policy: {str(e)}"}), 500

@app.route("/delete_policy", methods=["POST"])
def delete_policy():
    # 1. Authenticate token
    token = request.headers.get("Authorization")
    if not token or token not in ["HR1_TOKEN", "HR2_TOKEN"]:
        return jsonify({"error": "Unauthorized"}), 401
        
    data = request.get_json(silent=True)
    if not data or "filename" not in data:
        return jsonify({"error": "Missing filename"}), 400
        
    filename = secure_filename(os.path.basename(str(data["filename"]).strip()))
    
    # Defaults are protected
    if filename in ["hr_policy.pdf", "staffrecruitment.pdf"]:
        return jsonify({"error": f"Default document '{filename}' cannot be deleted, but can be overwritten by uploading a new version."}), 400
        
    current_pdfs = get_pdf_paths()
    if filename not in current_pdfs:
        return jsonify({"error": f"Document '{filename}' is not currently active."}), 404
        
    try:
        # Remove from dynamic list
        current_pdfs.remove(filename)
        save_pdf_paths(current_pdfs)
        
        # Delete file if exists
        if os.path.exists(filename):
            os.remove(filename)
            logger.info(f"Deleted file {filename} from disk.")
            
        # Dynamically rebuild vector DB
        rebuild_db()
        
        return jsonify({
            "status": "success",
            "message": f"Successfully retired and un-indexed '{filename}'!",
            "pdf_list": current_pdfs
        })
    except Exception as e:
        logger.error(f"Error deleting policy: {e}")
        return jsonify({"error": f"Failed to delete policy: {str(e)}"}), 500

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
