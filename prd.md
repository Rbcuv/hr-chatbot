# Product Requirements Document (PRD): HR Assist

## 1. Product Overview
**Product Name:** HR Assist (formerly HR Buddy)
**Product Vision:** To provide an intelligent, secure, and user-friendly conversational AI assistant that helps employees and HR personnel navigate company policies, recruitment guidelines, and general queries efficiently.

## 2. Target Audience & Roles (RBAC)
The application implements Role-Based Access Control (RBAC) to ensure data privacy and appropriate feature access.

1. **HR (Admin/Elevated):**
   - **Access:** Full access to all internal documents (`hr_policy.pdf`, `staffrecruitment.pdf`).
   - **Special Capabilities:** Internet-enabled search capabilities to answer external or broader queries not covered in the internal documents.
   - **Authentication:** Requires secure login.

2. **Staff (Internal Employee):**
   - **Access:** Restricted strictly to querying ingested internal policy and recruitment documents.
   - **Authentication:** Requires secure login.

3. **General (Guest):**
   - **Access:** Limited-access guest experience for general, non-confidential queries.
   - **Authentication:** No login required.

## 3. Key Features

### 3.1 Chat Interface & User Experience
- **Modern UI:** Premium "Glassmorphism" aesthetic with a responsive and dynamic design.
- **Markdown Support:** Rich text formatting in chat responses (bolding, lists, code blocks).
- **Source Grounding & Citations:** AI responses based on internal documents include page-level citations to ensure accuracy and build trust.
- **Timestamps:** Each message in the chat includes a timestamp for tracking conversation flow.
- **Session Management:** Secure login UI managing session-based access levels.

### 3.2 AI & Search Capabilities
- **Dual-Mode AI:** 
  - **Online Mode:** Powered by high-speed cloud LLMs via the **Groq API** (e.g., LLaMA 3) for rapid responses.
  - **Offline Mode:** Fallback to local execution using **Ollama**, ensuring the system remains functional even without internet connectivity or during API outages.
- **Vector Search (RAG):** Uses **ChromaDB** to store document embeddings. When a user asks a question, the system retrieves the most relevant chunks from the ingested PDFs to generate an accurate answer.
- **Document Ingestion:** Currently supports reasoning over large, complex PDFs (`hr_policy.pdf`, `staffrecruitment.pdf`).

### 3.3 Infrastructure & Deployment
- **Backend:** Built with **Python / Flask**, serving as a robust WSGI application.
- **Cloud-Ready:** Containerized with **Docker** and configured for easy deployment on platforms like **Hugging Face Spaces**.
- **Environment Management:** Utilizes `python-dotenv` for secure environment variable management (e.g., `GROQ_API_KEY`).

## 4. Technical Architecture

* **Frontend:** Vanilla HTML, CSS, JavaScript.
* **Backend Framework:** Flask (`app.py`).
* **LLM Provider:** Groq (Cloud) / Ollama (Local).
* **Embeddings & Vector Store:** Hugging Face Embeddings / ChromaDB.
* **Hosting:** Local development via Ngrok / Cloud deployment via Hugging Face Docker Space.

## 5. Security & Privacy
- **Environment Variables:** API keys and sensitive configurations are kept out of source control.
- **Data Segregation:** The RAG pipeline ensures that the LLM only uses provided document context for Staff queries, minimizing hallucinations and preventing leakage of external information into internal policy discussions.

## 6. Future Scope
- **Multi-modal Support:** Allowing users to upload their own temporary documents for the AI to analyze.
- **Conversational Memory:** Improving multi-turn conversations by retaining chat history context across deeper sessions.
- **Analytics Dashboard:** Providing HR with insights into the most frequently asked questions to help improve underlying policy documentation.
