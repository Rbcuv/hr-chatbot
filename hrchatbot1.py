from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

PERSIST_DIR = "./chroma_db"
PDF_PATH = "hr_policy.pdf"
PDF_PATH = "staffrecruitment.pdf"

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
texts = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    print("Loading existing vector DB (skipping re-embedding)...")
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    print(f"Embedding {len(texts)} chunks... (one-time, may take a few minutes)")
    db = Chroma.from_documents(
        texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print("Done! DB saved to disk.")

# Replace "phi3" below with whatever name shows in 'ollama list'
llm = ChatOllama(
    model="phi3",
    temperature=0,
    num_ctx=2048,
)

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below. Be concise.

Context:
{context}

Question: {question}
""")

retriever = db.as_retriever(search_kwargs={"k": 10})

qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\nReady! Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ").strip()
    if not query:
        continue
    if query.lower() == "exit":
        break
    print("Thinking...")
    print("Answer:", qa_chain.invoke(query))
    print()