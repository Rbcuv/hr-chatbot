from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# this lines is used to load pdf
loader = PyPDFLoader("hr_policy.pdf")
documents = loader.load()

#split the text in the pdf into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# to create vector database
db = Chroma.from_documents(texts)

# this line is to load ollama
llm = Ollama(model="phi3")

# to create QA system 
qa = RetrievalQA.from_chain_type(
    llm = llm,
    retriever=db.as_retriever()
)

# to start a chat loop
while True:
    query = input("Ask a question:")
    if query.lower() == "exit":
        break

    print("Answer:",qa.run(query))