import os

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if 'def load_or_build_db():' in line:
        # Skip the docstring line if it's the next one
        pass
    if 'try:' in line and 'embeddings = HuggingFaceEmbeddings' not in line:
        # Check if the next line is an except to avoid double-adding or messing up
        # This is a bit risky, let's just do a clean replace of the whole function
        pass

# Actually, let's just rewrite the whole load_or_build_db function
start_line = -1
end_line = -1
for i, line in enumerate(lines):
    if 'def load_or_build_db():' in line:
        start_line = i
    if start_line != -1 and 'return db' in line:
        end_line = i + 1
        break

if start_line != -1 and end_line != -1:
    function_code = [
        'def load_or_build_db():\n',
        '    """Load existing ChromaDB or build it fresh from the PDF."""\n',
        '    try:\n',
        '        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")\n',
        '    except Exception as e:\n',
        '        logger.error(f"Failed to initialize embeddings: {e}")\n',
        '        raise RuntimeError("Hugging Face embeddings could not be initialized.") from e\n',
        '\n',
        '    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):\n',
        '        logger.info("Loading existing vector DB...")\n',
        '        try:\n',
        '            db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)\n',
        '            logger.info("Vector DB loaded successfully.")\n',
        '            return db\n',
        '        except Exception as e:\n',
        '            logger.warning(f"Failed to load existing DB ({e}), rebuilding from PDF...")\n',
        '\n',
        '    if not os.path.exists(PDF_PATH):\n',
        '        raise FileNotFoundError(f"HR policy PDF not found at \'{PDF_PATH}\'")\n',
        '\n',
        '    logger.info("Building vector DB from PDF (one-time operation)...")\n',
        '    loader = PyPDFLoader(PDF_PATH)\n',
        '    documents = loader.load()\n',
        '    if not documents:\n',
        '        raise ValueError("PDF loaded but contains no pages.")\n',
        '\n',
        '    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)\n',
        '    texts = text_splitter.split_documents(documents)\n',
        '    logger.info(f"Split PDF into {len(texts)} chunks.")\n',
        '\n',
        '    db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=PERSIST_DIR)\n',
        '    logger.info("Vector DB built and saved.")\n',
        '    return db\n'
    ]
    lines[start_line:end_line] = function_code
    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Successfully patched app.py")
else:
    print("Could not find function to patch")
