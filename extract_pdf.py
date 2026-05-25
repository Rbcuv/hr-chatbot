import sys

try:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader("hr_policy.pdf")
    pages = loader.load()
    with open("pdf_summary.txt", "w", encoding="utf-8") as f:
        for i in range(min(5, len(pages))):
            f.write(pages[i].page_content + "\n\n")
    print("Successfully extracted with PyPDFLoader")
except Exception as e:
    print(f"Failed with PyPDFLoader: {e}")
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader("hr_policy.pdf")
        with open("pdf_summary.txt", "w", encoding="utf-8") as f:
            for i in range(min(5, len(reader.pages))):
                f.write(reader.pages[i].extract_text() + "\n\n")
        print("Successfully extracted with PyPDF2")
    except Exception as e2:
        print(f"Failed with PyPDF2: {e2}")
