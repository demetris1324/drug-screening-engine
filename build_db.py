import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pdfminer.high_level import extract_text

def build_vector_database():
    pdf_text_dir = "pdfs"
    db_folder = "./chroma_db"
    
    # Ensure old metadata-less index is completely wiped before rebuilding
    if os.path.exists(db_folder):
        print(f"Removing existing index at {db_folder} ...")
        shutil.rmtree(db_folder)
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    texts, metadatas = [], []

    if not os.path.exists(pdf_text_dir):
        print(f"Directory '{pdf_text_dir}' does not exist. Please create it and add your PDFs.")
        return

    for filename in sorted(os.listdir(pdf_text_dir)):
        if not filename.lower().endswith(".pdf"):
            continue
        print(f"Extracting {filename}...")
        try:
            file_path = os.path.join(pdf_text_dir, filename)
            raw_text = extract_text(file_path)
            chunks = splitter.split_text(raw_text)
        except Exception as e:
            print(f"FAILED {filename}: {e}")
            continue
        
        texts.extend(chunks)
        metadatas.extend([{"source": filename}] * len(chunks))
        print(f"  {len(chunks)} chunks")

    if texts:
        Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            persist_directory=db_folder
        )
        
        # Verify provenance across the full collection
        db = Chroma(persist_directory=db_folder, embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
        got = db.get(include=["metadatas"])
        sources = set(m.get("source") for m in got["metadatas"])
        print(f"Verified {len(sources)} distinct sources persisted, {len(got['metadatas'])} chunks total")
    else:
        print("No text chunks extracted.")

if __name__ == "__main__":
    build_vector_database()