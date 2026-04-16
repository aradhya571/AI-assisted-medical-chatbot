import os
import re
import uuid
import fitz  # PyMuPDF for fast PDF extraction
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION ---

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "medical_knowledge_base"

# Map the exact book names to their local file paths.
# IMPORTANT: Update the values on the right to match your actual file locations.
# IMPORTANT: Update the values on the right to match your actual file locations.
# Use this version if you are already inside the "Medical_final_project_github_folder" in your terminal
# Updated CONFIG to match your new PDF filename
BOOKS_CONFIG = {
    # "Apley and Solomon’s System of Orthopaedics and Trauma": "books/Apley's System of Orthopaedics and Fractures 9th ed.pdf",
    # "Ross_and_willson_anatomy_and_physiology": "books/Ross_and_willson_anatomy_and_physiology.pdf",
    # "Harrison’s Principles of Internal Medicine": "books/Harrison’s Principles of Internal Medicine, 21st Edition (Vol.1 & Vol.2).pdf",
    # "Robbins-and-Cotran-Pathologic-Basis-of-Disease": "books/Robbins-and-Cotran-Pathologic-Basis-of-Disease.pdf",
    # "Gray's Anatomy - 40th Ed": "books/Gray's Anatomy - 40th Ed.pdf"
    "Davidson's Principles & Practice of Medicine": "books/Davidson's Principles & Practice of Medicine.pdf"
}

# --- 2. TEXT EXTRACTION (PDF) ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Opens a PDF file and extracts text from all pages.
    Returns the raw text of the entire book.
    """
    if not os.path.exists(pdf_path):
        print(f"Warning: File not found at {pdf_path}. Skipping.")
        return ""

    try:
        doc = fitz.open(pdf_path)
        full_text = []
        print(f"  - Extracting text from {os.path.basename(pdf_path)} ({len(doc)} pages)...")
        
        for page in doc:
            # Extract plain text
            text = page.get_text("text") 
            full_text.append(text)
            
        return "\n".join(full_text)
    except Exception as e:
        print(f"  - Error reading PDF {pdf_path}: {e}")
        return ""

# --- 3. PREPROCESSING ---

def clean_text(text: str) -> str:
    """
    Removes noise, headers, footers, and page artifacts.
    """
    # Remove page numbers (e.g., "Page 12" or isolated numbers on lines)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)

    # Collapse multiple spaces/newlines into single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common headers (Customize based on your specific PDF artifacts)
    headers = ["CHAPTER", "SECTION", "PART", "Table of Contents"]
    for header in headers:
        text = text.replace(header, "")
        
    return text

# REPLACE THE OLD get_text_chunks FUNCTION WITH THIS ONE
def get_text_chunks(text: str):
    """
    Splits text into chunks using character count.
    ~1500 chars is roughly 300-400 tokens.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,    # Target size in characters
        chunk_overlap=300,  # Overlap to keep context
        separators=["\n\n", ". ", " ", ""] # Try to break at paragraphs/sentences first
    )
    return text_splitter.split_text(text)

# --- 4. METADATA ENRICHMENT ---

def enrich_metadata(chunk_text: str, book_name: str) -> dict:
    """
    Analyzes the chunk to extract structured metadata.
    NOTE: In a production system, this function would call an LLM API.
    """
    text_lower = chunk_text.lower()
    
    # Initialize default metadata structure
    meta = {
        "book_name": book_name,
        "medical_domain": "General Medicine",
        "body_system": "Unknown",
        "sub_system": "Unknown"
    }

    # 1. Determine Medical Domain based on Book Title or Keywords
    if "orthopaedics" in book_name.lower():
        meta["medical_domain"] = "Orthopaedics"
    elif "anatomy" in book_name.lower():
        meta["medical_domain"] = "Anatomy"
    elif "pathology" in book_name.lower():
        meta["medical_domain"] = "Pathology"
    elif "internal medicine" in book_name.lower():
        meta["medical_domain"] = "Internal Medicine"

    # 2. Determine Body System (Heuristic Keyword Match)
    system_keywords = {
        "respiratory": ["lung", "alveoli", "bronchi", "breath", "pulmonary"],
        "cardiovascular": ["heart", "aorta", "atrium", "ventricle", "blood pressure"],
        "musculoskeletal": ["bone", "muscle", "tendon", "ligament", "fracture", "joint"],
        "nervous": ["brain", "nerve", "spinal", "neuron", "cortex"],
        "digestive": ["stomach", "intestine", "liver", "colon", "esophagus"]
    }

    for system, keywords in system_keywords.items():
        if any(k in text_lower for k in keywords):
            meta["body_system"] = system
            
            # Simple heuristic for sub-system (first matching keyword)
            # In reality, this needs an LLM to be accurate.
            matched_keyword = next(k for k in keywords if k in text_lower)
            meta["sub_system"] = matched_keyword 
            break

    return meta

# --- 5. MAIN INGESTION PIPELINE ---

def ingest_books():
    # Initialize ChromaDB Client
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Embedding Model (All-MiniLM is efficient for laptop use)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"--- Starting Ingestion into '{COLLECTION_NAME}' ---")

    for book_name, file_path in BOOKS_CONFIG.items():
        print(f"\nProcessing Book: {book_name}")
        
        # 1. Extract
        raw_text = extract_text_from_pdf(file_path)
        if not raw_text:
            continue # Skip if extraction failed
        
        # 2. Clean
        cleaned_text = clean_text(raw_text)
        
        # 3. Chunk
        chunks = get_text_chunks(cleaned_text)
        print(f"  - Generated {len(chunks)} chunks.")
        
        # 4. Enrich & Prepare Batches
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Create unique ID: BookName_Index_UUID
            chunk_id = f"{book_name[:10].replace(' ', '')}_{i}_{str(uuid.uuid4())[:6]}"
            
            # Enrich Metadata
            metadata = enrich_metadata(chunk, book_name)
            
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(metadata)

        # 5. Store (Batch Upsert to prevent memory issues)
        batch_size = 200
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        print(f"  - Storing in ChromaDB (in {total_batches} batches)...")
        
        for i in range(0, len(documents), batch_size):
            collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size]
            )
            
        print(f"  - Completed {book_name}")

    print("\n--- Ingestion Complete ---")

# --- 6. EXECUTION ---

if __name__ == "__main__":
    # Ensure the directory exists
    if not os.path.exists("./books"):
        print("Note: Please create a './books' folder and place your PDF files there.")
        print("Then update the BOOKS_CONFIG paths in the script.")
    else:
        ingest_books()
        
    # --- TEST QUERY (Optional: To verify it worked) ---
    print("\n--- Verifying with a Test Query ---")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    
    results = collection.query(
        query_texts=["treatment for femur fracture"],
        n_results=1,
        where={"medical_domain": "Orthopaedics"} # Metadata Filter
    )
    
    if results['documents']:
        print("Test Query Result:")
        print(f"Metadata: {results['metadatas'][0]}")
        print(f"Snippet: {results['documents'][0][0][:200]}...")
    else:
        print("No results found (Did you add the PDF files?)")