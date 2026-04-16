import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "medical_knowledge_base"

# Get your Groq API key from: https://console.groq.com/keys
# Set it as an environment variable: export GROQ_API_KEY="your_key_here"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY not set. Please set it as an environment variable.")
    print("   Get your key at: https://console.groq.com/keys")
    exit(1)

# The model you requested. 
# NOTE: If this specific ID fails, try "llama3-70b-8192" or "llama3-8b-8192"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile" 

def get_context(collection, query_text):
    """
    Searches the vector DB and returns the top 3 chunks formatted as a string.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=3
    )
    
    # If no results found
    if not results['documents'][0]:
        return ""

    context_str = ""
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        # We add the source book info so the AI knows where the info came from
        context_str += f"\n--- SOURCE: {meta['book_name']} ({meta['medical_domain']}) ---\n"
        context_str += doc + "\n"
        
    return context_str

def run_medical_rag():
    # 1. Setup ChromaDB (The Memory)
    print("Loading Medical Knowledge Base...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )
        print(f"Successfully loaded {collection.count()} medical documents.")
    except Exception as e:
        print(f"Error loading database: {e}")
        return

    # 2. Setup Groq Client (The Brain)
    if GROQ_API_KEY.startswith("gsk_"):
        client = Groq(api_key=GROQ_API_KEY)
    else:
        print("Error: Please replace 'gsk_...' with your actual Groq API Key.")
        return

    print("\n" + "="*50)
    print("   AI MEDICAL ASSISTANT (Powered by Groq + Your Books)")
    print("   Type 'exit' to quit.")
    print("="*50 + "\n")

    # --- MAIN CHAT LOOP ---
    while True:
        user_query = input("Doctor's Question: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break
        
        if not user_query:
            continue

        print(f"Searching books and consulting {GROQ_MODEL_NAME}...")

        # A. RETRIEVE (Get the relevant book pages)
        context_text = get_context(collection, user_query)
        
        if not context_text:
            print("No relevant information found in the database.")
            continue

        # B. AUGMENT (Build the prompt)
        system_prompt = """You are an expert medical AI assistant. 
        Answer the user's question strictly using the provided Context from medical textbooks. 
        If the answer is not in the context, say "I cannot find that information in the provided books."
        Do not hallucinate or use outside knowledge.
        Always cite the book name when providing facts."""

        user_message = f"""
        Context:
        {context_text}

        Question: 
        {user_query}
        """

        # C. GENERATE (Ask Groq)
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=GROQ_MODEL_NAME,
                temperature=0.2, # Low temperature for factual accuracy
            )

            response = chat_completion.choices[0].message.content

            # D. DISPLAY
            print("\n" + "-"*60)
            print("AI ANSWER:")
            print(response)
            print("-" * 60 + "\n")

        except Exception as e:
            print(f"\nError from Groq API: {e}")
            print("Tip: Check if your API key is correct or if the model name is valid.\n")

if __name__ == "__main__":
    run_medical_rag()