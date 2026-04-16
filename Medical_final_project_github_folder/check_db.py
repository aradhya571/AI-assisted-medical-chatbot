import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "medical_knowledge_base"

try:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    print(f"Client created. Path: {CHROMA_PATH}")
    
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)
        print(f"Collection '{COLLECTION_NAME}' found.")
        print(f"Count: {collection.count()}")
    except Exception as e:
        print(f"Error getting collection: {e}")
        # List available collections
        print("Available collections:")
        for col in client.list_collections():
            print(f"- {col.name}")

except Exception as e:
    print(f"Error creating client: {e}")
