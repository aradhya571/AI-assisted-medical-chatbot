import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "medical_knowledge_base"

def list_all_books():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    data = collection.get(include=["metadatas"])

    books = set()
    for meta in data["metadatas"]:
        books.add(f"{meta['book_name']}  |  {meta['medical_domain']}")

    print("\n📚 BOOKS FOUND IN DATABASE:\n")
    for book in sorted(books):
        print("•", book)

    print(f"\n✅ Total unique books: {len(books)}")

if __name__ == "__main__":
    list_all_books()
