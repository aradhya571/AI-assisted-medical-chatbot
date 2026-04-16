from chromadb.config import Settings

def get_chroma_settings():
    """Get standardized ChromaDB settings for all modules."""
    return Settings(
        anonymized_telemetry=False,
        allow_reset=False,  # Important: Keep this consistent
        is_persistent=True
    )
