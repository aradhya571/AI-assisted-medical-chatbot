"""
Integrated Medical RAG System
Connects your OCR output to the RAG system - FIXED VERSION
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MedicalChunk:
    """Represents a chunk of medical data for embedding."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_type: str

class MedicalDataProcessor:
    """Processes extracted medical JSON data for vector storage."""
    
    def __init__(self):
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_extracted_data(self, json_path: str) -> Dict[str, Any]:
        """Load the extracted medical data from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded medical data from {json_path}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading JSON data: {e}")
            raise
    
    def create_chunks_from_medical_data(self, medical_data: Dict[str, Any]) -> List[MedicalChunk]:
        """
        FIXED: Convert medical JSON data into chunks for embedding.
        Matches the schema output by RobustMedicalOCR.
        """
        chunks = []
        
        # --- 1. Process Patient Information ---
        if 'patient' in medical_data:
            p = medical_data['patient']
            content = "Patient Information:\n"
            content += f"Name: {p.get('name', 'N/A')}\n"
            content += f"Age: {p.get('age', 'N/A')}\n"
            content += f"Gender: {p.get('gender', 'N/A')}\n"
            content += f"ID: {p.get('id', 'N/A')}"

            chunks.append(MedicalChunk(
                content=content,
                metadata={
                    'type': 'patient_info',
                    'patient_name': p.get('name', ''),
                    'patient_id': p.get('id', '')
                },
                chunk_id=str(uuid.uuid4()),
                source_type='metadata'
            ))

        # --- 2. Process Doctor / Lab Information ---
        # Combine Doctor and Lab details into one context chunk
        doc_lab_content = ""
        if 'doctor' in medical_data:
            d = medical_data['doctor']
            doc_lab_content += "Doctor Information:\n"
            doc_lab_content += f"Name: {d.get('name', 'N/A')}\n"
            doc_lab_content += f"Signature Text: {d.get('signature_text', 'N/A')}\n"
        
        if 'lab_details' in medical_data:
            l = medical_data['lab_details']
            doc_lab_content += "Lab Information:\n"
            doc_lab_content += f"Lab Name: {l.get('name', 'N/A')}\n"
            doc_lab_content += f"Report Date: {l.get('report_date', 'N/A')}\n"

        if doc_lab_content:
            chunks.append(MedicalChunk(
                content=doc_lab_content.strip(),
                metadata={'type': 'provider_info'},
                chunk_id=str(uuid.uuid4()),
                source_type='metadata'
            ))

        # --- 3. Process Test Results (The Critical Fix) ---
        if 'results' in medical_data and isinstance(medical_data['results'], list):
            results_list = medical_data['results']
            
            # We group results into blocks of text to maintain context
            # If the list is huge, you might split this into multiple chunks, 
            # but for typical lab reports, one large chunk is often better for LLM reasoning.
            
            content = "LABORATORY TEST RESULTS:\n"
            abnormal_findings = []

            for item in results_list:
                test_name = item.get('test_name', 'Unknown Test')
                value = item.get('value', '')
                unit = item.get('unit', '')
                flag = item.get('flag', '')
                ref_range = item.get('reference_range', '')
                
                # Format the line
                line = f"- {test_name}: {value} {unit}"
                if ref_range:
                    line += f" (Ref: {ref_range})"
                if flag and flag.lower() not in ['normal', '', 'none']:
                    line += f" [FLAG: {flag}]"
                    abnormal_findings.append(f"{test_name} ({value} {unit})")
                
                content += line + "\n"

            # Add summary of abnormals to the bottom of the chunk for better retrieval
            if abnormal_findings:
                content += "\nSUMMARY OF ABNORMAL FINDINGS:\n" + ", ".join(abnormal_findings)

            chunks.append(MedicalChunk(
                content=content,
                metadata={
                    'type': 'test_results', 
                    'count': len(results_list),
                    'has_abnormal': len(abnormal_findings) > 0
                },
                chunk_id=str(uuid.uuid4()),
                source_type='clinical_data'
            ))

        # --- 4. Process Clinical Notes ---
        if 'clinical_notes' in medical_data and medical_data['clinical_notes']:
            chunks.append(MedicalChunk(
                content=f"Clinical Notes:\n{medical_data['clinical_notes']}",
                metadata={'type': 'clinical_notes'},
                chunk_id=str(uuid.uuid4()),
                source_type='notes'
            ))
        
        self.logger.info(f"✅ Created {len(chunks)} chunks from medical data")
        return chunks

class MedicalRAGSystem:
    """RAG system for medical data using ChromaDB and LLM."""
    
    def __init__(self, 
                 chroma_db_path: str = "./medical_chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_provider: str = "groq",
                 groq_model: str = "llama-3.3-70b-versatile",
                 openai_model: str = "gpt-3.5-turbo"):
        
        self.chroma_db_path = chroma_db_path
        self.embedding_model_name = embedding_model
        self.llm_provider = llm_provider
        self.groq_model = groq_model
        self.openai_model = openai_model
        
        self.logger = self._setup_logging()
        self._setup_embedding_model()
        self._setup_chroma_client()
        self._setup_llm_client()
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _setup_embedding_model(self):
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            raise
    
    def _setup_chroma_client(self):
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
            self.logger.info(f"Connected to ChromaDB at {self.chroma_db_path}")
        except Exception as e:
            self.logger.error(f"Error setting up ChromaDB: {e}")
            raise
    
    def _setup_llm_client(self):
        try:
            if self.llm_provider == "groq":
                groq_api_key = os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    raise ValueError("GROQ_API_KEY not found in environment variables")
                self.llm_client = Groq(api_key=groq_api_key)
            elif self.llm_provider == "openai":
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                self.llm_client = OpenAI(api_key=openai_api_key)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
            
            self.logger.info(f"Initialized {self.llm_provider} LLM client")
        except Exception as e:
            self.logger.error(f"Error setting up LLM client: {e}")
            raise
    
    def create_or_get_collection(self, collection_name: str = "medical_reports"):
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            self.logger.info(f"Retrieved existing collection: {collection_name}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Medical reports and test results"}
            )
            self.logger.info(f"Created new collection: {collection_name}")
        return self.collection
    
    def add_chunks_to_database(self, chunks: List[MedicalChunk]):
        if not hasattr(self, 'collection'):
            self.create_or_get_collection()
        
        if not chunks:
            self.logger.error("❌ No chunks provided to add to database!")
            return
        
        try:
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                documents.append(chunk.content)
                metadata = {}
                for key, value in chunk.metadata.items():
                    if value is None:
                        metadata[key] = ""
                    elif isinstance(value, (int, float, bool, str)):
                        metadata[key] = value
                    else:
                        metadata[key] = str(value)
                metadatas.append(metadata)
                ids.append(chunk.chunk_id)
            
            self.logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(documents).tolist()
            
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                self.collection.add(
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end]
                )
            
            self.logger.info(f"✅ Added {len(chunks)} total chunks to ChromaDB")
            
        except Exception as e:
            self.logger.error(f"Error adding chunks to database: {e}")
            raise
    
    def search_medical_data(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if not hasattr(self, 'collection'):
            self.create_or_get_collection()
        
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i]
                    })
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error searching medical data: {e}")
            raise
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        try:
            context = "\n\n".join([
                f"Source: {doc['metadata'].get('type', 'unknown')}\n{doc['content']}"
                for doc in context_docs
            ])
            
            prompt = f"""You are a medical information assistant. Answer the user's question based on the provided medical report data.

CONTEXT FROM MEDICAL REPORTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer based ONLY on the information provided in the context.
2. If the information is not available, state that clearly.
3. When discussing test results, provide the exact Value and Unit found in the context.
4. Mention if a result is flagged (e.g., High, Low).

ANSWER:"""
            
            response = self.llm_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return f"Error generating response: {str(e)}"

    def query_medical_data(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        try:
            search_results = self.search_medical_data(question, n_results)
            answer = self.generate_answer(question, search_results)
            return {
                "question": question,
                "answer": answer,
                "sources": search_results,
                "source_count": len(search_results)
            }
        except Exception as e:
            self.logger.error(f"Error in query pipeline: {e}")
            return {"question": question, "answer": f"Error: {e}", "sources": [], "source_count": 0}

class MedicalDataRAGPipeline:
    """Complete pipeline for medical data RAG system."""
    
    def __init__(self, json_file_path: str, chroma_db_path: str = "./medical_chroma_db"):
        self.json_file_path = json_file_path
        self.processor = MedicalDataProcessor()
        self.rag_system = MedicalRAGSystem(chroma_db_path=chroma_db_path)
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self, collection_name: str = "medical_reports", force_recreate: bool = False):
        try:
            self.logger.info("Loading extracted medical data...")
            medical_data = self.processor.load_extracted_data(self.json_file_path)
            
            self.logger.info("Creating chunks from medical data...")
            chunks = self.processor.create_chunks_from_medical_data(medical_data)
            
            if not chunks:
                self.logger.error("❌ No chunks were created! Check your JSON structure.")
                return False
            
            if force_recreate:
                try:
                    self.rag_system.chroma_client.delete_collection(collection_name)
                    self.logger.info(f"Deleted existing collection: {collection_name}")
                except:
                    pass
            
            self.rag_system.create_or_get_collection(collection_name)
            self.rag_system.add_chunks_to_database(chunks)
            self.logger.info("✅ Database setup completed successfully!")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error setting up database: {e}")
            return False

    def query(self, question: str):
        return self.rag_system.query_medical_data(question)

    def interactive_query_session(self):
        print("\n" + "="*60)
        print("🏥 MEDICAL DATA QUERY SYSTEM")
        print("="*60)
        while True:
            q = input("\n❓ Question (or 'q' to quit): ").strip()
            if q.lower() in ['q', 'quit', 'exit']: break
            if not q: continue
            
            res = self.query(q)
            print(f"\n💡 Answer:\n{res['answer']}")
            print(f"\n📚 Sources: {res['source_count']}")

def main():
    print("\n" + "="*60)
    print("🏥 MEDICAL RAG SYSTEM - COMPLETE PIPELINE")
    print("="*60)
    
    pipeline = MedicalDataRAGPipeline(
        json_file_path="medical_ocr_output.json",
        chroma_db_path="./medical_chroma_db"
    )
    
    # 1. Setup DB (Load JSON -> Chunk -> Embed -> Store)
    if pipeline.setup_database(force_recreate=True):
        
        # 2. Run Test Questions
        questions = [
            "What is the patient's name and age?",
            "What is the hemoglobin level?", 
            "Who is the doctor?",
            "Are there any abnormal values?"
        ]
        
        print("\n🧪 TESTING...")
        for q in questions:
            print(f"\nQ: {q}")
            print(f"A: {pipeline.query(q)['answer']}")
            
        # 3. Interactive Mode
        pipeline.interactive_query_session()

if __name__ == "__main__":
    main()