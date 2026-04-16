# medical_consultation_app.py

import streamlit as st
import json
import chromadb
import base64
import os
import tempfile
import time
from functools import wraps
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from groq import Groq
from PIL import Image
import io
from datetime import datetime
from audio_recorder_streamlit import audio_recorder
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# IMPORTS - ENHANCED TRIAGE ENGINE & OCR
# ============================================================================
try:
    from enhanced_triage_engine import MedicalTriageEngine
except ImportError:
    st.error("⚠️ 'enhanced_triage_engine.py' not found. Please save it in the same folder.")

try:
    from ocr_engine import RobustMedicalOCR
except ImportError:
    st.error("⚠️ 'ocr_engine.py' not found. Please save your OCR code in the same folder.")

# ============================================================================
# CONFIGURATION
# ============================================================================
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "medical_knowledge_base"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please set it as an environment variable or in .env file")
    st.stop()
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def initialize_session_state():
    """Initialize all session state variables"""
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "patient_data" not in st.session_state:
        st.session_state.patient_data = None
    
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    
    if "groq_client" not in st.session_state:
        if GROQ_API_KEY.startswith("gsk_"):
            try:
                st.session_state.groq_client = Groq(api_key=GROQ_API_KEY)
            except Exception as e:
                st.error(f"Failed to initialize Groq client: {e}")
                st.session_state.groq_client = None
    
    if "asked_questions" not in st.session_state:
        st.session_state.asked_questions = set()
    
    if "chief_complaint" not in st.session_state:
        st.session_state.chief_complaint = None
    
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None
    
    if "triage_engine" not in st.session_state:
        try:
            st.session_state.triage_engine = MedicalTriageEngine(groq_api_key=GROQ_API_KEY)
        except Exception as e:
            st.error(f"Failed to initialize triage engine: {e}")
            st.session_state.triage_engine = None
    
    if "system_status" not in st.session_state:
        st.session_state.system_status = {
            "textbook_available": False,
            "triage_available": False,
            "last_error": None
        }

initialize_session_state()

# ============================================================================
# DATABASE SETUP
# ============================================================================
@st.cache_resource
def get_book_collection():
    """
    Initialize and return ChromaDB collection with comprehensive error handling
    """
    try:
        # Ensure directory exists
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Setup embedding function
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get collection
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )
        
        # Check document count
        doc_count = collection.count()
        
        # Update status based on count
        if doc_count > 0:
            st.session_state.system_status["textbook_available"] = True
            st.session_state.system_status["last_error"] = None
            print(f"✅ ChromaDB loaded: {doc_count:,} documents")
        else:
            st.session_state.system_status["textbook_available"] = False
            st.session_state.system_status["last_error"] = "Collection is empty"
            print(f"⚠️ ChromaDB collection exists but is empty")
        
        return collection
        
    except Exception as e:
        st.session_state.system_status["textbook_available"] = False
        st.session_state.system_status["last_error"] = str(e)
        print(f"❌ ChromaDB error: {e}")
        return None

# Initialize
book_collection = get_book_collection()
if book_collection:
    try:
        count = book_collection.count()
        print(f"📚 ChromaDB Status: {count} documents loaded")
        
        # Show in sidebar
        st.sidebar.info(f"📚 Textbook Database: {count} documents")
        
    except Exception as e:
        print(f"❌ ChromaDB count error: {e}")
        st.sidebar.error(f"❌ Database error: {e}")
else:
    print("❌ ChromaDB: Collection is None")
    st.sidebar.warning("⚠️ Textbook database not initialized")
    
    # Add quick fix button
    if st.sidebar.button("🔧 Initialize Database with Sample Data"):
        with st.spinner("Setting up..."):
            success, msg = populate_sample_data()
            if success:
                st.success(msg)
                st.cache_resource.clear()
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Failed: {msg}")

# ============================================================================
# UTILITY DECORATORS
# ============================================================================
def retry_on_failure(max_retries=3, delay=2):
    """
    Decorator for retrying failed API calls with exponential backoff
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    
                    # Check if error is retryable
                    if "rate limit" in error_str:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        if attempt < max_retries - 1:
                            print(f"⚠️ Rate limit hit, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                        continue
                    elif "timeout" in error_str or "connection" in error_str:
                        if attempt < max_retries - 1:
                            print(f"⚠️ Connection issue, retrying... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                        continue
                    else:
                        # Non-retryable error, fail immediately
                        break
            
            # All retries failed
            raise last_exception
        return wrapper
    return decorator

# ============================================================================
# DATA FORMATTING FUNCTIONS
# ============================================================================
def format_json_to_text(medical_data: dict) -> Tuple[str, List[str]]:
    """
    Convert JSON medical data to formatted text
    
    Args:
        medical_data: Dictionary containing patient medical data
    
    Returns:
        Tuple of (formatted_text, abnormal_findings_list)
    """
    summary = "**Patient Clinical Data**\n\n"
    abnormal_findings = []
    
    try:
        # Extract Patient Info
        if 'patient' in medical_data:
            p = medical_data['patient']
            summary += f"**Patient:** {p.get('name', 'N/A')} | **Age:** {p.get('age', 'N/A')} | **Gender:** {p.get('gender', 'N/A')}\n\n"

        # Extract Lab Results
        if 'results' in medical_data and isinstance(medical_data['results'], list):
            summary += "**Laboratory Results:**\n"
            for item in medical_data['results']:
                test = item.get('test_name', 'Unknown')
                val = item.get('value', '')
                unit = item.get('unit', '')
                flag = item.get('flag', '')
                
                line = f"- {test}: {val} {unit}"
                if flag and flag.lower() not in ['normal', 'none', '']:
                    line += f" ⚠️ **{flag}**"
                    abnormal_findings.append(test)
                summary += line + "\n"

        # Add Clinical Notes
        if 'clinical_notes' in medical_data and medical_data['clinical_notes']:
            summary += f"\n**Clinical Notes / Impression:**\n{medical_data['clinical_notes']}\n"

        # Add Doctor/Lab info
        if 'doctor' in medical_data:
            summary += f"\n**Doctor:** {medical_data['doctor'].get('name', 'N/A')}"
        
        return summary, abnormal_findings
        
    except Exception as e:
        print(f"Error formatting JSON: {e}")
        return f"**Patient Data:** {str(medical_data)[:500]}...", []

# ============================================================================
# IMAGE HANDLING
# ============================================================================
def encode_image(image_obj: Image.Image) -> str:
    """
    Encode PIL Image to base64 string
    
    Args:
        image_obj: PIL Image object
    
    Returns:
        Base64 encoded string
    """
    try:
        buffered = io.BytesIO()
        image_obj.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        raise

# ============================================================================
# AUDIO TRANSCRIPTION
# ============================================================================
def transcribe_audio(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribe audio using Groq's Whisper API
    
    Args:
        audio_bytes: Audio data in bytes
    
    Returns:
        Transcribed text or None if failed
    """
    try:
        client = st.session_state.groq_client
        
        if not client:
            st.error("Audio transcription unavailable - API client not initialized")
            return None
        
        # Save audio bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_audio_path = tmp_audio.name
        
        # Transcribe using Groq
        with open(tmp_audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(tmp_audio_path, audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="json",
                language="en"
            )
        
        # Cleanup
        os.remove(tmp_audio_path)
        
        return transcription.text
        
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        if 'tmp_audio_path' in locals() and os.path.exists(tmp_audio_path):
            try:
                os.remove(tmp_audio_path)
            except:
                pass
        return None

# ============================================================================
# KEYWORD EXTRACTION WITH VALIDATION
# ============================================================================
def extract_keywords_fallback(text: str) -> str:
    """
    Simple rule-based fallback when LLM extraction fails
    
    Args:
        text: User query text
    
    Returns:
        Comma-separated keywords
    """
    # Common medical symptom keywords
    medical_terms = {
        'pain', 'fever', 'cough', 'headache', 'nausea', 'fatigue',
        'dizziness', 'vomiting', 'diarrhea', 'rash', 'swelling',
        'bleeding', 'chest', 'abdomen', 'back', 'arm', 'leg',
        'fracture', 'infection', 'injury', 'breathing', 'heart',
        'stomach', 'throat', 'ear', 'eye', 'joint', 'muscle',
        'weakness', 'numbness', 'confusion', 'anxiety', 'depression'
    }
    
    # Extract words that match medical terms
    words = text.lower().split()
    found_terms = [word for word in words if word in medical_terms]
    
    if found_terms:
        return ', '.join(found_terms[:5])
    else:
        # Last resort: return first few meaningful words (remove common words)
        stop_words = {'i', 'a', 'the', 'is', 'am', 'are', 'have', 'has', 'been', 'my', 'me'}
        meaningful_words = [w for w in words if w.lower() not in stop_words]
        return ', '.join(meaningful_words[:5])

def extract_keywords(text: str, image_obj: Optional[Image.Image] = None) -> Tuple[str, bool]:
    """
    Extract medical keywords with validation and fallback
    
    Args:
        text: User query text
        image_obj: Optional PIL Image object
    
    Returns:
        Tuple of (keywords_string, extraction_successful)
    """
    client = st.session_state.groq_client
    
    if not client:
        return extract_keywords_fallback(text), False
    
    prompt = f"""Extract ONLY 3-5 key medical terms from this query.

Rules:
- Output ONLY comma-separated terms
- No preamble, no explanation
- Medical terminology preferred
- Focus on symptoms, conditions, anatomy

Query: {text}

Example output: "Fatigue, Anemia, Hemoglobin, Iron Deficiency"
Your output:"""
    
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    if image_obj:
        try:
            base64_img = encode_image(image_obj)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })
        except Exception as e:
            print(f"Failed to encode image for keyword extraction: {e}")
    
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=0.1,
            max_tokens=100
        )
        
        keywords = response.choices[0].message.content.strip()
        
        # VALIDATION
        # Remove common preambles
        keywords = keywords.replace("Here are the key terms:", "")
        keywords = keywords.replace("Key medical terms:", "")
        keywords = keywords.replace("Output:", "")
        keywords = keywords.strip()
        
        # Check if it's actually comma-separated
        terms = [t.strip() for t in keywords.split(',')]
        
        # Validation checks
        if len(terms) < 2:
            return extract_keywords_fallback(text), False
        
        if len(keywords) > 200:
            return extract_keywords_fallback(text), False
        
        # Check for full sentences (periods indicate sentences)
        if '.' in keywords and len(keywords) > 50:
            return extract_keywords_fallback(text), False
        
        # Success!
        return keywords, True
        
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return extract_keywords_fallback(text), False

# ============================================================================
# TEXTBOOK SEARCH WITH QUALITY FILTERING
# ============================================================================
def search_textbooks(query: str, min_quality_score: float = 0.6) -> Tuple[str, Dict]:
    """
    Search medical textbooks with quality filtering
    
    Args:
        query: Search query string
        min_quality_score: Maximum distance to include (lower = more strict)
    
    Returns:
        Tuple of (context_string, metadata_dict)
    """
    # Check if collection is available
    if not book_collection:
        return "", {
            "status": "error",
            "message": "Textbook database not available",
            "results_count": 0
        }
    
    try:
        # Perform search
        results = book_collection.query(
            query_texts=[query], 
            n_results=5
        )
        
        # Validate results exist
        if not results['documents'] or not results['documents'][0]:
            return "", {
                "status": "no_results",
                "message": "No relevant textbook content found",
                "results_count": 0
            }
        
        # Filter by quality score
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        filtered_results = []
        for doc, distance, meta in zip(documents, distances, metadatas):
            if distance <= min_quality_score:
                filtered_results.append({
                    'document': doc,
                    'distance': distance,
                    'metadata': meta,
                    'quality': 'excellent' if distance < 0.3 else 'good'
                })
        
        # Check if we have good results
        if not filtered_results:
            return "", {
                "status": "low_quality",
                "message": f"Found {len(documents)} results but all were low quality",
                "results_count": 0,
                "best_distance": min(distances) if distances else 1.0
            }
        
        # Build context from filtered results
        context = ""
        for i, result in enumerate(filtered_results, 1):
            book_name = result['metadata'].get('book_name', 'Medical Textbook')
            quality_indicator = "⭐" if result['quality'] == 'excellent' else "✓"
            
            context += f"\n**{quality_indicator} Source {i}: {book_name}**\n"
            context += f"{result['document']}\n\n"
        
        # Return context with metadata
        return context, {
            "status": "success",
            "results_count": len(filtered_results),
            "filtered_out": len(documents) - len(filtered_results),
            "best_quality": filtered_results[0]['quality'],
            "sources": [r['metadata'].get('book_name') for r in filtered_results]
        }
        
    except Exception as e:
        print(f"Textbook search error: {e}")
        return "", {
            "status": "error",
            "message": str(e),
            "results_count": 0
        }

# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================
def extract_critical_symptoms(chat_history: List[Dict]) -> List[str]:
    """
    Extract high-priority symptoms from conversation history
    
    Args:
        chat_history: List of message dictionaries
    
    Returns:
        List of critical symptom strings
    """
    critical_keywords = [
        # Cardiac
        'chest pain', 'heart pain', 'palpitation', 'crushing pain',
        'radiating pain', 'left arm pain',
        # Neurological
        'severe headache', 'worst headache', 'sudden numbness', 'confusion',
        'difficulty speaking', 'vision loss', 'seizure', 'slurred speech',
        # Respiratory
        'can\'t breathe', 'difficulty breathing', 'shortness of breath',
        'gasping', 'blue lips',
        # Trauma
        'head injury', 'loss of consciousness', 'severe bleeding',
        'unconscious', 'knocked out',
        # Other emergencies
        'suicidal', 'overdose', 'severe abdominal pain', 'vomiting blood',
        'coughing blood', 'severe burn'
    ]
    
    critical_found = []
    
    for msg in chat_history:
        if msg["role"] == "user":
            content_lower = msg["content"].lower()
            
            # Check for critical keywords
            for keyword in critical_keywords:
                if keyword in content_lower:
                    critical_found.append(msg["content"])
                    break
    
    # Deduplicate and return first 3 most critical
    return list(dict.fromkeys(critical_found))[:3]

def extract_conversation_summary(chat_history: List[Dict], max_exchanges: int = 10) -> str:
    """
    Extract intelligent conversation summary with context preservation
    
    Args:
        chat_history: List of message dicts
        max_exchanges: Maximum number of back-and-forth exchanges to include
    
    Returns:
        Formatted conversation summary string
    """
    if not chat_history:
        return ""
    
    # Strategy 1: Include recent exchanges (keeps context)
    recent_history = chat_history[-(max_exchanges * 2):] if len(chat_history) > max_exchanges * 2 else chat_history
    
    # Strategy 2: Extract critical symptoms from full history
    critical_symptoms = extract_critical_symptoms(chat_history)
    
    # Build summary
    summary = ""
    
    # Add critical symptoms first (if found in earlier conversation)
    if critical_symptoms:
        summary += "**KEY SYMPTOMS FROM FULL CONVERSATION:**\n"
        for symptom in critical_symptoms:
            summary += f"- {symptom}\n"
        summary += "\n"
    
    # Add recent conversation
    summary += "**RECENT CONVERSATION:**\n"
    for msg in recent_history:
        role = "Patient" if msg["role"] == "user" else "Doctor"
        content = msg['content']
        
        # Truncate very long messages
        if len(content) > 200:
            content = content[:197] + "..."
        
        summary += f"{role}: {content}\n"
    
    # Add conversation statistics
    if len(chat_history) > max_exchanges * 2:
        summary += f"\n*(Showing last {max_exchanges} exchanges of {len(chat_history)//2} total)*\n"
    
    return summary

def format_conversation_for_prompt(chat_history: List[Dict], patient_context: str = "") -> str:
    """
    Format conversation optimally for LLM prompt
    
    Args:
        chat_history: List of message dictionaries
        patient_context: Additional patient data context
    
    Returns:
        Formatted prompt section string
    """
    summary = extract_conversation_summary(chat_history, max_exchanges=8)
    
    # Add token count estimate
    estimated_tokens = len(summary.split()) * 1.3
    
    prompt_section = f"""**CONVERSATION CONTEXT:**
{summary}

**PATIENT DATA:**
{patient_context if patient_context else "No additional patient data provided."}
"""
    
    # Log for debugging
    print(f"📊 Conversation summary: ~{int(estimated_tokens)} tokens")
    
    return prompt_section

# ============================================================================
# DOCTOR RECOMMENDATIONS FORMATTING
# ============================================================================
def format_doctor_recommendations(triage_result) -> str:
    """
    Format doctor recommendations for display
    
    Args:
        triage_result: TriageResult object from triage engine
    
    Returns:
        Formatted recommendations string
    """
    if not triage_result.recommended_doctors:
        return ""
    
    output = "\n\n---\n\n### 👨‍⚕️ Recommended Specialists\n\n"
    output += "Based on your symptoms, I recommend consulting:\n\n"
    
    # Group doctors by department
    from collections import defaultdict
    dept_doctors = defaultdict(list)
    for doc in triage_result.recommended_doctors:
        dept_doctors[doc.department].append(doc.name)
    
    for dept, doctors in dept_doctors.items():
        output += f"**{dept}:**\n"
        for doctor in doctors:
            output += f"- {doctor}\n"
        output += "\n"
    
    # Add helpline numbers
    if triage_result.helpline_numbers:
        output += "### 📞 Contact Information\n\n"
        for number in triage_result.helpline_numbers:
            output += f"- {number}\n"
    
    return output

# ============================================================================
# API CALL WITH RETRY LOGIC
# ============================================================================
@retry_on_failure(max_retries=3, delay=2)
def call_groq_api(client, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> str:
    """
    Wrapper for Groq API calls with retry logic
    
    Args:
        client: Groq client instance
        messages: List of message dictionaries
        model: Model name string
        temperature: Temperature value
        max_tokens: Maximum tokens for response
    
    Returns:
        Response content string
    """
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Validate response
    if not response.choices or not response.choices[0].message:
        raise ValueError("Invalid API response structure")
    
    content = response.choices[0].message.content
    
    if content is None or content.strip() == "":
        raise ValueError("Empty response from API")
    
    return content

# ============================================================================
# SIMPLE FALLBACK RESPONSE
# ============================================================================
def generate_simple_response(user_query: str, chat_history: List[Dict]) -> str:
    """
    Minimal response generation without textbook search or complex features
    Last resort fallback
    
    Args:
        user_query: User's query text
        chat_history: Conversation history
    
    Returns:
        Simple response string
    """
    client = st.session_state.groq_client
    
    if not client:
        return "⚠️ System temporarily unavailable. Please try again in a moment."
    
    # Very simple prompt
    recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history
    conversation = "\n".join([
        f"{'Patient' if m['role'] == 'user' else 'Doctor'}: {m['content']}"
        for m in recent_messages
    ])
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful medical AI assistant. Provide a brief, helpful response."
        },
        {
            "role": "user",
            "content": f"Conversation:\n{conversation}\n\nPatient: {user_query}\n\nYour response:"
        }
    ]
    
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Simple response also failed: {e}")
        return "⚠️ I'm experiencing difficulties. Please try rephrasing your question."

# ============================================================================
# MAIN RESPONSE GENERATION
# ============================================================================
def generate_response(
    user_query: str,
    chat_history: List[Dict],
    patient_context: str = "",
    image_obj: Optional[Image.Image] = None
) -> str:
    """
    Generate response with comprehensive error handling
    
    Args:
        user_query: User's query text
        chat_history: List of previous messages
        patient_context: Additional patient data
        image_obj: Optional uploaded image
    
    Returns:
        Generated response string
    """
    try:
        # Validate client
        if "groq_client" not in st.session_state or st.session_state.groq_client is None:
            return "⚠️ System configuration error. Please refresh the page."
        
        client = st.session_state.groq_client
        
        # Extract keywords with error handling
        try:
            keywords, extraction_success = extract_keywords(user_query, image_obj)
            if not extraction_success:
                print("⚠️ Keyword extraction had issues, using fallback")
        except Exception as e:
            print(f"Keyword extraction failed: {e}")
            keywords = user_query
        
        # Search textbooks with error handling
        try:
            book_context, search_metadata = search_textbooks(keywords)
        except Exception as e:
            print(f"Textbook search failed: {e}")
            book_context = ""
            search_metadata = {"status": "error", "message": str(e)}
        
        # Format conversation
        try:
            conversation_context = format_conversation_for_prompt(chat_history, patient_context)
        except Exception as e:
            print(f"Conversation formatting failed: {e}")
            conversation_context = f"Patient data: {patient_context}"
        
        # Determine response mode
        questions_asked = sum(1 for msg in chat_history if msg["role"] == "assistant" and "?" in msg["content"])
        user_wants_answer = any(kw in user_query.lower() for kw in [
            "what do i have", "what is wrong", "diagnosis", "tell me what",
            "what's my diagnosis", "give me answer", "what is it"
        ])
        
        # Run triage with error handling
        triage_result = None
        try:
            if st.session_state.triage_engine:
                triage_result = st.session_state.triage_engine.analyze_symptoms(
                    conversation_text=conversation_context + " " + user_query,
                    lab_data=patient_context,
                    conversation_history=chat_history,
                    uploaded_image=image_obj
                )
        except Exception as e:
            print(f"Triage analysis failed: {e}")
            # Create minimal triage result
            @dataclass
            class MinimalTriage:
                class Severity(Enum):
                    ROUTINE = "ROUTINE"
                severity: Enum = Severity.ROUTINE
                recommended_doctors: list = None
                helpline_numbers: list = None
                red_flags: list = None
                suggested_tests: list = None
                primary_concern: str = "General concern"
                reasoning: str = ""
                score: float = 0
            
            triage_result = MinimalTriage()
            triage_result.recommended_doctors = []
            triage_result.helpline_numbers = []
            triage_result.red_flags = []
            triage_result.suggested_tests = []
        
        # HANDLE EMERGENCY CASES IMMEDIATELY
        if triage_result and hasattr(triage_result.severity, 'value') and triage_result.severity.value == "EMERGENCY":
            emergency_msg = f"""🚨 **EMERGENCY MEDICAL SITUATION**

**CALL 911 / EMERGENCY SERVICES IMMEDIATELY**

**Why this is urgent:**
{triage_result.reasoning if triage_result.reasoning else "Your symptoms require immediate medical attention."}
"""
            
            if triage_result.red_flags:
                emergency_msg += "\n**Critical symptoms identified:**\n"
                for flag in triage_result.red_flags:
                    emergency_msg += f"- {flag}\n"
            
            emergency_msg += """
**Do NOT wait:**
- Call emergency services now
- Do not drive yourself
- Stay calm and follow dispatcher instructions

**Emergency Contacts:**
"""
            if triage_result.helpline_numbers:
                for number in triage_result.helpline_numbers:
                    emergency_msg += f"- {number}\n"
            else:
                emergency_msg += "- 911 (US) / 999 (UK) / 112 (EU)\n"
            
            return emergency_msg
        
        # HANDLE URGENT CASES
        urgent_prefix = ""
        if triage_result and hasattr(triage_result.severity, 'value') and triage_result.severity.value == "URGENT":
            urgent_prefix = """⚠️ **URGENT MEDICAL ATTENTION NEEDED**

Your symptoms require prompt medical evaluation within the next few hours.

"""
            user_wants_answer = True  # Force assessment mode
        
        # BUILD RED FLAGS CONTEXT
        red_flags_context = ""
        if triage_result and triage_result.red_flags:
            red_flags_context = f"""
**⚠️ CRITICAL SYMPTOMS IDENTIFIED:**
{chr(10).join(f'- {flag}' for flag in triage_result.red_flags)}

**These symptoms require particular attention in your assessment.**
"""
        
        # Build response based on mode
        if user_wants_answer or questions_asked >= 5:
            # ASSESSMENT MODE
            textbook_note = ""
            if search_metadata.get('status') == 'success':
                textbook_note = f"**TEXTBOOK KNOWLEDGE ({search_metadata['results_count']} high-quality sources):**\n{book_context}"
            elif search_metadata.get('status') == 'low_quality':
                textbook_note = "**TEXTBOOK KNOWLEDGE:** Limited relevant information found. Relying primarily on general medical knowledge."
            else:
                textbook_note = "**TEXTBOOK KNOWLEDGE:** Textbook search unavailable. Using general medical knowledge."
            
            system_prompt = f"""You are an experienced Medical AI Consultant.

{conversation_context}

{red_flags_context}

{textbook_note}
"""
            
            if triage_result:
                system_prompt += f"""
**TRIAGE ASSESSMENT:**
- Severity: {triage_result.severity.value if hasattr(triage_result.severity, 'value') else 'Unknown'}
- Primary Concern: {triage_result.primary_concern}
"""
            
            system_prompt += """
**YOUR TASK:**
1. Provide clear assessment based on information gathered
2. Mention possible conditions (differential diagnosis)
3. Emphasize urgency level if applicable
4. Suggest next steps (tests, specialist, or when to seek care)
5. Be empathetic and concise

**IMPORTANT:** Provide the assessment, don't ask more questions."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_query}]}
            ]
            
            if image_obj:
                try:
                    base64_img = encode_image(image_obj)
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                    })
                except Exception as e:
                    print(f"Image encoding failed: {e}")
            
            try:
                ai_response = call_groq_api(
                    client=client,
                    messages=messages,
                    model=MODEL_NAME,
                    temperature=0.7,
                    max_tokens=800
                )
                
                ai_response = urgent_prefix + ai_response
                
                # ADD SUGGESTED TESTS
                if triage_result and triage_result.suggested_tests:
                    ai_response += "\n\n### 🔬 Recommended Tests\n\n"
                    ai_response += "When you see your doctor, they may order:\n"
                    for test in triage_result.suggested_tests:
                        ai_response += f"- {test}\n"
                
                # Add doctor recommendations
                if triage_result:
                    ai_response += format_doctor_recommendations(triage_result)
                
                return ai_response
                
            except Exception as e:
                error_type = type(e).__name__
                
                if "rate limit" in str(e).lower():
                    return "⚠️ **System is experiencing high demand.** Please wait a moment and try again."
                elif "timeout" in str(e).lower():
                    return "⚠️ **Connection timeout.** Please check your internet connection and try again."
                elif "authentication" in str(e).lower():
                    return "⚠️ **System configuration error.** Please contact support."
                else:
                    print(f"API call failed with {error_type}: {e}")
                    # Try simple fallback
                    return generate_simple_response(user_query, chat_history)
        
        else:
            # INTERVIEW MODE
            textbook_context_note = book_context if search_metadata.get('status') == 'success' else "General medical knowledge."
            
            system_prompt = f"""You are a compassionate Medical AI conducting a patient interview.

{conversation_context}

{red_flags_context}

**TEXTBOOK CONTEXT:**
{textbook_context_note}
"""
            
            if triage_result:
                system_prompt += f"""
**TRIAGE INSIGHTS:**
- Suspected area: {triage_result.primary_concern}
- Current severity: {triage_result.severity.value if hasattr(triage_result.severity, 'value') else 'Unknown'}
"""
            
            system_prompt += """
**YOUR TASK:**
Ask ONE focused follow-up question to clarify symptoms.
Focus on: duration, severity, triggers, or associated symptoms.

**CRITICAL:** Ask ONLY ONE question. Be concise and empathetic."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_query}]}
            ]
            
            if image_obj:
                try:
                    base64_img = encode_image(image_obj)
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                    })
                except Exception as e:
                    print(f"Image encoding failed: {e}")
            
            try:
                return call_groq_api(
                    client=client,
                    messages=messages,
                    model=MODEL_NAME,
                    temperature=0.7,
                    max_tokens=200
                )
            except Exception as e:
                error_type = type(e).__name__
                
                if "rate limit" in str(e).lower():
                    return "⚠️ **System is experiencing high demand.** Please wait a moment and try again."
                elif "timeout" in str(e).lower():
                    return "⚠️ **Connection timeout.** Please try again."
                else:
                    print(f"API call failed with {error_type}: {e}")
                    return generate_simple_response(user_query, chat_history)
        
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"Unexpected error in generate_response: {type(e).__name__}: {e}")
        
        # Try simple fallback
        try:
            return generate_simple_response(user_query, chat_history)
        except:
            return """⚠️ **I'm experiencing technical difficulties.**

Please try:
1. Refreshing the page
2. Rephrasing your question
3. For urgent concerns, contact emergency services or your doctor

I apologize for the inconvenience."""

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: black;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: black;
    }
    
    /* Audio recorder styling */
    .stAudio { 
        display: none; 
    }
    
    div[data-testid="stAudioRecorder"] {
        background-color: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    div[data-testid="stAudioRecorder"] > div {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        font-weight: 500;
    }
    
    /* Warning/Error boxes */
    .stAlert {
        border-radius: 5px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #3498db !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR: UPLOAD & CONTEXT SECTION
# ============================================================================
with st.sidebar:
    st.title("🩺 Medical AI Assistant")
    st.markdown("---")
    
    # System Status Indicator
    with st.expander("🔧 System Status", expanded=False):
        if st.session_state.system_status["textbook_available"]:
            st.success("✅ Textbook Database: Online")
        else:
            st.error("❌ Textbook Database: Offline")
        
        if st.session_state.triage_engine:
            st.success("✅ Triage Engine: Active")
        else:
            st.warning("⚠️ Triage Engine: Limited")
        
        if st.session_state.groq_client:
            st.success("✅ AI Model: Connected")
        else:
            st.error("❌ AI Model: Disconnected")
        
        if st.session_state.system_status.get("last_error"):
            st.error(f"Last Error: {st.session_state.system_status['last_error']}")
    
    st.markdown("---")
    st.subheader("📤 Upload Patient Data")
    
    tab1, tab2, tab3 = st.tabs(["📄 Text/JSON", "📑 PDF Report", "📷 Image"])
    
    # ========================================================================
    # TAB 1: Text & JSON Upload
    # ========================================================================
    with tab1:
        upload_type = st.radio(
            "Select input format:",
            ["Text Input", "JSON File"],
            label_visibility="collapsed"
        )
        
        if upload_type == "Text Input":
            text_input = st.text_area(
                "Enter patient information or symptoms:",
                height=150,
                key="text_input",
                placeholder="Describe symptoms, medical history, or paste lab results..."
            )
            
            if st.button("📝 Load Text", key="load_text", type="primary"):
                if text_input and text_input.strip():
                    st.session_state.patient_data = text_input
                    st.session_state.uploaded_image = None
                    st.success("✅ Text loaded successfully!")
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter some text first")
        
        else:
            json_file = st.file_uploader(
                "Upload JSON file:",
                type=["json"],
                key="json_upload",
                help="Upload a JSON file containing patient medical data"
            )
            
            if json_file:
                try:
                    data = json.load(json_file)
                    formatted, abnormal = format_json_to_text(data)
                    st.session_state.patient_data = formatted
                    st.session_state.uploaded_image = None
                    
                    st.success("✅ JSON loaded successfully!")
                    
                    if abnormal:
                        st.warning(f"⚠️ {len(abnormal)} abnormal findings detected")
                    
                    with st.expander("📋 View Extracted Data"):
                        st.markdown(formatted)
                
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON format")
                except Exception as e:
                    st.error(f"❌ Error processing JSON: {e}")
    
    # ========================================================================
    # TAB 2: PDF Upload with OCR
    # ========================================================================
    with tab2:
        pdf_file = st.file_uploader(
            "Upload Medical PDF Report:",
            type=["pdf"],
            key="pdf_upload",
            help="Upload a PDF containing lab results or medical reports"
        )
        
        if pdf_file and st.button("🔍 Process PDF", key="process_pdf", type="primary"):
            with st.spinner("📄 Scanning document... (This may take a moment)"):
                try:
                    # Save uploaded PDF to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(pdf_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Initialize OCR Engine
                    extractor = RobustMedicalOCR(
                        pdf_path=tmp_path,
                        groq_api_key=GROQ_API_KEY
                    )
                    
                    # Process the document
                    extracted_data = extractor.process_document()
                    
                    # Format and load into session state
                    formatted_text, abnormal = format_json_to_text(extracted_data)
                    st.session_state.patient_data = formatted_text
                    st.session_state.uploaded_image = None
                    
                    # Cleanup
                    os.remove(tmp_path)
                    
                    st.success("✅ PDF processed successfully!")
                    
                    if abnormal:
                        st.warning(f"⚠️ {len(abnormal)} abnormal findings detected")
                    
                    with st.expander("📋 View Extracted Data"):
                        st.markdown(formatted_text)
                
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {e}")
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except:
                            pass
    
    # ========================================================================
    # TAB 3: Image Upload
    # ========================================================================
    with tab3:
        img_file = st.file_uploader(
            "Upload Medical Image (X-ray, Scan, etc.):",
            type=["jpg", "png", "jpeg"],
            key="img_upload",
            help="Upload images like X-rays, CT scans, or photos of symptoms"
        )
        
        if img_file:
            try:
                img = Image.open(img_file)
                st.session_state.uploaded_image = img
                st.session_state.patient_data = "Medical image uploaded for analysis."
                
                st.image(img, caption="Uploaded Image", use_container_width=True)
                st.success("✅ Image loaded successfully!")
            
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
    
    st.markdown("---")
    
    # ========================================================================
    # CURRENT CONTEXT DISPLAY
    # ========================================================================
    if st.session_state.patient_data or st.session_state.uploaded_image:
        st.subheader("📋 Active Context")
        
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, width=150)
        
        if st.session_state.patient_data:
            with st.expander("📄 View Patient Data"):
                st.markdown(st.session_state.patient_data)
        
        if st.button("🗑️ Clear Context", key="clear_context", type="secondary"):
            st.session_state.patient_data = None
            st.session_state.uploaded_image = None
            st.success("✅ Context cleared")
            st.rerun()
    
    st.markdown("---")
    
    # ========================================================================
    # CONVERSATION CONTROLS
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 New Chat", key="new_conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.asked_questions = set()
            st.session_state.chief_complaint = None
            st.success("✅ Started new conversation")
            st.rerun()
    
    with col2:
        # Export conversation
        if st.session_state.messages and st.button("💾 Export", key="export_conversation"):
            conversation_text = ""
            for msg in st.session_state.messages:
                role = "Patient" if msg["role"] == "user" else "AI Doctor"
                conversation_text += f"{role}: {msg['content']}\n\n"
            
            st.download_button(
                label="📥 Download",
                data=conversation_text,
                file_name=f"medical_consultation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================
st.title("💬 Medical Consultation Chat")

# Welcome message
if not st.session_state.messages:
    st.info("""
    👋 **Welcome to Medical AI Assistant!**
    
    I'm here to help you understand your health concerns. You can:
    - Describe your symptoms
    - Upload medical reports (JSON, PDF)
    - Share medical images (X-rays, photos)
    - Ask questions about your health
    
    **Note:** I'm an AI assistant for educational purposes. Always consult with healthcare professionals for medical decisions.
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"]:
            st.image(message["image"], width=300)

# ============================================================================
# INPUT SECTION: Text + Audio
# ============================================================================
input_col1, input_col2 = st.columns([6, 1])

# Audio input
with input_col2:
    st.markdown("<div style='margin-top: 8px;'>", unsafe_allow_html=True)
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="1x",
        pause_threshold=2.0,
        sample_rate=41_000
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Handle audio input
if audio_bytes and audio_bytes != st.session_state.audio_bytes:
    st.session_state.audio_bytes = audio_bytes
    
    with st.spinner("🎤 Transcribing audio..."):
        transcribed_text = transcribe_audio(audio_bytes)
    
    if transcribed_text:
        st.success(f"✅ Transcribed: {transcribed_text[:100]}...")
        
        # Add user message
        user_msg = {"role": "user", "content": transcribed_text}
        st.session_state.messages.append(user_msg)
        
        # Generate response
        with st.spinner("🤔 Analyzing..."):
            try:
                response = generate_response(
                    transcribed_text,
                    st.session_state.messages,
                    st.session_state.patient_data or "",
                    st.session_state.uploaded_image
                )
            except Exception as e:
                print(f"Error generating response: {e}")
                response = "⚠️ I encountered an error. Please try again."
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Text input
with input_col1:
    if prompt := st.chat_input("Describe your symptoms or ask a question..."):
        try:
            # Add user message
            user_msg = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_msg)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = generate_response(
                            prompt,
                            st.session_state.messages,
                            st.session_state.patient_data or "",
                            st.session_state.uploaded_image
                        )
                        st.markdown(response)
                    
                    except Exception as e:
                        error_msg = "⚠️ I encountered an error processing your message. Please try again."
                        st.error(error_msg)
                        response = error_msg
                        print(f"Error in chat interface: {e}")
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        except Exception as e:
            st.error("⚠️ An unexpected error occurred. Please refresh the page.")
            print(f"Critical error in chat interface: {e}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("⚠️ **Disclaimer:** Educational purposes only. Not a substitute for professional medical advice.")

with col2:
    st.caption("📚 **Sources:** Apley & Solomon, Harrison's, Robbins, Gray's Anatomy")

with col3:
    st.caption("💡 **Tip:** Type 'What do I have?' when ready for assessment")

# System info in footer
with st.expander("ℹ️ System Information", expanded=False):
    st.write(f"""
    - **Model:** {MODEL_NAME}
    - **Messages in conversation:** {len(st.session_state.messages)}
    - **Patient data loaded:** {'Yes' if st.session_state.patient_data else 'No'}
    - **Image uploaded:** {'Yes' if st.session_state.uploaded_image else 'No'}
    - **Textbook search:** {'Available' if book_collection else 'Unavailable'}
    - **Triage engine:** {'Active' if st.session_state.triage_engine else 'Inactive'}
    """)

# ============================================================================
# ERROR RECOVERY
# ============================================================================
# Add a reset button in case of persistent errors
if st.session_state.system_status.get("last_error"):
    with st.sidebar:
        st.markdown("---")
        if st.button("🔧 Reset System", key="reset_system", type="secondary"):
            # Clear problematic state
            for key in list(st.session_state.keys()):
                if key not in ['messages', 'patient_data', 'uploaded_image']:
                    del st.session_state[key]
            st.success("✅ System reset. Please refresh the page.")
            time.sleep(2)
            st.rerun()