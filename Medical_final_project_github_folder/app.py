import streamlit as st 
import json 
import chromadb 
import base64 
import os 
import tempfile 
from dotenv import load_dotenv
from chromadb.utils import embedding_functions 
from groq import Groq 
from PIL import Image 
import io 
from datetime import datetime 
from audio_recorder_streamlit import audio_recorder

# Load environment variables from .env file
load_dotenv() 

# IMPORT ENHANCED TRIAGE ENGINE 
try: 
    from enhanced_triage_engine import MedicalTriageEngine 
except ImportError: 
    st.error("⚠️ 'enhanced_triage_engine.py' not found. Please save it in the same folder.") 

# IMPORT YOUR OCR ENGINE 
try: 
    from ocr_engine import RobustMedicalOCR 
except ImportError: 
    st.error("⚠️ 'ocr_engine.py' not found. Please save your OCR code in the same folder.") 

# --- CONFIGURATION --- 
CHROMA_PATH = "./chroma_db" 
COLLECTION_NAME = "medical_knowledge_base" 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please set it as an environment variable or in .env file")
    st.stop()
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"


# --- INITIALIZE SESSION STATE --- 
if "messages" not in st.session_state: 
    st.session_state.messages = [] 
if "patient_data" not in st.session_state: 
    st.session_state.patient_data = None 
if "uploaded_image" not in st.session_state: 
    st.session_state.uploaded_image = None 
if "groq_client" not in st.session_state: 
    if GROQ_API_KEY.startswith("gsk_"): 
        st.session_state.groq_client = Groq(api_key=GROQ_API_KEY) 
if "asked_questions" not in st.session_state: 
    st.session_state.asked_questions = set() 
if "chief_complaint" not in st.session_state: 
    st.session_state.chief_complaint = None 
if "audio_bytes" not in st.session_state: 
    st.session_state.audio_bytes = None 
if "triage_engine" not in st.session_state:
    st.session_state.triage_engine = MedicalTriageEngine(
        groq_api_key=GROQ_API_KEY,
        db_path="doctors.db",    # SQLite DB auto-created from doctors.sql on first run
        sql_path="doctors.sql",  # must be in same folder as app.py
    )
# Cache for image analysis so we don't re-run Groq Vision on every call
if "image_triage_cache" not in st.session_state:
    st.session_state.image_triage_cache = None
# Track whether collected_info has been initialized
if "collected_info" not in st.session_state:
    st.session_state.collected_info = {
        # Basic demographics
        "age": False,
        "gender": False,
        
        # Core symptom characterization (OPQRST framework)
        "chief_complaint": False,
        "onset": False,  # When did it start? Sudden vs gradual?
        "duration": False,  # How long?
        "location": False,  # Where exactly?
        "quality": False,  # Type of pain/sensation (sharp, dull, burning, etc.)
        "severity": False,  # 1-10 scale
        "timing": False,  # Constant vs intermittent? Pattern?
        "triggers": False,  # What makes it better/worse?
        
        # Critical safety screening
        "red_flags_assessed": False,  # Checked for danger signs
        "associated_symptoms": False,  # Other symptoms occurring together
        
        # Context
        "medical_history": False,  # Past conditions, surgeries
        "medications": False,  # Current meds, allergies
        "recent_events": False,  # Trauma, travel, stress, new activities
    }

# --- SETUP DATABASE --- 
@st.cache_resource 
def get_book_collection(): 
    try: 
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH) 
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction( 
            model_name="all-MiniLM-L6-v2" 
        ) 
        collection = chroma_client.get_collection( 
            name=COLLECTION_NAME, 
            embedding_function=embedding_func 
        ) 
        return collection 
    except Exception as e: 
        return None 

book_collection = get_book_collection() 

# --- HELPER FUNCTIONS --- 
def format_json_to_text(medical_data): 
    summary = "**Patient Clinical Data**\n\n" 
    abnormal_findings = [] 
     
    # 1. Extract Patient Info 
    if 'patient' in medical_data: 
        p = medical_data['patient'] 
        summary += f"**Patient:** {p.get('name', 'N/A')} | **Age:** {p.get('age', 'N/A')} | **Gender:** {p.get('gender', 'N/A')}\n\n" 

    # 2. Extract Lab Results 
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

    # 3. Add Clinical Notes 
    if 'clinical_notes' in medical_data and medical_data['clinical_notes']: 
        summary += f"\n**Clinical Notes / Impression:**\n{medical_data['clinical_notes']}\n" 

    # 4. Add Doctor/Lab info 
    if 'doctor' in medical_data: 
        summary += f"\n**Doctor:** {medical_data['doctor'].get('name', 'N/A')}" 

    return summary, abnormal_findings 

def encode_image(image_obj): 
    buffered = io.BytesIO() 
    image_obj.save(buffered, format="JPEG") 
    return base64.b64encode(buffered.getvalue()).decode('utf-8') 

def transcribe_audio(audio_bytes): 
    """Transcribe audio using Groq's Whisper API with proper configuration""" 
    try: 
        client = st.session_state.groq_client 
        
        if not audio_bytes or len(audio_bytes) == 0:
            st.error("No audio data received")
            return None
         
        # Create temporary WAV file with proper handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio: 
            tmp_audio.write(audio_bytes) 
            tmp_audio_path = tmp_audio.name 
         
        try:
            with open(tmp_audio_path, "rb") as audio_file: 
                # Pass audio to Groq Whisper with explicit parameters for accuracy
                transcription = client.audio.transcriptions.create( 
                    file=("audio.wav", audio_file, "audio/wav"), 
                    model="whisper-large-v3-turbo",
                    language="en",  # Explicitly set to English
                    temperature=0    # Lower temperature for more accurate transcription
                ) 
            
            os.remove(tmp_audio_path) 
            
            # Extract text from response
            result_text = transcription.text if hasattr(transcription, 'text') else str(transcription)
            
            if result_text and result_text.strip():
                print(f"✓ Transcription successful: {result_text}")
                return result_text.strip()
            else:
                st.warning("Transcription returned empty result - try speaking again")
                return None
                
        except Exception as e:
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
            st.error(f"Transcription failed: {str(e)}")
            print(f"Transcription API error: {e}")
            return None
            
    except Exception as e: 
        st.error(f"Error processing audio: {str(e)}") 
        print(f"Audio processing error: {e}")
        return None 

def extract_keywords(text, image_obj=None): 
    client = st.session_state.groq_client 
     
    prompt = f"""Extract 3-5 key medical terms or conditions from this query for textbook search. 
    Query: {text} 
    Output only the terms, comma-separated (e.g., "Anemia, Hemoglobin, Iron Deficiency")""" 
     
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}] 
     
    if image_obj: 
        base64_img = encode_image(image_obj) 
        messages[0]["content"].append({ 
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"} 
        }) 

    try: 
        response = client.chat.completions.create( 
            messages=messages, 
            model=MODEL_NAME, 
            temperature=0.1 
        ) 
        return response.choices[0].message.content.strip() 
    except: 
        return text 


def update_collected_info(user_query, conversation_summary): 
    """Update collected_info flags based on user input using strict validation"""
    client = st.session_state.groq_client 
     
    prompt = f"""Analyze the patient's input and conversation history to determine what clinical information has been meaningfully captured.

**INFORMATION CATEGORIES:**
1. age: Patient's age
2. gender: Patient's gender/sex
3. chief_complaint: Main symptom or reason for consultation
4. onset: When the symptom started (sudden vs gradual)
5. duration: How long the symptom has lasted
6. location: Specific location of symptom (if applicable)
7. quality: Character/type of symptom (sharp, dull, throbbing, burning, etc.)
8. severity: Pain/symptom severity (preferably 1-10 scale)
9. timing: Pattern - constant, intermittent, worse at certain times
10. triggers: What makes it better or worse (movement, food, rest, etc.)
11. red_flags_assessed: Checked for danger signs (fever, sudden onset, neurological symptoms, chest pain, breathing difficulty, etc.)
12. associated_symptoms: Other symptoms occurring with main complaint
13. medical_history: Past medical conditions, surgeries, chronic diseases
14. medications: Current medications and drug allergies
15. recent_events: Recent trauma, travel, illness, stress, new activities

**CONVERSATION HISTORY:**
{conversation_summary}

**LATEST USER INPUT:**
{user_query}

**INSTRUCTIONS:**
- Mark a field as true ONLY if the patient has provided meaningful, specific information about it
- Vague or incomplete answers should remain false
- If patient mentions "no fever" or "no other symptoms", mark red_flags_assessed and associated_symptoms as true
- Be strict: "it hurts" doesn't capture severity; "a few days" is duration but "3 days" is better

Output ONLY valid JSON with boolean values for ALL 15 fields."""

    try: 
        response = client.chat.completions.create( 
            messages=[{"role": "user", "content": prompt}], 
            model=MODEL_NAME,
            response_format={"type": "json_object"}, 
            temperature=0 
        ) 
        new_info = json.loads(response.choices[0].message.content) 
        for key in st.session_state.collected_info: 
            if new_info.get(key): 
                st.session_state.collected_info[key] = True 
    except Exception as e: 
        print(f"Error updating collected info: {e}") 


def search_textbooks(query): 
    if not book_collection: 
        return "" 
     
    results = book_collection.query(query_texts=[query], n_results=5) 
     
    context = "" 
    if results['documents']: 
        for i, doc in enumerate(results['documents'][0]): 
            meta = results['metadatas'][0][i] 
            context += f"\n**Source: {meta.get('book_name', 'Medical Textbook')}**\n{doc}\n\n" 
    return context 

def extract_conversation_summary(chat_history): 
    summary = "" 
    for msg in chat_history: 
        if msg["role"] == "assistant": 
            if "?" in msg["content"]: 
                summary += f"Doctor asked: {msg['content']}\n" 
        elif msg["role"] == "user": 
            summary += f"Patient answered: {msg['content']}\n" 
    return summary 

def build_full_user_text(chat_history):
    """
    Builds a single string of all user messages across the conversation.
    Used to give the triage engine the complete symptom picture — not just
    the last message — so scoring reflects the entire interview.
    """
    parts = []
    for msg in chat_history:
        if msg["role"] == "user" and isinstance(msg["content"], str):
            parts.append(msg["content"])
    return " ".join(parts)

def format_doctor_recommendations(triage_result): 
    """Format doctor recommendations for display — only called for SEVERE/EMERGENCY or when explicitly requested by user""" 
    if not triage_result or not triage_result.recommended_doctors: 
        return "" 
     
    output = "\n\n---\n\n### 👨‍⚕️ Recommended Specialists\n\n" 
    output += "Based on your symptoms, I recommend consulting:\n\n" 
     
    from collections import defaultdict 
    dept_doctors = defaultdict(list) 
    for doc in triage_result.recommended_doctors: 
        dept_doctors[doc.department].append(doc.name) 
     
    for dept, doctors in dept_doctors.items(): 
        output += f"**{dept}:**\n" 
        for doctor in doctors: 
            output += f"- {doctor}\n" 
        output += "\n" 
     
    if triage_result.helpline_numbers: 
        output += "### 📞 Contact Information\n\n" 
        for number in triage_result.helpline_numbers: 
            output += f"- {number}\n" 
     
    return output 

def show_interview_progress():
    """Display what information has been collected - optional sidebar widget"""
    collected = sum(st.session_state.collected_info.values())
    total = len(st.session_state.collected_info)
    
    progress = collected / total if total > 0 else 0
    st.sidebar.progress(progress)
    st.sidebar.caption(f"Information gathered: {collected}/{total}")
    
    if collected < 10:
        st.sidebar.info("📋 Still gathering information...")
    elif collected < 18:
        st.sidebar.info("📋 Getting more details...")
    else:
        st.sidebar.success("✅ Ready for assessment!")

def generate_response(user_query, chat_history, patient_context="", image_obj=None): 
    client = st.session_state.groq_client 
     
    keywords = extract_keywords(user_query, image_obj) 
    book_context = search_textbooks(keywords) 
     
    explicit_request_keywords = [ 
        "what do i have", "what is wrong", "what's my diagnosis",  
        "tell me what", "explain", "what condition", "what disease", 
        "give me answer", "what is it", "diagnosis", "assessment" 
    ] 
    user_wants_answer = any(keyword in user_query.lower() for keyword in explicit_request_keywords) 
     
    conversation_summary = extract_conversation_summary(chat_history) 
    update_collected_info(user_query, conversation_summary) 
    questions_asked = sum(1 for msg in chat_history if msg["role"] == "assistant" and "?" in msg["content"]) 
     
    asked_questions_list = list(st.session_state.asked_questions) 
    asked_questions_text = "\n".join([f"- {q}" for q in asked_questions_list]) if asked_questions_list else "None." 

    # ===== DECISION LOGIC: When to move to diagnosis stage =====
    # More nuanced: require minimum critical info OR explicit request
    critical_items_collected = sum([
        st.session_state.collected_info["chief_complaint"],
        st.session_state.collected_info["duration"],
        st.session_state.collected_info["severity"],
        st.session_state.collected_info["red_flags_assessed"],
    ])
    enough_info_for_assessment = critical_items_collected >= 4
    is_diagnosis_stage = user_wants_answer or (questions_asked >= 6 and enough_info_for_assessment)
    
    # ===== INTERVIEW PRIORITY LOGIC =====
    # Determine what to ask next based on what's missing
    missing_critical = [k for k, v in st.session_state.collected_info.items() if not v]
    missing_count = len(missing_critical)
    
    if not st.session_state.collected_info["red_flags_assessed"]:
        priority = "red_flags"
    elif not st.session_state.collected_info["chief_complaint"]:
        priority = "chief_complaint"
    elif not all([st.session_state.collected_info[k] for k in ["onset", "duration", "severity"]]):
        priority = "basic_characterization"
    elif not st.session_state.collected_info["associated_symptoms"]:
        priority = "associated_symptoms"
    elif not all([st.session_state.collected_info[k] for k in ["medical_history", "medications"]]):
        priority = "medical_context"
    else:
        priority = "detailed_characterization" 

    # ------------------------------------------------------------------ 
    # TRIAGE: Only runs during DIAGNOSIS stage, not during interview.
    # This prevents partial/premature scoring on incomplete data, and
    # avoids burning API calls on every single user message.
    # ------------------------------------------------------------------ 
    triage_result = None

    if is_diagnosis_stage:
        # Build full conversation text from ALL user messages so the triage
        # engine scores the complete symptom picture, not just the last reply.
        full_user_text = build_full_user_text(chat_history)
        combined_text = full_user_text + " " + user_query

        # Pass cached image analysis to avoid re-running Groq Vision
        triage_result, updated_image_cache = st.session_state.triage_engine.analyze_symptoms(
            conversation_text=combined_text,
            lab_data=patient_context,
            conversation_history=chat_history,
            uploaded_image=image_obj,
            cached_image_analysis=st.session_state.image_triage_cache
        )

        # Persist image analysis result so future diagnosis calls reuse it
        if updated_image_cache is not None:
            st.session_state.image_triage_cache = updated_image_cache

        # Backend logging only — never shown on frontend
        print(f"🔍 TRIAGE: {triage_result.severity.value}, Score: {triage_result.score}")
        print(f"   Departments: {triage_result.recommended_departments}")
        print(f"   Doctors: {[d.name for d in triage_result.recommended_doctors]}")
    else:
        # Still in interview stage — skip triage entirely
        print(f"💬 INTERVIEW stage (Q#{questions_asked}) — triage skipped")

     
    if is_diagnosis_stage: 
        # FINAL DIAGNOSIS WITH DOCTOR RECOMMENDATIONS
        info_collected = sum(st.session_state.collected_info.values())
        system_prompt = f"""You are an experienced physician providing a clinical assessment after a thorough patient interview.

**COMPLETE INTERVIEW SUMMARY:**
{conversation_summary}

**INFORMATION GATHERED:** {info_collected}/22 clinical data points
**PATIENT DATA:** {patient_context if patient_context else "Not provided."}
**RELEVANT MEDICAL LITERATURE:** {book_context}

**YOUR CLINICAL ASSESSMENT SHOULD INCLUDE:**

**1. SYMPTOM SUMMARY (2-3 sentences)**
- Briefly restate the patient's presentation in clinical terms
- Example: "You're experiencing a sharp, right-sided headache that started 3 hours ago, rated 7/10 in severity, with no associated fever or neurological symptoms."

**2. DIFFERENTIAL DIAGNOSIS (Most likely conditions)**
- List 2-4 possible explanations in order of likelihood
- Use simple language: "This could be..." rather than medical jargon
- IMPORTANT: Only include conditions strongly suggested by the symptoms
- For each, briefly explain why it fits (1 sentence)
- Example format:
  * "Most likely: Tension headache - fits the pattern of unilateral sharp pain without red flags"
  * "Also possible: Migraine - though you haven't mentioned sensitivity to light or nausea"
  * "Less likely but consider: Cluster headache - would typically have eye watering or nasal symptoms"

**3. RED FLAGS ASSESSMENT**
- If ANY concerning features: clearly state "⚠️ Concerning features that warrant immediate evaluation:"
- Be specific about which symptoms are worrying and why
- If no red flags: reassure with "No immediate danger signs identified"

**4. HOME MANAGEMENT** (only for mild, non-urgent cases)
- Evidence-based self-care recommendations
- OTC medication suggestions with EXPLICIT WARNINGS about risks, interactions, and contraindications
  * Example: "Ibuprofen (Advil) 400mg may help, but avoid if you have stomach ulcers, kidney problems, or are pregnant. Do not exceed recommended dose."
- Non-pharmacological approaches (rest, hydration, heat/cold, positioning)

**5. WHEN TO SEEK CARE**
- Be specific: "See a doctor within 24 hours if..." vs "Seek emergency care immediately if..."
- List concrete scenarios that would warrant escalation
- Example: "Seek immediate care if: headache becomes sudden and severe, you develop confusion, weakness, vision loss, or high fever"

**6. NEXT STEPS**
- For mild cases: "Monitor for X days; if no improvement, consult your doctor"
- For moderate/concerning cases: "I recommend seeing a physician within [timeframe] for examination and possibly [tests]"
- For serious cases: "This warrants urgent medical evaluation"

**TONE & STYLE:**
- Empathetic but professional
- Explain medical reasoning in simple terms
- Acknowledge uncertainty where appropriate
- DO NOT be alarmist, but DO NOT downplay serious symptoms
- Keep total response under 400 words for readability

**CLOSING:**
End with: "Is there anything specific you'd like me to explain further?"

**CRITICAL: DO NOT:**
- Prescribe prescription medications
- Guarantee a diagnosis without examination
- Suggest invasive procedures
- Ask more questions (assessment phase is complete)
- Repeat information already discussed""" 
         
        messages = [ 
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": [{"type": "text", "text": user_query}]} 
        ] 
         
        if image_obj: 
            base64_img = encode_image(image_obj) 
            messages[1]["content"].append({ 
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"} 
            }) 

        response = client.chat.completions.create( 
            messages=messages, 
            model=MODEL_NAME, 
            temperature=0.7, 
            max_tokens=800 
        ) 
         
        ai_response = response.choices[0].message.content 
         
        if "?" in ai_response: 
            st.session_state.asked_questions.add(ai_response) 

        # Append doctor recommendations ONLY for MODERATE, SEVERE or EMERGENCY severity.
        # below: doctors are NOT shown on frontend.
        if triage_result:
            from enhanced_triage_engine import SeverityLevel
            if triage_result.severity in [SeverityLevel.MODERATE ,SeverityLevel.SEVERE, SeverityLevel.EMERGENCY]:
                ai_response += format_doctor_recommendations(triage_result)
         
        return ai_response 
    else: 
        # INTERVIEW STAGE — continue gathering information 
        system_prompt = f"""You are a compassionate, methodical physician conducting a patient interview.

**CONVERSATION SO FAR:**
{conversation_summary}

**PREVIOUSLY ASKED QUESTIONS:**
{asked_questions_text}

**INFORMATION COLLECTED:** {sum(st.session_state.collected_info.values())}/22 items
**STILL NEEDED:** {', '.join(missing_critical[:5]) if missing_critical else "Ready for assessment"}

**PATIENT DATA:** {patient_context if patient_context else "None yet."}
**TEXTBOOK CONTEXT:** {book_context if book_context else "General medical knowledge."}

**CURRENT INTERVIEW PRIORITY: {priority}**

**YOUR TASK - Ask ONE focused question based on priority:**

**If priority is "red_flags":**
- Ask about danger signs relevant to their complaint:
  * For headache: "Have you experienced any fever, neck stiffness, confusion, vision changes, or sudden severe onset?"
  * For chest pain: "Is the pain crushing/squeezing? Does it radiate to your arm/jaw? Any shortness of breath or sweating?"
  * For abdominal pain: "Any fever, vomiting blood, black stools, or severe constant pain?"
  * General: "Any difficulty breathing, chest pain, severe bleeding, or sudden neurological symptoms?"

**If priority is "chief_complaint":**
- Ask: "What is the main symptom that's bothering you?"

**If priority is "basic_characterization":**
- Focus on: When did this start? How long has it been going on? On a scale of 1-10, how severe is it?

**If priority is "associated_symptoms":**
- Ask: "Have you noticed any other symptoms along with your main complaint? For example, nausea, fatigue, changes in appetite, or anything else unusual?"

**If priority is "medical_context":**
- Ask about: Past medical conditions, current medications, drug allergies, or recent injuries/illnesses

**If priority is "detailed_characterization":**
- Explore: Exact location, quality/character of symptom, what makes it better/worse, pattern

**CRITICAL RULES:**
1. Ask ONLY ONE question per response
2. Keep it conversational and empathetic - you're a doctor, not a form
3. DO NOT repeat previously asked questions (check the list above)
4. If user mentions concerning symptoms, acknowledge urgency
5. Use simple, patient-friendly language
6. Be brief - 1-2 sentences maximum

Now ask your next question:""" 

        messages = [ 
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": [{"type": "text", "text": user_query}]} 
        ] 
         
        if image_obj: 
            base64_img = encode_image(image_obj) 
            messages[1]["content"].append({ 
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"} 
            }) 

        response = client.chat.completions.create( 
            messages=messages, 
            model=MODEL_NAME, 
            temperature=0.7, 
            max_tokens=200 
        ) 
         
        ai_response = response.choices[0].message.content 

        if "?" in ai_response: 
            st.session_state.asked_questions.add(ai_response) 

        return ai_response 


# --- PAGE CONFIG --- 
logo = Image.open("logo.png")
st.set_page_config(page_title="HealthAssist-AI", page_icon=logo, layout="wide") 

# --- CUSTOM CSS --- 
st.markdown(""" 
    <style> 
    /* Main container styling */ 
    .main { 
        background-color: #f8f9fa; 
    } 
     
    /* Chat message styling */ 
    .stChatMessage { 
        background-color: #grey; 
        border-radius: 10px; 
        padding: 15px; 
        margin-bottom: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    } 
     
    /* Sidebar styling */ 
    .css-1d391kg { 
        background-color: #ffffff; 
    } 
     
    /* Audio recorder styling - compact version */ 
    .stAudio { display: none; } 
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
    </style> 
""", unsafe_allow_html=True) 

# --- SIDEBAR: UPLOAD SECTION --- 
with st.sidebar: 
    st.title("HealthAssist-AI") 
    st.markdown("---") 
     
    st.subheader("📤 Upload Patient Data") 
     
    # tab1, tab2, tab3 = st.tabs(["📄 Text/JSON", "📑 PDF Report", "📷 Image"])
    tab1, tab2= st.tabs(["📑 PDF Report", "📷 Image"])
     
    # TAB 1: Text & JSON 
    # with tab1: 
    #     upload_type = st.radio("Format:", ["Text Input", "JSON File"], label_visibility="collapsed") 
         
    #     if upload_type == "Text Input": 
    #         text_input = st.text_area("Enter info/symptoms:", height=150, key="text_input") 
    #         if st.button("Load Text", key="load_text"): 
    #             if text_input: 
    #                 st.session_state.patient_data = text_input 
    #                 st.session_state.uploaded_image = None 
    #                 st.session_state.image_triage_cache = None  # reset cache on new context
    #                 st.success("✅ Text loaded!") 
    #                 st.rerun() 
    #     else: 
    #         json_file = st.file_uploader("Upload JSON:", type=["json"], key="json_upload") 
    #         if json_file: 
    #             data = json.load(json_file) 
    #             formatted, _ = format_json_to_text(data) 
    #             st.session_state.patient_data = formatted 
    #             st.session_state.uploaded_image = None 
    #             st.session_state.image_triage_cache = None  # reset cache on new context
    #             st.success("✅ JSON loaded!") 
    #             with st.expander("View Data"): 
    #                 st.markdown(formatted) 

    # TAB 1: PDF INTEGRATION 
    with tab1: 
        pdf_file = st.file_uploader("Upload Medical PDF:", type=["pdf"], key="pdf_upload") 
         
        if pdf_file and st.button("Process PDF"): 
            with st.spinner("Scanning document... (This may take a moment)"): 
                try: 
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file: 
                        tmp_file.write(pdf_file.getvalue()) 
                        tmp_path = tmp_file.name 

                    extractor = RobustMedicalOCR(pdf_path=tmp_path, groq_api_key=GROQ_API_KEY) 
                    extracted_data = extractor.process_document() 
                    formatted_text, _ = format_json_to_text(extracted_data) 
                    st.session_state.patient_data = formatted_text 
                    st.session_state.uploaded_image = None 
                    st.session_state.image_triage_cache = None  # reset cache on new context
                     
                    os.remove(tmp_path) 
                     
                    st.success("✅ PDF Processed Successfully!") 
                    with st.expander("View Extracted Data"): 
                        st.markdown(formatted_text) 
                         
                except Exception as e: 
                    st.error(f"Error processing PDF: {e}") 
                    if 'tmp_path' in locals() and os.path.exists(tmp_path): 
                        os.remove(tmp_path) 

    # TAB 2: Image Upload 
    with tab2: 
        img_file = st.file_uploader("Upload Image:", type=["jpg", "png", "jpeg"], key="img_upload") 
        if img_file: 
            img = Image.open(img_file) 
            st.session_state.uploaded_image = img 
            st.session_state.patient_data = "Medical image uploaded for analysis." 
            st.session_state.image_triage_cache = None  # reset cache — new image uploaded
            st.image(img, caption="Uploaded Image", use_container_width=True) 
            st.success("✅ Image loaded!") 
     
    st.markdown("---") 
     
    # Current Context Display 
    if st.session_state.patient_data or st.session_state.uploaded_image: 
        st.subheader("📋 Active Context") 
        if st.session_state.uploaded_image: 
            st.image(st.session_state.uploaded_image, width=150) 
        if st.session_state.patient_data: 
            with st.expander("View patient data"): 
                st.markdown(st.session_state.patient_data) 
         
        if st.button("🗑️ Clear Context", type="secondary"): 
            st.session_state.patient_data = None 
            st.session_state.uploaded_image = None 
            st.session_state.image_triage_cache = None  # reset cache on context clear
            st.rerun() 
     
    st.markdown("---") 
     
    if st.button("🔄 New Conversation"): 
        st.session_state.messages = [] 
        st.session_state.asked_questions = set() 
        st.session_state.chief_complaint = None 
        st.session_state.image_triage_cache = None  # reset cache on new conversation
        st.session_state.collected_info = {
            # Basic demographics
            "age": False,
            "gender": False,
            
            # Core symptom characterization (OPQRST framework)
            "chief_complaint": False,
            "onset": False,  # When did it start? Sudden vs gradual?
            "duration": False,  # How long?
            "location": False,  # Where exactly?
            "quality": False,  # Type of pain/sensation (sharp, dull, burning, etc.)
            "severity": False,  # 1-10 scale
            "timing": False,  # Constant vs intermittent? Pattern?
            "triggers": False,  # What makes it better/worse?
            
            # Critical safety screening
            "red_flags_assessed": False,  # Checked for danger signs
            "associated_symptoms": False,  # Other symptoms occurring together
            
            # Context
            "medical_history": False,  # Past conditions, surgeries
            "medications": False,  # Current meds, allergies
            "recent_events": False,  # Trauma, travel, stress, new activities
        }
        st.rerun() 

# --- MAIN CHAT INTERFACE --- 

st.title("Medical Consultation Chat") 

if not st.session_state.messages: 
    st.info("👋 Welcome! Tell me what's bothering you today, and I'll ask you some questions to better understand your symptoms.") 

# Display chat history 
for message in st.session_state.messages: 
    with st.chat_message(message["role"]): 
        st.markdown(message["content"]) 
        if "image" in message: 
            st.image(message["image"], width=300) 

# Chat input with audio option 
input_col1, input_col2 = st.columns([6, 1]) 

with input_col2: 
    st.markdown("<div style='margin-top: 8px;'>", unsafe_allow_html=True) 
    audio_bytes = audio_recorder( 
        text="", 
        recording_color="#e74c3c", 
        neutral_color="#3498db", 
        icon_name="microphone", 
        icon_size="1x", 
        pause_threshold=1.0, 
        sample_rate=16_000 
    ) 
    st.markdown("</div>", unsafe_allow_html=True) 

# Handle audio input 
if audio_bytes and audio_bytes != st.session_state.audio_bytes: 
    st.session_state.audio_bytes = audio_bytes 
     
    with st.spinner("Transcribing audio..."): 
        transcribed_text = transcribe_audio(audio_bytes) 
     
    if transcribed_text: 
        st.success(f"Transcribed: {transcribed_text}") 
         
        user_msg = {"role": "user", "content": transcribed_text} 
        st.session_state.messages.append(user_msg) 
         
        with st.spinner("Thinking..."): 
            response = generate_response( 
                transcribed_text, 
                st.session_state.messages, 
                st.session_state.patient_data or "", 
                st.session_state.uploaded_image 
            ) 
         
        st.session_state.messages.append({"role": "assistant", "content": response}) 
        st.rerun() 

# Text input 
with input_col1: 
    if prompt := st.chat_input("Describe your symptoms..."): 
        user_msg = {"role": "user", "content": prompt} 
        st.session_state.messages.append(user_msg) 
         
        with st.chat_message("user"): 
            st.markdown(prompt) 

        with st.chat_message("assistant"): 
            with st.spinner("Thinking..."): 
                response = generate_response( 
                    prompt, 
                    st.session_state.messages, 
                    st.session_state.patient_data or "", 
                    st.session_state.uploaded_image 
                ) 
                st.markdown(response) 
         
        st.session_state.messages.append({"role": "assistant", "content": response}) 

# Footer 
st.markdown("---") 
st.caption("⚠️ Educational purposes only. Not a substitute for professional medical advice.") 
st.caption("💡 Tip: Type 'What do I have?' when ready for assessment")