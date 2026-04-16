# Medical Consultation & Intelligent Triage System

An AI-powered medical consultation platform that combines OCR processing, medical RAG (Retrieval-Augmented Generation), and intelligent triage routing to help patients get connected with the right medical specialists.

## Features

- **Medical OCR Engine** - Extract and process medical documents (PDFs, lab reports, test results)
- **Medical RAG System** - Vector-based semantic search through medical textbooks and knowledge bases
- **Intelligent Triage Engine** - 3-layer architecture for symptom assessment and specialist routing:
  - Layer 1: LLM normalizes symptoms into structured medical facts
  - Layer 2: Deterministic scoring engine for consistent evaluation
  - Layer 3: SQL-based department and doctor recommendation
- **Streamlit Web Interface** - Interactive patient consultation platform
- **Real-time Processing** - Audio input support with medical consultation

## Project Structure

```
├── app.py                          # Primary Streamlit application
├── medical_consultation_app.py      # Enhanced Streamlit app with error handling
├── enhanced_triage_engine.py        # Core triage logic and scoring
├── ocr_engine.py                   # Medical document OCR processor
├── medical_rag_system.py           # RAG pipeline for medical knowledge
├── ingest_medical_books.py         # Vector database ingestion script
├── doctors.sql                     # Database schema (18 departments, 60+ doctors)
├── chroma_config.py                # ChromaDB configuration
├── requirements.txt                # Python dependencies
├── chroma_db/                      # Vector database storage
├── books/                          # Medical textbooks directory
└── test_triage.py                  # Testing script
```

## Prerequisites

- Python 3.8+
- Tesseract OCR (for PDF text extraction)
- Groq API key (free, get it at https://console.groq.com/keys)
- Medical textbooks (e.g., Davidson's Principles & Practice of Medicine)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/aradhya571/Medical_final_project_github_folder.git
cd Medical_final_project_github_folder
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Groq API key (get free key at https://console.groq.com/keys)
# On Windows:
# notepad .env
# On macOS/Linux:
# nano .env
```

Example `.env` file:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

5. **Initialize the database (optional):**
```bash
python ingest_medical_books.py
```

## Usage

### Run the Web Application
```bash
streamlit run app.py
```

### Test the Triage Engine
```bash
python test_triage.py
```

### Query the Medical Knowledge Base
```bash
python search_medical_db.py
```

### Process Medical Documents
```python
from ocr_engine import MedicalOCREngine
engine = MedicalOCREngine()
result = engine.extract_medical_data("path/to/document.pdf")
```

## Key Components

### Enhanced Triage Engine
- **Symptom Normalization**: Uses LLM to convert free-form patient input into structured medical facts
- **Scoring Algorithm**: Deterministic Python-based evaluation (no randomness, temperature=0)
- **Database Routing**: SQL queries to find appropriate departments and doctors based on conditions and severity

### Medical RAG System
- Ingests medical textbooks into ChromaDB vector store
- Uses SentenceTransformer embeddings for semantic search
- Provides context-aware medical information to LLM responses

### OCR Engine
- Processes PDFs using PyMuPDF and Tesseract
- Extracts structured medical data (lab values, test results)
- Uses Groq LLM for intelligent text interpretation

## Database Schema

The system uses SQLite with the following key tables:
- **Departments** - 18 medical specialties
- **Doctors** - 60+ healthcare professionals
- **Conditions** - Medical conditions with severity thresholds
- **Routing Rules** - Department and doctor recommendations based on conditions

## Technologies Used

- **Frontend**: Streamlit
- **LLM**: Groq (scout-17b-16e model)
- **Vector DB**: ChromaDB
- **OCR**: PyMuPDF, Tesseract, OpenCV
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **Database**: SQLite
- **Audio**: streamlit-audio-recorder

## Configuration

Edit [doctors.sql](doctors.sql) to customize:
- Medical departments and specialties
- Doctor profiles and availability
- Condition-to-department routing rules
- Severity thresholds

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Aradhya Mittal**
- GitHub: [@aradhya571](https://github.com/aradhya571)

## Acknowledgments

- Medical knowledge sourced from Davidson's Principles & Practice of Medicine
- Powered by Groq LLM API
- Built with Streamlit and ChromaDB
