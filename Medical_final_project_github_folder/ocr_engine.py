import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager

import cv2
import numpy as np
import pytesseract
import pdfplumber
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from groq import Groq
from dotenv import load_dotenv

# --- DATA STRUCTURES ---
@dataclass
class TextBlock:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float

@dataclass
class ExtractedTable:
    extracted_source: str
    raw_text: str
    structured_data: List[Dict[str, Any]]
    page_number: int

@dataclass
class MedicalReport:
    filename: str
    metadata: Dict[str, Any]
    results: List[Dict[str, Any]]
    clinical_notes: str

class RobustMedicalOCR:
    def __init__(self, pdf_path: str, groq_api_key: Optional[str] = None):
        self.pdf_path = Path(pdf_path)
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    # --- STRATEGY 1: DIGITAL EXTRACTION ---
    def _extract_tables_digital(self, page) -> List[ExtractedTable]:
        tables = []
        extracted_tables = page.extract_tables()
        
        table_settings = {
            "vertical_strategy": "text", 
            "horizontal_strategy": "text",
            "intersection_x_tolerance": 15
        }
        
        if not extracted_tables:
            extracted_tables = page.extract_tables(table_settings)

        if extracted_tables:
            for table_data in extracted_tables:
                cleaned_data = [[cell if cell else "" for cell in row] for row in table_data]
                structured = self._structure_tabular_data(cleaned_data)
                raw_text = "\n".join([" | ".join(map(str, row)) for row in cleaned_data])
                
                tables.append(ExtractedTable(
                    extracted_source="pdfplumber_digital",
                    raw_text=raw_text,
                    structured_data=structured,
                    page_number=page.page_number
                ))
        return tables

    # --- STRATEGY 2: OCR FALLBACK ---
    def _cluster_text_into_rows(self, data: Dict) -> List[List[str]]:
        n_boxes = len(data['text'])
        blocks = []
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30 and data['text'][i].strip():
                blocks.append({
                    "text": data['text'][i],
                    "x": data['left'][i],
                    "y": data['top'][i],
                    "h": data['height'][i]
                })

        blocks.sort(key=lambda b: b['y'])
        rows = []
        if not blocks:
            return rows

        current_row = [blocks[0]]
        y_threshold = blocks[0]['h'] / 2 

        for block in blocks[1:]:
            last_block = current_row[-1]
            if abs(block['y'] - last_block['y']) < y_threshold:
                current_row.append(block)
            else:
                current_row.sort(key=lambda b: b['x'])
                rows.append(current_row)
                current_row = [block]
        
        if current_row:
            current_row.sort(key=lambda b: b['x'])
            rows.append(current_row)

        text_rows = []
        for row in rows:
            row_str = []
            for i, block in enumerate(row):
                if i > 0 and (block['x'] - (row[i-1]['x'] + len(row[i-1]['text']) * 10)) > 20:
                     row_str.append("|")
                row_str.append(block['text'])
            text_rows.append(" ".join(row_str))
            
        return text_rows

    def _extract_tables_ocr_fallback(self, image_path: str, page_num: int) -> List[ExtractedTable]:
        self.logger.info(f"Using OCR fallback for page {page_num}")
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
        text_rows = self._cluster_text_into_rows(data)
        raw_text = "\n".join(text_rows)
        
        return [ExtractedTable(
            extracted_source="ocr_fallback",
            raw_text=raw_text,
            structured_data=[],
            page_number=page_num
        )]

    # --- PROCESSING PIPELINE ---
    def process_document(self) -> Dict[str, Any]:
        full_text_context = ""
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # 1. Capture Header/Footer/Misc Text (Digital) 
                # ### NEW: This captures doctor names outside of tables
                raw_page_text = page.extract_text() or ""
                full_text_context += f"\n--- Page {i+1} Raw Text (Header/Footer) ---\n{raw_page_text}\n"

                # 2. Extract Tables (Digital)
                tables = self._extract_tables_digital(page)
                
                # 3. Check for Scanned Page
                # If page is mostly empty or tables failed, assume scanned
                if not tables and len(raw_page_text) < 50:
                    images = convert_from_path(str(self.pdf_path), first_page=i+1, last_page=i+1)
                    temp_img = f"temp_page_{i}.jpg"
                    images[0].save(temp_img)
                    
                    # OCR extracts everything (header + tables together)
                    ocr_tables = self._extract_tables_ocr_fallback(temp_img, i+1)
                    for t in ocr_tables:
                        full_text_context += f"\n--- Page {i+1} OCR Text ---\n{t.raw_text}"
                    os.remove(temp_img)
                else:
                    # Append structured table text for clarity
                    for t in tables:
                        full_text_context += f"\n--- Page {i+1} Table Data ---\n{t.raw_text}"

        return self._ai_parse_results(full_text_context)

    def _structure_tabular_data(self, rows: List[List[str]]) -> List[Dict]:
        structured = []
        headers = []
        for row in rows:
            row_str = " ".join(row).lower()
            if "test" in row_str and ("result" in row_str or "value" in row_str):
                headers = [h.lower() for h in row]
                continue
            
            if not headers:
                if len(row) >= 2:
                    structured.append({"parameter": row[0], "value": row[1]})
            else:
                item = {}
                for idx, cell in enumerate(row):
                    if idx < len(headers):
                        item[headers[idx]] = cell
                structured.append(item)
        return structured

    def _ai_parse_results(self, context_text: str) -> Dict[str, Any]:
        if not self.groq_client:
            return {"error": "No LLM Client", "raw_context": context_text}
            
        # ### UPDATED PROMPT: Added 'doctor' and 'lab_details' to JSON structure
        prompt = f"""
        Analyze the following text extracted from a Medical Lab Report.
        
        EXTRACT THE FOLLOWING JSON STRUCTURE:
        {{
            "patient": {{ 
                "name": "Extract patient name", 
                "age": "Extract age", 
                "gender": "Extract gender", 
                "id": "Extract Patient ID/MRN" 
            }},
            "doctor": {{
                "name": "Name of referring doctor or pathologist",
                "signature_text": "Text found near signature (e.g. 'Dr. Amit', 'MD Pathology')"
            }},
            "lab_details": {{
                "name": "Name of the lab/hospital",
                "report_date": "Date of report generation"
            }},
            "results": [
                {{
                    "test_name": "exact parameter name",
                    "value": "numeric value only",
                    "unit": "g/dL, %, etc",
                    "flag": "High/Low/Normal/Critical",
                    "reference_range": "extracted range"
                }}
            ],
            "clinical_notes": "summary of remarks or interpretation"
        }}

        RULES:
        1. Doctor names are often at the bottom (signature area) or top (referral).
        2. If multiple doctors are listed, prioritize the one signing the report.
        3. Infer full numbers if spaces exist (e.g. "1 50" -> 150).
        4. Return ONLY valid JSON.

        INPUT TEXT:
        {context_text}
        """

        try:
            response = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct", # Switched to 70b for better detail extraction
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"AI Parsing failed: {e}")
            return {"raw_text": context_text}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    # Run the extractor
    # Make sure you have a valid PDF path here
    extractor = RobustMedicalOCR("cbc-report-format.pdf", groq_api_key=api_key)
    data = extractor.process_document()
    
    output_filename = "medical_ocr_output.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Extraction Complete. Data saved to {output_filename}")