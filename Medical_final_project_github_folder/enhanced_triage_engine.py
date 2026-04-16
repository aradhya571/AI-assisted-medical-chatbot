"""
enhanced_triage_engine.py
=========================
Architecture
------------
  Layer 1 — LLM Extraction (temp=0, strict JSON)
      Translates messy natural language into a canonical structured fact sheet.
      Handles negation, paraphrasing, pain scale, duration, history.
      The LLM never assigns a score or severity — it only normalises language.

  Layer 2 — Deterministic Scoring Engine (pure Python)
      Receives the structured fact sheet.
      Applies fixed scoring rules.
      Produces the same score every time for the same inputs. No randomness.

  Layer 3 — SQL Routing
      Given detected conditions + final severity, queries the DB for departments
      and doctors. Fully deterministic. Editable without touching Python.

Usage (from app.py)
-------------------
  engine = MedicalTriageEngine(groq_api_key=KEY, db_path="doctors.db")
  triage_result, image_cache = engine.analyze_symptoms(
      conversation_text    = full_user_text,
      lab_data             = patient_context,
      conversation_history = chat_history,
      uploaded_image       = image_obj,
      cached_image_analysis= st.session_state.image_triage_cache
  )
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import base64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class SeverityLevel(Enum):
    EMERGENCY = "🚨 EMERGENCY"
    SEVERE    = "🔴 SEVERE"
    MODERATE  = "🟡 MODERATE"
    MILD      = "🟢 MILD"
    MINIMAL   = "⚪ MINIMAL"

    @property
    def as_int(self) -> int:
        """Maps severity to the integer used in condition_routing.min_severity."""
        return {
            SeverityLevel.MINIMAL:   1,
            SeverityLevel.MILD:      2,
            SeverityLevel.MODERATE:  3,
            SeverityLevel.SEVERE:    4,
            SeverityLevel.EMERGENCY: 5,
        }[self]


@dataclass
class ImageAnalysis:
    findings:       List[str]
    severity_score: int
    red_flags:      List[str]
    image_type:     str
    confidence:     float


@dataclass
class DoctorRecommendation:
    name:       str
    department: str


@dataclass
class TriageScore:
    severity:                SeverityLevel
    score:                   int
    red_flags:               List[str]
    warning_signs:           List[str]
    recommendation:          str
    confidence:              float
    detected_conditions:     List[str]                  = field(default_factory=list)
    negated_conditions:      List[str]                  = field(default_factory=list)
    chief_complaint:         Optional[str]              = None
    image_findings:          Optional[List[str]]        = None
    lab_findings:            Optional[List[str]]        = None
    recommended_doctors:     List[DoctorRecommendation] = field(default_factory=list)
    recommended_departments: List[str]                  = field(default_factory=list)
    helpline_numbers:        List[str]                  = field(default_factory=list)
    scoring_trace:           List[str]                  = field(default_factory=list)


# =============================================================================
# CANONICAL CONDITION VOCABULARY
# =============================================================================
# This is the fixed vocabulary the LLM extractor maps patient language into.
# Scores and routing live in the SQL DB.
# To add a new condition: add it here AND add rows to doctors.sql.

CANONICAL_CONDITIONS: List[str] = [
    # Cardiac
    "chest_pain", "heart_attack", "crushing_chest_pain", "palpitations",
    "irregular_heartbeat",
    # Respiratory
    "difficulty_breathing", "shortness_of_breath", "cough", "asthma_attack",
    "pneumonia", "tuberculosis", "coughing_blood", "blue_lips",
    # Neurological
    "stroke", "slurred_speech", "facial_droop", "seizure", "sudden_confusion",
    "worst_headache_of_life", "headache", "migraine", "dizziness", "paralysis",
    "numbness",
    # Gastrointestinal
    "abdominal_pain", "severe_abdominal_pain", "vomiting", "vomiting_blood",
    "diarrhea", "black_stool", "blood_in_stool", "nausea", "jaundice",
    # Musculoskeletal
    "fracture", "joint_pain", "back_pain", "neck_pain", "sprain", "arthritis",
    "knee_pain", "shoulder_pain", "trauma",
    # Skin
    "rash", "itching", "allergic_reaction", "anaphylaxis", "wound", "burn",
    "acne", "spreading_redness",
    # OB/GYN
    "pregnancy_complication", "vaginal_bleeding", "menstrual_irregularity",
    "pelvic_pain", "labour_signs",
    # Pediatric
    "child_fever", "child_breathing_issue", "infant_emergency", "febrile_seizure",
    # Mental health
    "suicidal_ideation", "self_harm", "acute_psychosis", "depression",
    "anxiety_disorder", "panic_attack",
    # Kidney / Urinary
    "kidney_pain", "urinary_tract_infection", "blood_in_urine", "kidney_failure",
    "dialysis_issue",
    # ENT
    "ear_pain", "hearing_loss", "sore_throat", "sinusitis", "tonsillitis",
    "nosebleed", "throat_obstruction",
    # Oncology
    "cancer", "tumor", "unexplained_weight_loss", "chemotherapy_complication",
    # General / Systemic
    "fever", "high_fever", "fatigue", "weakness", "fainting", "unconscious",
    "severe_bleeding", "anemia", "infection", "sepsis",
]


# =============================================================================
# SCORING WEIGHTS
# =============================================================================
# Fixed point values per confirmed (non-negated) condition.
# These mirror the score_weight values in condition_routing rows.

CONDITION_SCORES: Dict[str, int] = {
    # Emergency tier
    "heart_attack":              25,
    "stroke":                    25,
    "unconscious":               25,
    "severe_bleeding":           20,
    "anaphylaxis":               25,
    "sepsis":                    25,
    "crushing_chest_pain":       20,
    "worst_headache_of_life":    15,
    "sudden_confusion":          15,
    "slurred_speech":            15,
    "facial_droop":              15,
    "seizure":                   18,
    "paralysis":                 18,
    "vomiting_blood":            15,
    "coughing_blood":            15,
    "black_stool":               12,
    "blue_lips":                 20,
    "throat_obstruction":        15,
    "difficulty_breathing":      15,
    "infant_emergency":          15,
    "febrile_seizure":           15,
    "suicidal_ideation":         20,
    "acute_psychosis":           15,
    "kidney_failure":            18,
    "labour_signs":              12,
    "pregnancy_complication":    10,
    # Severe tier
    "chest_pain":                15,
    "asthma_attack":             10,
    "pneumonia":                 12,
    "shortness_of_breath":        8,
    "severe_abdominal_pain":     12,
    "blood_in_stool":            10,
    "blood_in_urine":             8,
    "fracture":                  15,
    "trauma":                    10,
    "spreading_redness":         10,
    "burn":                       8,
    "high_fever":                 8,
    "fainting":                   8,
    "self_harm":                 12,
    "dialysis_issue":            15,
    "nosebleed":                  5,
    "vaginal_bleeding":           8,
    "chemotherapy_complication": 12,
    "tumor":                     15,
    "cancer":                    15,
    "tuberculosis":              10,
    "irregular_heartbeat":        6,
    "palpitations":               6,
    # Moderate tier
    "abdominal_pain":             8,
    "vomiting":                   5,
    "diarrhea":                   4,
    "jaundice":                   8,
    "headache":                   5,
    "migraine":                   6,
    "dizziness":                  4,
    "numbness":                   5,
    "joint_pain":                 4,
    "back_pain":                  4,
    "neck_pain":                  4,
    "knee_pain":                  4,
    "shoulder_pain":              4,
    "arthritis":                  4,
    "rash":                       4,
    "allergic_reaction":          5,
    "wound":                      4,
    "pelvic_pain":                5,
    "kidney_pain":                8,
    "urinary_tract_infection":    5,
    "ear_pain":                   3,
    "sinusitis":                  4,
    "tonsillitis":                4,
    "unexplained_weight_loss":    6,
    "panic_attack":               6,
    "infection":                  5,
    "fever":                      5,
    "anemia":                     5,
    "child_fever":                5,
    "child_breathing_issue":     10,
    "depression":                 4,
    "anxiety_disorder":           4,
    # Mild tier
    "cough":                      3,
    "nausea":                     3,
    "fatigue":                    3,
    "weakness":                   3,
    "itching":                    2,
    "acne":                       1,
    "sore_throat":                2,
    "hearing_loss":               4,
    "sprain":                     3,
    "menstrual_irregularity":     3,
}


# =============================================================================
# SYMPTOM CLUSTERS
# =============================================================================
# Bonus points when ALL conditions in a cluster are simultaneously present.
# Applied after individual scoring — rewards dangerous combinations.

SYMPTOM_CLUSTERS: List[Dict] = [
    {
        "name":       "Cardiac cluster (chest pain + shortness of breath)",
        "conditions": {"chest_pain", "shortness_of_breath"},
        "bonus":      20,
        "red_flag":   True,
    },
    {
        "name":       "Cardiac cluster (chest pain + difficulty breathing)",
        "conditions": {"chest_pain", "difficulty_breathing"},
        "bonus":      20,
        "red_flag":   True,
    },
    {
        "name":       "Stroke cluster (speech + facial droop)",
        "conditions": {"slurred_speech", "facial_droop"},
        "bonus":      20,
        "red_flag":   True,
    },
    {
        "name":       "Stroke cluster (speech + confusion)",
        "conditions": {"slurred_speech", "sudden_confusion"},
        "bonus":      20,
        "red_flag":   True,
    },
    {
        "name":       "Respiratory infection cluster (fever + SOB)",
        "conditions": {"fever", "shortness_of_breath"},
        "bonus":      15,
        "red_flag":   True,
    },
    {
        "name":       "Respiratory infection cluster (fever + difficulty breathing)",
        "conditions": {"fever", "difficulty_breathing"},
        "bonus":      15,
        "red_flag":   True,
    },
    {
        "name":       "GI bleed cluster",
        "conditions": {"vomiting_blood", "black_stool"},
        "bonus":      15,
        "red_flag":   True,
    },
    {
        "name":       "Sepsis cluster (fever + confusion + weakness)",
        "conditions": {"fever", "weakness", "sudden_confusion"},
        "bonus":      15,
        "red_flag":   True,
    },
    {
        "name":       "Anaphylaxis cluster",
        "conditions": {"allergic_reaction", "difficulty_breathing"},
        "bonus":      20,
        "red_flag":   True,
    },
]

# Conditions that alone indicate SEVERE minimum; two or more → EMERGENCY
EMERGENCY_CONDITIONS: set = {
    "heart_attack", "stroke", "unconscious", "severe_bleeding", "anaphylaxis",
    "sepsis", "crushing_chest_pain", "worst_headache_of_life", "sudden_confusion",
    "slurred_speech", "facial_droop", "seizure", "paralysis", "vomiting_blood",
    "coughing_blood", "black_stool", "blue_lips", "throat_obstruction",
    "difficulty_breathing", "infant_emergency", "febrile_seizure",
    "suicidal_ideation", "acute_psychosis", "kidney_failure",
}

HELPLINE_NUMBERS: List[str] = [
    "+91-8447333999",
    "+91-120-2333999",
]


# =============================================================================
# DATABASE INITIALISER
# =============================================================================

def _init_db(db_path: str, sql_path: str) -> None:
    """Create and seed the SQLite database from doctors.sql if it doesn't exist."""
    if not os.path.exists(db_path):
        logger.info(f"Initialising database at {db_path} from {sql_path}")
        with open(sql_path, "r") as f:
            sql = f.read()
        con = sqlite3.connect(db_path)
        con.executescript(sql)
        con.commit()
        con.close()
        logger.info("Database initialised.")
    else:
        logger.info(f"Using existing database at {db_path}")


# =============================================================================
# MAIN ENGINE
# =============================================================================

class MedicalTriageEngine:

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        db_path:      str           = "doctors.db",
        sql_path:     str           = "doctors.sql",
    ):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.db_path = db_path

        self.groq_client = None
        if self.api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.api_key)
                logger.info("Groq client ready.")
            except ImportError:
                logger.warning("groq package not installed.")

        if os.path.exists(sql_path):
            _init_db(db_path, sql_path)
        else:
            logger.warning(f"{sql_path} not found — doctor routing unavailable.")

    # =========================================================================
    # LAYER 1: LLM EXTRACTION
    # =========================================================================

    def _extract_clinical_facts(self, full_text: str) -> Dict:
        """
        Sends the full patient conversation to the LLM at temperature=0.

        The LLM's ONLY job: translate natural language into a structured
        clinical fact sheet using our canonical condition vocabulary.

        It does NOT score. It does NOT assign severity.
        It handles:
          - Paraphrasing: "my tummy is killing me" → abdominal_pain / severe_abdominal_pain
          - Negation: "I don't have fever" → fever goes into symptoms_negated
          - Pain scale extraction
          - Duration estimation
          - Relevant medical history

        Returns dict with keys:
          symptoms_present, symptoms_negated, pain_scale, duration_hours,
          patient_age, relevant_history, chief_complaint
        """
        if not self.groq_client:
            logger.warning("No Groq client — skipping LLM extraction.")
            return {}

        prompt = f"""
You are a clinical triage data extractor. Your ONLY job is to read the patient
conversation below and map what they said to canonical medical terms.

CANONICAL CONDITION LIST — use ONLY these exact strings, no others:
{json.dumps(CANONICAL_CONDITIONS, indent=2)}

PATIENT CONVERSATION:
\"\"\"
{full_text}
\"\"\"

INSTRUCTIONS:

1. Read the ENTIRE conversation carefully as a back-and-forth dialogue.
   The conversation is formatted as "Doctor: ..." and "Patient: ..." turns.
   Patient answers are often SHORT and IMPLICIT — you must resolve them
   using the Doctor's preceding question. Examples:
     Doctor: "Do you have sore throat, cough, or headache?"
     Patient: "yes, all"
     → Extract: sore_throat, cough, headache (all confirmed present)

     Doctor: "On a scale of 1-10, how bad is your headache?"
     Patient: "7"
     → Extract: pain_scale = 7

     Doctor: "Any difficulty breathing or nasal congestion?"
     Patient: "yes, it's annoying"
     → Extract: sinusitis or shortness_of_breath depending on context

     Doctor: "Any fever or chills?"
     Patient: "no not really"
     → fever goes into symptoms_NEGATED

2. Negation examples: "I don't have X", "no fever", "it's not X",
   "not really", "none of those", "I ruled out X".

2. Map what the patient describes to the closest canonical condition(s).
   Use the MOST SPECIFIC matching condition available. Examples:

   CHEST / CARDIAC:
     "something heavy on my chest" → "crushing_chest_pain"
     "feels like pressure in my chest" → "crushing_chest_pain"
     "tight feeling across my chest" → "crushing_chest_pain"
     "elephant sitting on my chest" → "crushing_chest_pain"
     "sharp stabbing chest pain" → "chest_pain"
     "mild chest discomfort" → "chest_pain"
     "I feel my heart racing" → "palpitations"

   BREATHING:
     "I cannot breathe properly" → "difficulty_breathing"
     "struggling to breathe" → "difficulty_breathing"
     "slightly short of breath" → "shortness_of_breath"

   ABDOMEN:
     "my tummy is killing me" → "severe_abdominal_pain"
     "stomach hurts badly" → "severe_abdominal_pain"
     "mild stomach ache" → "abdominal_pain"

   NEURO:
     "worst headache of my life" → "worst_headache_of_life"
     "thunderclap headache" → "worst_headache_of_life"
     "regular headache" → "headache"

3. If unsure whether something is present or negated, OMIT it entirely. Do not guess.

4. pain_scale: extract numeric 0-10 if mentioned, else null.

5. duration_hours: estimate in hours (24 for "since yesterday", 48 for "2 days",
   168 for "1 week"), else null.

6. qualitative_intensity: if the patient uses words like "intense", "severe",
   "constant", "unbearable", "excruciating", "really bad", "very strong" WITHOUT
   giving a numeric pain scale, set this to one of: "mild", "moderate", "severe",
   "very_severe". If they give a numeric pain scale instead, set this to null.

7. patient_age: integer if mentioned, else null.

8. relevant_history: list of pre-existing conditions mentioned (free text).

9. chief_complaint: one short sentence summarising the main problem.

Return ONLY valid JSON with exactly these keys. No explanation. No markdown.
{{
  "symptoms_present":      [],
  "symptoms_negated":      [],
  "pain_scale":            null,
  "qualitative_intensity": null,
  "duration_hours":        null,
  "patient_age":           null,
  "relevant_history":      [],
  "chief_complaint":       ""
}}
"""
        try:
            response = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a clinical data extractor. "
                            "Output only valid JSON. No explanations. No markdown."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            facts = json.loads(response.choices[0].message.content)
            logger.info(
                f"LLM extracted — present: {facts.get('symptoms_present', [])} "
                f"| negated: {facts.get('symptoms_negated', [])}"
            )
            return facts

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {}

    # =========================================================================
    # IMAGE ANALYSIS  (Vision API — expensive, cached by app.py)
    # =========================================================================

    def encode_image_to_base64(self, image_obj) -> str:
        buf = BytesIO()
        image_obj.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def analyze_medical_image(self, image_obj) -> Optional[ImageAnalysis]:
        """
        Sends the image to Groq Vision for clinical finding extraction.
        Deterministic scoring is applied to the extracted findings.
        App.py caches the result — this runs at most ONCE per image per session.
        """
        if not self.groq_client:
            return None

        prompt = """
You are a medical image triage analyst.

Return ONLY a JSON object with these keys:
{
  "image_type":   "X-ray | MRI | Wound | Skin | Lab report | Other",
  "findings":     ["finding 1", "finding 2"],
  "red_flags":    ["critical finding if any, else empty list"],
  "is_emergency": true or false,
  "confidence":   0-100
}

Set is_emergency = true ONLY for immediately life-threatening findings.
No explanations. Return only the JSON.
"""
        try:
            b64 = self.encode_image_to_base64(image_obj)
            response = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                temperature=0,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)

            # Deterministic scoring from findings
            score = 30 if data.get("is_emergency") else 0
            image_kw_scores = {
                "fracture": 15, "pneumonia": 12, "tumor": 15, "mass": 12,
                "obstruction": 15, "bleeding": 12, "infarct": 20,
                "stroke": 20, "aneurysm": 20,
            }
            for finding in data.get("findings", []):
                fl = finding.lower()
                for kw, w in image_kw_scores.items():
                    if kw in fl:
                        score = max(score, w * 2)

            return ImageAnalysis(
                findings=data.get("findings", []),
                severity_score=min(score, 50),
                red_flags=data.get("red_flags", []),
                image_type=data.get("image_type", "Unknown"),
                confidence=data.get("confidence", 50) / 100.0,
            )

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return None

    # =========================================================================
    # LAB / PDF ANALYSIS  (fully deterministic)
    # =========================================================================

    def _analyze_lab_content(self, lab_text: str) -> Tuple[int, List[str], List[str]]:
        """
        Deterministic keyword scan of extracted PDF/lab text.
        Returns (score_boost, red_flags, warnings).
        No LLM involved — same output for same input, always.
        """
        score     = 0
        red_flags = []
        warnings  = []
        lab_lower = lab_text.lower()

        for kw in ["critical", "panic value", "alert", "stat"]:
            if kw in lab_lower:
                score += 15
                red_flags.append(f"Lab flag: {kw.upper()}")

        lab_conditions = {
            "fracture": 15, "pneumonia": 12, "sinusitis": 6,
            "tumor": 15, "mass": 12, "obstruction": 15, "anemia": 5,
            "infection": 8, "malignancy": 15, "infarct": 20, "sepsis": 20,
        }
        for condition, weight in lab_conditions.items():
            if condition in lab_lower:
                score += weight
                warnings.append(f"Report finding: {condition.title()}")

        abnormal_count = (
            lab_lower.count("high") +
            lab_lower.count("low") +
            lab_lower.count("positive") +
            lab_lower.count("abnormal")
        )
        if abnormal_count > 0:
            added = min(abnormal_count * 2, 10)
            score += added
            warnings.append(f"{abnormal_count} abnormal lab value(s) detected")

        return score, list(set(red_flags)), list(set(warnings))

    # =========================================================================
    # LAYER 2: DETERMINISTIC SCORING ENGINE
    # =========================================================================

    def _score(
        self,
        symptoms_present:      List[str],
        pain_scale:            Optional[int],
        qualitative_intensity: Optional[str],
        duration_hours:        Optional[int],
        lab_score:             int,
        lab_red_flags:         List[str],
        lab_warnings:          List[str],
        image_analysis:        Optional[ImageAnalysis],
    ) -> Tuple[int, List[str], List[str], List[str]]:
        """
        Pure Python scoring. Zero LLM calls. Zero randomness.
        Given the same structured inputs, produces the identical output every time.

        Returns (final_score, red_flags, warnings, audit_trace).
        """
        score     = 0
        red_flags = []
        warnings  = []
        trace     = []
        present   = set(symptoms_present)

        # A. Individual condition scores
        for condition in present:
            weight = CONDITION_SCORES.get(condition, 0)
            if weight > 0:
                score += weight
                trace.append(f"{condition}: +{weight}")
                if condition in EMERGENCY_CONDITIONS:
                    red_flags.append(condition.replace("_", " ").title())
                elif weight >= 8:
                    warnings.append(condition.replace("_", " ").title())

        # B. Symptom cluster bonuses
        for cluster in SYMPTOM_CLUSTERS:
            if cluster["conditions"].issubset(present):
                bonus = cluster["bonus"]
                score += bonus
                trace.append(f"Cluster '{cluster['name']}': +{bonus}")
                if cluster.get("red_flag"):
                    red_flags.append(cluster["name"])

        # C. Pain scale (numeric — takes priority over qualitative)
        if pain_scale is not None:
            if pain_scale >= 9:
                pts = 12
                warnings.append(f"Very severe pain ({pain_scale}/10)")
            elif pain_scale >= 7:
                pts = 8
                warnings.append(f"Severe pain ({pain_scale}/10)")
            elif pain_scale >= 4:
                pts = 5
                warnings.append(f"Moderate pain ({pain_scale}/10)")
            else:
                pts = 2
            score += pts
            trace.append(f"Pain scale {pain_scale}/10: +{pts}")

        # C2. Qualitative intensity — only used when NO numeric pain scale given.
        # Captures signals like "intense", "constant", "unbearable", "really bad"
        elif qualitative_intensity is not None:
            intensity_pts = {
                "very_severe": 10,
                "severe":       7,
                "moderate":     4,
                "mild":         1,
            }
            pts = intensity_pts.get(qualitative_intensity, 0)
            if pts:
                score += pts
                warnings.append(f"Qualitative intensity: {qualitative_intensity.replace('_', ' ')}")
                trace.append(f"Qualitative intensity ({qualitative_intensity}): +{pts}")

        # D. Duration
        if duration_hours is not None:
            if duration_hours >= 720:
                pts = 15
                warnings.append("Chronic symptoms (>1 month)")
            elif duration_hours >= 168:
                pts = 10
                warnings.append("Symptoms >1 week")
            elif duration_hours >= 72:
                pts = 5
                warnings.append("Symptoms >3 days")
            elif duration_hours >= 24:
                pts = 2
            elif duration_hours >= 3:  
                pts = 1    
            else:
                pts = 0
            if pts:
                score += pts
                trace.append(f"Duration ~{duration_hours}h: +{pts}")

        # E. Lab data
        if lab_score > 0:
            score += lab_score
            red_flags.extend(lab_red_flags)
            warnings.extend(lab_warnings)
            trace.append(f"Lab data: +{lab_score}")

        # F. Image analysis
        if image_analysis:
            score += image_analysis.severity_score
            red_flags.extend(image_analysis.red_flags)
            trace.append(f"Image analysis: +{image_analysis.severity_score}")

        # G. Cap score
        final_score = min(score, 100)
        # Soft cap: mild/moderate stacking cannot reach EMERGENCY without real red flags
        if not red_flags and final_score > 60:
            final_score = 60
            trace.append("Soft cap applied (no red flags): capped at 60")

        trace.append(f"FINAL SCORE: {final_score}")
        return final_score, list(set(red_flags)), list(set(warnings)), trace

    # =========================================================================
    # SEVERITY CLASSIFIER  (fully deterministic)
    # =========================================================================

    @staticmethod
    def _classify_severity(
        score:     int,
        red_flags: List[str],
        present:   set,
    ) -> SeverityLevel:
        """
        Maps score + red flags to a SeverityLevel. Strict priority order.
        Same inputs always produce the same SeverityLevel.
        """
        emergency_hits = len(present.intersection(EMERGENCY_CONDITIONS))

        if emergency_hits >= 2:
            return SeverityLevel.EMERGENCY
        if emergency_hits == 1:
            if score >= 40 or len(red_flags) >= 2:
                return SeverityLevel.EMERGENCY
            return SeverityLevel.SEVERE
        if score >= 30 or len(red_flags) >= 2:
            return SeverityLevel.EMERGENCY
        if score >= 20:
            return SeverityLevel.SEVERE
        if score >= 12:
            return SeverityLevel.MODERATE
        if score >= 6:
            return SeverityLevel.MILD
        return SeverityLevel.MINIMAL

    # =========================================================================
    # LAYER 3: SQL ROUTING  (fully deterministic)
    # =========================================================================

    def _get_departments_and_doctors(
        self,
        detected_conditions: List[str],
        severity:            SeverityLevel,
        max_per_dept:        int = 2,
    ) -> Tuple[List[str], List[DoctorRecommendation]]:
        """
        Queries the SQLite DB for relevant departments and doctors.
        Only departments whose min_severity <= current severity are included.
        Given the same inputs, always returns the same list.
        """
        if not os.path.exists(self.db_path) or not detected_conditions:
            return [], []

        severity_int  = severity.as_int
        placeholders  = ",".join("?" * len(detected_conditions))

        try:
            con = sqlite3.connect(self.db_path)
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            # Departments that match detected conditions at the current severity
            cur.execute(f"""
                SELECT DISTINCT d.id, d.name
                FROM departments d
                JOIN condition_routing cr ON cr.department_id = d.id
                WHERE cr.condition_key IN ({placeholders})
                  AND cr.min_severity <= ?
                ORDER BY d.name
            """, (*detected_conditions, severity_int))

            dept_rows  = cur.fetchall()
            dept_names = [r["name"] for r in dept_rows]
            dept_ids   = [r["id"]   for r in dept_rows]

            # Doctors per department (limited)
            doctors: List[DoctorRecommendation] = []
            for dept_id, dept_name in zip(dept_ids, dept_names):
                cur.execute(
                    "SELECT name FROM doctors WHERE department_id = ? LIMIT ?",
                    (dept_id, max_per_dept)
                )
                for row in cur.fetchall():
                    doctors.append(DoctorRecommendation(
                        name=row["name"], department=dept_name
                    ))

            con.close()
            return dept_names, doctors

        except Exception as e:
            logger.error(f"DB routing error: {e}")
            return [], []

    # =========================================================================
    # PUBLIC ENTRY POINT
    # =========================================================================

    def analyze_symptoms(
        self,
        conversation_text:     str                      = "",
        lab_data:              str                      = "",
        conversation_history:  Optional[List[Dict]]     = None,
        uploaded_image:        Optional[Any]             = None,
        cached_image_analysis: Optional[ImageAnalysis]  = None,
    ) -> Tuple[TriageScore, Optional[ImageAnalysis]]:
        """
        Main triage entry point. Called ONLY during diagnosis stage (gated by app.py).

        Parameters
        ----------
        conversation_text      Full concatenated user messages (built by app.py)
        lab_data               Extracted PDF / lab text (patient_context)
        conversation_history   Full message list — enriches conversation_text
        uploaded_image         PIL Image (if provided)
        cached_image_analysis  Previously computed ImageAnalysis — avoids re-calling Vision

        Returns
        -------
        (TriageScore, ImageAnalysis | None)
        The ImageAnalysis is returned so app.py can persist it in session state.
        """

        # Build a formatted Q&A transcript from the full conversation history.
        # We include BOTH roles (doctor questions + patient answers) so the LLM
        # extractor can resolve implicit patient replies like "yes, all" or "7"
        # that only make sense in context of the preceding question.
        if conversation_history:
            lines = []
            for msg in conversation_history:
                if not isinstance(msg.get("content"), str):
                    continue
                role_label = "Doctor" if msg["role"] == "assistant" else "Patient"
                lines.append(f"{role_label}: {msg['content']}")
            full_transcript = "\n".join(lines)
            # Append the current user query if not already in history
            conversation_text = (full_transcript + "\nPatient: " + conversation_text).strip()

        # ---------------------------------------------------------------
        # LAYER 1: LLM extraction — language → structured facts
        # ---------------------------------------------------------------
        facts = self._extract_clinical_facts(conversation_text)

        raw_present           = facts.get("symptoms_present", [])
        raw_negated           = facts.get("symptoms_negated",  [])
        pain_scale            = facts.get("pain_scale")
        qualitative_intensity = facts.get("qualitative_intensity")
        duration_hrs          = facts.get("duration_hours")
        chief                 = facts.get("chief_complaint", "")

        # Validate: only accept conditions in our canonical vocabulary
        # This prevents LLM hallucination from corrupting the scoring step
        valid_present = [c for c in raw_present if c in CANONICAL_CONDITIONS]
        valid_negated = [c for c in raw_negated if c in CANONICAL_CONDITIONS]

        # Critical step: remove anything the patient explicitly denied
        negated_set   = set(valid_negated)
        final_present = [c for c in valid_present if c not in negated_set]

        logger.info(f"Confirmed present: {final_present}")
        logger.info(f"Negated (excluded): {valid_negated}")

        # ---------------------------------------------------------------
        # Lab / PDF analysis (deterministic)
        # ---------------------------------------------------------------
        lab_score, lab_red, lab_warn = 0, [], []
        if lab_data:
            lab_score, lab_red, lab_warn = self._analyze_lab_content(lab_data)

        # ---------------------------------------------------------------
        # Image analysis — use cache if available
        # ---------------------------------------------------------------
        img_analysis = cached_image_analysis
        if uploaded_image is not None and img_analysis is None:
            logger.info("Running image analysis (not cached)...")
            img_analysis = self.analyze_medical_image(uploaded_image)

        # ---------------------------------------------------------------
        # LAYER 2: Deterministic scoring
        # ---------------------------------------------------------------
        final_score, red_flags, warnings, trace = self._score(
            symptoms_present=final_present,
            pain_scale=pain_scale,
            qualitative_intensity=qualitative_intensity,
            duration_hours=duration_hrs,
            lab_score=lab_score,
            lab_red_flags=lab_red,
            lab_warnings=lab_warn,
            image_analysis=img_analysis,
        )

        # ---------------------------------------------------------------
        # Severity classification (deterministic)
        # ---------------------------------------------------------------
        severity = self._classify_severity(
            score=final_score,
            red_flags=red_flags,
            present=set(final_present),
        )

        rec_map = {
            SeverityLevel.EMERGENCY: "🚨 **EMERGENCY:** Seek immediate medical care. Call emergency services now.",
            SeverityLevel.SEVERE:    "🔴 **URGENT:** Go to the Emergency Room or Urgent Care within 4-6 hours.",
            SeverityLevel.MODERATE:  "🟡 **CONSULT:** Schedule a doctor's appointment within 24-48 hours.",
            SeverityLevel.MILD:      "🟢 **MONITOR:** Watch your symptoms. See a doctor if they worsen.",
            SeverityLevel.MINIMAL:   "⚪ **SELF-CARE:** Likely minor. Rest, hydration, and self-monitoring advised.",
        }

        # ---------------------------------------------------------------
        # LAYER 3: SQL routing
        # Doctors shown on frontend ONLY for SEVERE and EMERGENCY.
        # Routing is still computed for MODERATE for backend logging,
        # but the doctors list will be empty.
        # ---------------------------------------------------------------
        dept_names, doctors, helplines = [], [], []
        if severity in [SeverityLevel.MODERATE, SeverityLevel.SEVERE, SeverityLevel.EMERGENCY]:
            dept_names, doctors = self._get_departments_and_doctors(
                detected_conditions=final_present,
                severity=severity,
                max_per_dept=2,
            )
            helplines = HELPLINE_NUMBERS.copy()

        # Confidence: how much data we had
        confidence = 0.2
        if final_present:                  confidence += 0.20
        if lab_data:                       confidence += 0.20
        if img_analysis:                   confidence += 0.20
        if len(conversation_text) > 150:   confidence += 0.10
        if pain_scale is not None:         confidence += 0.05
        if duration_hrs is not None:       confidence += 0.05

        logger.info(
            f"TRIAGE | Severity={severity.value} | Score={final_score} | "
            f"Conditions={final_present} | Depts={dept_names}"
        )
        logger.info(f"SCORE TRACE | {trace}")

        triage_result = TriageScore(
            severity=severity,
            score=final_score,
            red_flags=red_flags,
            warning_signs=warnings,
            recommendation=rec_map[severity],
            confidence=min(confidence, 1.0),
            detected_conditions=final_present,
            negated_conditions=valid_negated,
            chief_complaint=chief,
            image_findings=img_analysis.findings if img_analysis else None,
            lab_findings=lab_warn,
            recommended_doctors=doctors,
            recommended_departments=dept_names,
            helpline_numbers=helplines,
            scoring_trace=trace,
        )

        return triage_result, img_analysis