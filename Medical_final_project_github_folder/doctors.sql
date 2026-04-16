-- =============================================================================
-- MEDICAL TRIAGE DATABASE
-- =============================================================================
-- Three tables:
--   1. departments          — all medical departments
--   2. doctors              — all doctors, linked to a department
--   3. condition_routing    — which department handles which condition,
--                             and the MINIMUM severity level at which that
--                             department should be recommended.
--
-- Severity levels (used in condition_routing.min_severity):
--   1 = MINIMAL
--   2 = MILD
--   3 = MODERATE
--   4 = SEVERE
--   5 = EMERGENCY
--
-- The triage engine queries condition_routing at runtime:
--   SELECT DISTINCT department_id
--   FROM condition_routing
--   WHERE condition_key IN (<detected_conditions>)
--     AND min_severity <= <current_severity_int>
-- =============================================================================

PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- TABLE 1: departments
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS departments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT
);

INSERT INTO departments (name, description) VALUES
    ('Internal Medicine',        'General adult medicine, chronic disease management'),
    ('Critical Care',            'Intensive care, life-threatening emergencies'),
    ('General Medicine',         'Primary care, minor illnesses, first contact'),
    ('Cardiology',               'Heart and cardiovascular system'),
    ('Neuro Sciences',           'Brain, spine, and nervous system'),
    ('Respiratory Medicine',     'Lungs, airways, breathing disorders'),
    ('Gastroenterology',         'Digestive system, liver, stomach, bowel'),
    ('Orthopaedics',             'Bones, joints, muscles, fractures, sports injuries'),
    ('Obstetrics & Gynaecology', 'Pregnancy, childbirth, female reproductive health'),
    ('Pediatrics',               'Children and infants up to 18 years'),
    ('Psychiatry',               'Mental health, behavioural disorders'),
    ('Oncology',                 'Cancer diagnosis and treatment'),
    ('Radiology',                'Medical imaging, X-ray, MRI, CT interpretation'),
    ('ENT',                      'Ear, nose, throat, sinuses, hearing'),
    ('Urology',                  'Urinary tract, kidneys (surgical), bladder'),
    ('Nephrology',               'Kidneys (medical), dialysis, renal disease'),
    ('Dermatology',              'Skin, hair, nails, allergic reactions'),
    ('Anesthesiology',           'Anaesthesia, pain management, perioperative care');


-- -----------------------------------------------------------------------------
-- TABLE 2: doctors
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS doctors (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    department_id INTEGER NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
    phone         TEXT,
    notes         TEXT
);

INSERT INTO doctors (name, department_id, phone, notes) VALUES
    -- Internal Medicine (id=1)
    ('Dr. Ashok Kumar Agarwal',   1, NULL, NULL),
    ('Dr. Pankaj Bansal',         1, NULL, NULL),
    ('Dr. Anurag Prasad',         1, NULL, NULL),

    -- Critical Care (id=2)
    ('Dr. Ram Murti Sharma',      2, NULL, NULL),
    ('Dr. Ankit',                 2, NULL, NULL),

    -- General Medicine (id=3)
    ('Dr. Abhishek Deepak',       3, NULL, NULL),
    ('Dr. Ravinder Singh Ahlawat',3, NULL, NULL),

    -- Cardiology (id=4)
    ('Dr. Subhendu Mohanty',      4, NULL, NULL),

    -- Neuro Sciences (id=5)
    ('Dr. Ravindra Srivastava',   5, NULL, NULL),
    ('Dr. Showkat Nazir Wani',    5, NULL, NULL),

    -- Respiratory Medicine (id=6)
    ('Dr. Devendra Kumar',        6, NULL, NULL),
    ('Dr. Mohan Bandhu Gupta',    6, NULL, NULL),

    -- Gastroenterology (id=7)
    ('Dr. Abhishek Deepak',       7, NULL, NULL),

    -- Orthopaedics (id=8)
    ('Dr. Rajni Ranjan',          8, NULL, NULL),
    ('Dr. V. K. Gautam',          8, NULL, NULL),
    ('Dr. Kulbhushan Kamboj',     8, NULL, NULL),
    ('Dr. Nishit Palo',           8, NULL, NULL),
    ('Dr. Rakesh Kumar',          8, NULL, NULL),
    ('Dr. Rahul Kaul',            8, NULL, NULL),

    -- Obstetrics & Gynaecology (id=9)
    ('Dr. Ruchi Srivastava',      9, NULL, NULL),
    ('Dr. Neerja Goel',           9, NULL, NULL),
    ('Dr. Archana Mehta',         9, NULL, NULL),
    ('Dr. Samta Gupta',           9, NULL, NULL),
    ('Dr. Shelly Agarwal',        9, NULL, NULL),
    ('Dr. Megha Ranjan',          9, NULL, NULL),

    -- Pediatrics (id=10)
    ('Dr. R.K. Thapar',          10, NULL, NULL),
    ('Dr. Ranjit Ghuliani',      10, NULL, NULL),

    -- Psychiatry (id=11)
    ('Dr. Abhinit Kumar',        11, NULL, NULL),
    ('Dr. Kunal Kumar',          11, NULL, NULL),

    -- Oncology (id=12)
    ('Dr. Anil Thakwani',        12, NULL, NULL),

    -- Radiology (id=13)
    ('Dr. Vishal Gupta',         13, NULL, NULL),
    ('Dr. Khemendra Kumar',      13, NULL, NULL),

    -- ENT (id=14)
    ('Dr. Rohit Saxena',         14, NULL, NULL),
    ('Dr. Shubhi Tyagi',         14, NULL, NULL),
    ('Dr. Vivek Kumar Pathak',   14, NULL, NULL),

    -- Urology (id=15)
    ('Dr. Tarun Singh',          15, NULL, NULL),

    -- Nephrology (id=16)
    ('Dr. Bheem Raj Gupta',      16, NULL, NULL),

    -- Dermatology (id=17)
    ('Dr. Shitij Goel',          17, NULL, NULL),

    -- Anesthesiology (id=18)
    ('Dr. Ankit',                18, NULL, NULL);


-- -----------------------------------------------------------------------------
-- TABLE 3: condition_routing
-- -----------------------------------------------------------------------------
-- condition_key   : canonical symptom/condition name (matches what the LLM
--                   extractor returns — keep these lowercase, snake_case)
-- department_id   : which department handles this condition
-- min_severity    : minimum severity integer before this dept is recommended
--                   1=MINIMAL 2=MILD 3=MODERATE 4=SEVERE 5=EMERGENCY
-- score_weight    : how many points this condition adds to the triage score
--                   when detected as PRESENT (not negated)
-- is_emergency_flag: 1 if detecting this condition alone should push toward
--                   EMERGENCY regardless of score (e.g. stroke, heart attack)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS condition_routing (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_key      TEXT    NOT NULL,
    department_id      INTEGER NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
    min_severity       INTEGER NOT NULL DEFAULT 3,   -- 3 = MODERATE
    score_weight       INTEGER NOT NULL DEFAULT 5,
    is_emergency_flag  INTEGER NOT NULL DEFAULT 0    -- 0=false 1=true
);

-- Index for fast lookup by condition
CREATE INDEX IF NOT EXISTS idx_condition_key ON condition_routing(condition_key);

-- =============================================================================
-- CARDIAC
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('chest_pain',            4,  3, 15, 0),   -- Cardiology, show from MODERATE
    ('chest_pain',            2,  4, 15, 0),   -- Critical Care, show from SEVERE
    ('chest_pain',            1,  3, 15, 0),   -- Internal Medicine, show from MODERATE
    ('heart_attack',          4,  5, 25, 1),   -- Cardiology, EMERGENCY only
    ('heart_attack',          2,  5, 25, 1),   -- Critical Care, EMERGENCY only
    ('palpitations',          4,  2,  6, 0),   -- Cardiology, show from MILD
    ('palpitations',          1,  2,  6, 0),   -- Internal Medicine, from MILD
    ('irregular_heartbeat',   4,  3,  8, 0),
    ('crushing_chest_pain',   4,  5, 20, 1),
    ('crushing_chest_pain',   2,  5, 20, 1);

-- =============================================================================
-- RESPIRATORY
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('difficulty_breathing',  6,  4, 15, 0),   -- Respiratory, from SEVERE
    ('difficulty_breathing',  2,  5, 15, 1),   -- Critical Care, EMERGENCY
    ('shortness_of_breath',   6,  3,  8, 0),
    ('shortness_of_breath',   1,  3,  8, 0),
    ('cough',                 6,  2,  3, 0),
    ('cough',                 3,  2,  3, 0),
    ('asthma_attack',         6,  3, 10, 0),
    ('asthma_attack',         2,  4, 10, 0),
    ('pneumonia',             6,  3, 12, 0),
    ('pneumonia',             1,  3, 12, 0),
    ('tuberculosis',          6,  3, 10, 0),
    ('coughing_blood',        6,  4, 15, 1),
    ('coughing_blood',        2,  5, 15, 1),
    ('blue_lips',             2,  5, 20, 1);

-- =============================================================================
-- NEUROLOGICAL
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('stroke',                5,  5, 25, 1),
    ('stroke',                2,  5, 25, 1),
    ('slurred_speech',        5,  4, 15, 1),
    ('slurred_speech',        2,  5, 15, 1),
    ('facial_droop',          5,  4, 15, 1),
    ('seizure',               5,  4, 18, 1),
    ('seizure',               2,  5, 18, 1),
    ('sudden_confusion',      5,  4, 15, 1),
    ('sudden_confusion',      2,  5, 15, 1),
    ('worst_headache_of_life',5,  4, 15, 1),
    ('worst_headache_of_life',2,  5, 15, 1),
    ('headache',              5,  4,  5, 0),   -- Neuro only from SEVERE
    ('headache',              3,  2,  3, 0),   -- General Medicine from MILD
    ('migraine',              5,  3,  6, 0),
    ('dizziness',             5,  3,  4, 0),
    ('dizziness',             3,  2,  3, 0),
    ('paralysis',             5,  4, 18, 1),
    ('numbness',              5,  3,  5, 0),
    ('numbness',              8,  3,  5, 0);

-- =============================================================================
-- GASTROINTESTINAL
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('abdominal_pain',        7,  3,  8, 0),
    ('abdominal_pain',        3,  2,  5, 0),
    ('severe_abdominal_pain', 7,  4, 12, 0),
    ('severe_abdominal_pain', 2,  5, 12, 0),
    ('vomiting',              7,  2,  5, 0),
    ('vomiting',              3,  2,  5, 0),
    ('vomiting_blood',        7,  4, 15, 1),
    ('vomiting_blood',        2,  5, 15, 1),
    ('diarrhea',              7,  2,  4, 0),
    ('diarrhea',              3,  2,  4, 0),
    ('black_stool',           7,  4, 12, 1),
    ('black_stool',           2,  5, 12, 1),
    ('blood_in_stool',        7,  3, 10, 0),
    ('nausea',                7,  2,  3, 0),
    ('nausea',                3,  2,  3, 0),
    ('jaundice',              7,  3,  8, 0),
    ('jaundice',              1,  3,  8, 0);

-- =============================================================================
-- MUSCULOSKELETAL
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('fracture',              8,  3, 15, 0),
    ('fracture',              2,  4, 15, 0),
    ('joint_pain',            8,  2,  4, 0),
    ('back_pain',             8,  2,  4, 0),
    ('back_pain',             5,  4,  6, 0),
    ('neck_pain',             8,  2,  4, 0),
    ('neck_pain',             5,  4,  6, 0),
    ('sprain',                8,  2,  3, 0),
    ('arthritis',             8,  2,  4, 0),
    ('knee_pain',             8,  2,  4, 0),
    ('shoulder_pain',         8,  2,  4, 0),
    ('trauma',                8,  3, 10, 0),
    ('trauma',                2,  4, 10, 0);

-- =============================================================================
-- SKIN / DERMATOLOGY
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('rash',                  17, 2,  4, 0),
    ('rash',                  3,  2,  4, 0),
    ('itching',               17, 1,  2, 0),
    ('allergic_reaction',     17, 2,  5, 0),
    ('allergic_reaction',     3,  2,  5, 0),
    ('anaphylaxis',           2,  5, 25, 1),
    ('wound',                 17, 2,  4, 0),
    ('wound',                 3,  2,  4, 0),
    ('burn',                  17, 3,  8, 0),
    ('burn',                  2,  4,  8, 0),
    ('acne',                  17, 1,  1, 0),
    ('spreading_redness',     17, 4, 10, 0),
    ('spreading_redness',     2,  5, 10, 0);

-- =============================================================================
-- OBSTETRICS & GYNAECOLOGY
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('pregnancy_complication', 9, 3, 10, 0),
    ('pregnancy_complication', 2, 5, 10, 0),
    ('vaginal_bleeding',       9, 3,  8, 0),
    ('menstrual_irregularity', 9, 2,  3, 0),
    ('pelvic_pain',            9, 2,  5, 0),
    ('labour_signs',           9, 4, 12, 0),
    ('labour_signs',           2, 5, 12, 0);

-- =============================================================================
-- PEDIATRICS
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('child_fever',           10, 2,  5, 0),
    ('child_fever',            2, 4,  8, 0),
    ('child_breathing_issue', 10, 3, 10, 0),
    ('child_breathing_issue',  2, 4, 10, 0),
    ('infant_emergency',      10, 4, 15, 1),
    ('infant_emergency',       2, 5, 15, 1),
    ('febrile_seizure',       10, 4, 15, 1),
    ('febrile_seizure',        2, 5, 15, 1);

-- =============================================================================
-- MENTAL HEALTH / PSYCHIATRY
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('suicidal_ideation',     11, 4, 20, 1),
    ('suicidal_ideation',      2, 5, 20, 1),
    ('self_harm',             11, 3, 12, 0),
    ('self_harm',              2, 4, 12, 0),
    ('acute_psychosis',       11, 4, 15, 1),
    ('acute_psychosis',        2, 5, 15, 1),
    ('depression',            11, 2,  4, 0),
    ('anxiety_disorder',      11, 2,  4, 0),
    ('panic_attack',          11, 3,  6, 0),
    ('panic_attack',           3, 2,  4, 0);

-- =============================================================================
-- KIDNEY / URINARY
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('kidney_pain',           16, 3,  8, 0),
    ('kidney_pain',           15, 3,  8, 0),
    ('urinary_tract_infection',15,2,  5, 0),
    ('urinary_tract_infection', 3,2,  5, 0),
    ('blood_in_urine',        15, 3,  8, 0),
    ('blood_in_urine',        16, 3,  8, 0),
    ('kidney_failure',        16, 4, 18, 1),
    ('kidney_failure',         2, 5, 18, 1),
    ('dialysis_issue',        16, 4, 15, 0);

-- =============================================================================
-- ENT
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('ear_pain',              14, 2,  3, 0),
    ('hearing_loss',          14, 2,  4, 0),
    ('sore_throat',           14, 2,  3, 0),
    ('sore_throat',            3, 1,  2, 0),
    ('sinusitis',             14, 2,  4, 0),
    ('tonsillitis',           14, 2,  4, 0),
    ('nosebleed',             14, 3,  5, 0),
    ('nosebleed',              2, 4,  8, 0),
    ('throat_obstruction',    14, 4, 15, 1),
    ('throat_obstruction',     2, 5, 15, 1);

-- =============================================================================
-- ONCOLOGY
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('cancer',                12, 3, 15, 0),
    ('cancer',                 1, 3, 10, 0),
    ('tumor',                 12, 3, 15, 0),
    ('tumor',                  5, 3, 12, 0),
    ('unexplained_weight_loss',12,2,  6, 0),
    ('unexplained_weight_loss', 1,2,  5, 0),
    ('chemotherapy_complication',12,4,12,0),
    ('chemotherapy_complication', 2,5,12,0);

-- =============================================================================
-- GENERAL / SYSTEMIC
-- =============================================================================
INSERT INTO condition_routing (condition_key, department_id, min_severity, score_weight, is_emergency_flag) VALUES
    ('fever',                  3, 2,  5, 0),
    ('fever',                  1, 3,  5, 0),
    ('high_fever',             1, 3,  8, 0),
    ('high_fever',             2, 4,  8, 0),
    ('fatigue',                1, 2,  3, 0),
    ('weakness',               1, 2,  3, 0),
    ('weakness',               5, 3,  4, 0),
    ('fainting',               1, 3,  8, 0),
    ('fainting',               2, 4,  8, 0),
    ('unconscious',            2, 5, 25, 1),
    ('severe_bleeding',        2, 5, 20, 1),
    ('anemia',                 1, 2,  5, 0),
    ('infection',              3, 2,  5, 0),
    ('infection',              1, 3,  6, 0),
    ('sepsis',                 2, 5, 25, 1),
    ('sepsis',                 1, 4, 18, 0);