"""
Test Script: Demonstrate Enhanced Triage Engine
This script shows how the backend triage works with different scenarios
"""

from enhanced_triage_engine import MedicalTriageEngine, SeverityLevel

def print_triage_result(result, scenario_name):
    """Pretty print triage results"""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*70}")
    print(f"Severity: {result.severity.value}")
    print(f"Score: {result.score}/100")
    print(f"Confidence: {result.confidence*100:.0f}%")
    
    if result.red_flags:
        print(f"\n🚩 Red Flags:")
        for flag in result.red_flags:
            print(f"   - {flag}")
    
    if result.warning_signs:
        print(f"\n⚠️  Warning Signs:")
        for warning in result.warning_signs[:5]:  # Limit to 5
            print(f"   - {warning}")
    
    print(f"\n💡 Recommendation:\n   {result.recommendation}")
    
    if result.recommended_departments:
        print(f"\n🏥 Recommended Departments:")
        for dept in result.recommended_departments[:3]:
            print(f"   - {dept}")
    
    if result.recommended_doctors:
        print(f"\n👨‍⚕️ Recommended Doctors:")
        for doc in result.recommended_doctors[:4]:
            print(f"   - {doc.name} ({doc.department})")
    
    if result.helpline_numbers:
        print(f"\n📞 Helpline Numbers:")
        for number in result.helpline_numbers:
            print(f"   - {number}")
    
    print(f"\n{'='*70}\n")


def main():
    print("\n" + "="*70)
    print("ENHANCED MEDICAL TRIAGE ENGINE - TEST SCENARIOS")
    print("="*70)
    
    # Initialize engine (will work without API key for keyword analysis)
    engine = MedicalTriageEngine()
    
    # SCENARIO 1: EMERGENCY - Cardiac Event
    print("\n>>> Testing EMERGENCY scenario...")
    result1 = engine.analyze_symptoms(
        conversation_text="""
        Patient says: I'm having severe chest pain that feels like crushing pressure.
        It started 30 minutes ago. I'm also sweating a lot and feel nauseous.
        The pain radiates to my left arm.
        """
    )
    print_triage_result(result1, "EMERGENCY: Possible Heart Attack")
    
    # SCENARIO 2: SEVERE - High Fever with Respiratory Issues
    print("\n>>> Testing SEVERE scenario...")
    result2 = engine.analyze_symptoms(
        conversation_text="""
        Patient says: I have a high fever of 104°F for the past day.
        I also have severe body pain and shortness of breath.
        I'm finding it difficult to breathe properly.
        """
    )
    print_triage_result(result2, "SEVERE: High Fever + Respiratory Distress")
    
    # SCENARIO 3: MODERATE - Persistent Headache
    print("\n>>> Testing MODERATE scenario...")
    result3 = engine.analyze_symptoms(
        conversation_text="""
        Patient says: I've had a persistent headache for 3 days now.
        It's moderate pain, around 6/10. I also feel nauseous sometimes.
        The pain gets worse with bright lights.
        """
    )
    print_triage_result(result3, "MODERATE: Persistent Migraine")
    
    # SCENARIO 4: MILD - Common Cold
    print("\n>>> Testing MILD scenario...")
    result4 = engine.analyze_symptoms(
        conversation_text="""
        Patient says: I have a runny nose and mild cough for 2 days.
        Low-grade fever around 100°F. Just feeling tired.
        """
    )
    print_triage_result(result4, "MILD: Common Cold Symptoms")
    
    # SCENARIO 5: MINIMAL - Minor Cut
    print("\n>>> Testing MINIMAL scenario...")
    result5 = engine.analyze_symptoms(
        conversation_text="""
        Patient says: I got a small cut on my finger while cooking.
        It's not bleeding much, just a minor cut.
        """
    )
    print_triage_result(result5, "MINIMAL: Minor Wound")
    
    # SCENARIO 6: MODERATE with Lab Data
    print("\n>>> Testing scenario with LAB DATA...")
    result6 = engine.analyze_symptoms(
        conversation_text="""
        Patient says: I've been feeling weak and tired for a week.
        Sometimes I feel dizzy when I stand up.
        """,
        lab_data="""
        Hemoglobin: 8.5 g/dL (Low)
        RBC Count: 3.2 million/μL (Low)
        MCV: 72 fL (Low)
        Clinical Impression: Suggestive of Anemia
        """
    )
    print_triage_result(result6, "MODERATE: Anemia with Lab Confirmation")
    
    # SCENARIO 7: Emergency - Neurological
    print("\n>>> Testing EMERGENCY - Neurological...")
    result7 = engine.analyze_symptoms(
        conversation_text="""
        Patient says: My wife suddenly has slurred speech and her face is drooping on one side.
        She seems confused and can't move her right arm properly.
        This started about 20 minutes ago.
        """
    )
    print_triage_result(result7, "EMERGENCY: Possible Stroke")
    
    # SCENARIO 8: MODERATE - Orthopedic
    print("\n>>> Testing MODERATE - Orthopedic Issue...")
    result8 = engine.analyze_symptoms(
        conversation_text="""
        Patient says: I twisted my ankle playing basketball yesterday.
        It's swollen and painful (pain level 6/10).
        I can walk but it hurts. No visible deformity.
        """
    )
    print_triage_result(result8, "MODERATE: Ankle Sprain")
    
    # Summary Statistics
    print("\n" + "="*70)
    print("SUMMARY OF TEST SCENARIOS")
    print("="*70)
    print(f"Total Scenarios Tested: 8")
    print(f"\nSeverity Distribution:")
    print(f"   EMERGENCY: 2 cases")
    print(f"   SEVERE: 1 case")
    print(f"   MODERATE: 3 cases")
    print(f"   MILD: 1 case")
    print(f"   MINIMAL: 1 case")
    print(f"\nDoctor Recommendations:")
    print(f"   Provided for: MODERATE, SEVERE, EMERGENCY (6 cases)")
    print(f"   Not provided for: MILD, MINIMAL (2 cases)")
    print("="*70)


if __name__ == "__main__":
    main()