import spacy
import re

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("Run: python -m spacy download en_core_web_sm")

SYMPTOM_KEYWORDS = [
    "fever", "cough", "pain", "fatigue", "nausea", "vomiting", "dizziness",
    "headache", "chills", "sweating", "shortness of breath", "chest pain",
    "back pain", "joint pain", "sore throat", "runny nose", "diarrhea",
    "constipation", "bloating", "rash", "itching", "swelling", "weakness",
    "loss of appetite", "weight loss", "weight gain", "insomnia", "anxiety",
    "depression", "palpitations", "blurred vision", "numbness", "tingling",
]

DISEASE_KEYWORDS = [
    "diabetes", "hypertension", "cancer", "asthma", "pneumonia", "tuberculosis",
    "covid", "influenza", "arthritis", "anemia", "hypothyroidism", "hyperthyroidism",
    "kidney disease", "liver disease", "heart disease", "stroke", "epilepsy",
    "migraine", "alzheimer", "parkinson", "depression", "anxiety disorder",
    "obesity", "malaria", "typhoid", "dengue", "hepatitis",
]

MEDICATION_KEYWORDS = [
    "paracetamol", "ibuprofen", "aspirin", "amoxicillin", "metformin",
    "insulin", "lisinopril", "atorvastatin", "omeprazole", "azithromycin",
    "ciprofloxacin", "prednisone", "levothyroxine", "amlodipine", "losartan",
    "metoprolol", "gabapentin", "sertraline", "fluoxetine", "cetirizine",
    "tablet", "capsule", "syrup", "injection",
]


def extract_medical_entities(text: str) -> dict:
    text_lower = text.lower()
    doc = nlp(text)

    found_symptoms   = [s for s in SYMPTOM_KEYWORDS if s in text_lower]
    found_diseases   = [d for d in DISEASE_KEYWORDS if d in text_lower]
    found_medications = [m for m in MEDICATION_KEYWORDS if m in text_lower]

    named_entities = []
    body_parts = []

    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "TIME", "QUANTITY"]:
            named_entities.append({"text": ent.text, "label": ent.label_})

    measurements = re.findall(
        r"\b\d+\.?\d*\s*(?:mg|ml|kg|lb|°F|°C|mmHg|bpm|%)\b",
        text, re.IGNORECASE
    )

    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    return {
        "symptoms":   list(set(found_symptoms)),
        "diseases":   list(set(found_diseases)),
        "medications": list(set(found_medications)),
        "body_parts": list(set(body_parts)),
        "measurements": list(set(measurements)),
        "dates":      list(set(dates)),
        "named_entities": named_entities,
        "total_entities_found": len(found_symptoms) + len(found_diseases) + len(found_medications),
    }


def get_entity_summary(entities: dict) -> str:
    parts = []
    if entities["symptoms"]:
        parts.append(f"Symptoms: {', '.join(entities['symptoms'])}")
    if entities["diseases"]:
        parts.append(f"Conditions: {', '.join(entities['diseases'])}")
    if entities["medications"]:
        parts.append(f"Medications: {', '.join(entities['medications'])}")
    if entities["measurements"]:
        parts.append(f"Measurements: {', '.join(entities['measurements'])}")
    return " | ".join(parts) if parts else "No specific medical entities detected."