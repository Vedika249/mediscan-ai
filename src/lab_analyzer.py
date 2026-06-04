import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Normal reference ranges for common lab values
NORMAL_RANGES = {
    "glucose":            {"min": 70,   "max": 100,  "unit": "mg/dL",  "name": "Blood Glucose"},
    "hba1c":              {"min": 4.0,  "max": 5.7,  "unit": "%",      "name": "HbA1c"},
    "systolic_bp":        {"min": 90,   "max": 120,  "unit": "mmHg",   "name": "Systolic BP"},
    "diastolic_bp":       {"min": 60,   "max": 80,   "unit": "mmHg",   "name": "Diastolic BP"},
    "heart_rate":         {"min": 60,   "max": 100,  "unit": "bpm",    "name": "Heart Rate"},
    "hemoglobin":         {"min": 12.0, "max": 17.5, "unit": "g/dL",   "name": "Hemoglobin"},
    "wbc":                {"min": 4.5,  "max": 11.0, "unit": "K/uL",   "name": "WBC Count"},
    "platelets":          {"min": 150,  "max": 400,  "unit": "K/uL",   "name": "Platelets"},
    "creatinine":         {"min": 0.6,  "max": 1.2,  "unit": "mg/dL",  "name": "Creatinine"},
    "cholesterol":        {"min": 0,    "max": 200,  "unit": "mg/dL",  "name": "Total Cholesterol"},
    "hdl":                {"min": 40,   "max": 999,  "unit": "mg/dL",  "name": "HDL Cholesterol"},
    "ldl":                {"min": 0,    "max": 100,  "unit": "mg/dL",  "name": "LDL Cholesterol"},
    "triglycerides":      {"min": 0,    "max": 150,  "unit": "mg/dL",  "name": "Triglycerides"},
    "tsh":                {"min": 0.4,  "max": 4.0,  "unit": "mIU/L",  "name": "TSH (Thyroid)"},
    "sodium":             {"min": 136,  "max": 145,  "unit": "mEq/L",  "name": "Sodium"},
    "potassium":          {"min": 3.5,  "max": 5.0,  "unit": "mEq/L",  "name": "Potassium"},
    "temperature":        {"min": 97.0, "max": 99.5, "unit": "°F",     "name": "Body Temperature"},
    "oxygen_saturation":  {"min": 95,   "max": 100,  "unit": "%",      "name": "SpO2"},
    "bmi":                {"min": 18.5, "max": 24.9, "unit": "kg/m²",  "name": "BMI"},
}


def check_lab_values(lab_data: dict) -> dict:
    normal = []
    abnormal = []
    critical = []

    for key, value in lab_data.items():
        if key not in NORMAL_RANGES or value is None or value == "":
            continue

        ref = NORMAL_RANGES[key]
        val = float(value)
        label = ref["name"]
        unit = ref["unit"]

        deviation = None
        if val < ref["min"]:
            deviation = "LOW"
            pct = ((ref["min"] - val) / ref["min"]) * 100
        elif val > ref["max"]:
            deviation = "HIGH"
            pct = ((val - ref["max"]) / ref["max"]) * 100
        else:
            pct = 0

        entry = {
            "parameter": label,
            "value": val,
            "unit": unit,
            "normal_range": f"{ref['min']} - {ref['max']}",
            "status": deviation or "NORMAL",
        }

        if deviation is None:
            normal.append(entry)
        elif pct > 50:
            entry["severity"] = "CRITICAL"
            critical.append(entry)
        else:
            entry["severity"] = "MILD" if pct < 20 else "MODERATE"
            abnormal.append(entry)

    return {
        "normal": normal,
        "abnormal": abnormal,
        "critical": critical,
        "total_checked": len(normal) + len(abnormal) + len(critical),
    }


def analyze_lab_results(lab_data: dict) -> dict:
    # Step 1: Rule-based check
    range_check = check_lab_values(lab_data)

    # Step 2: Build summary for Groq
    lab_text = "\n".join([
        f"{NORMAL_RANGES[k]['name']}: {v} {NORMAL_RANGES[k]['unit']}"
        for k, v in lab_data.items()
        if k in NORMAL_RANGES and v not in [None, ""]
    ])

    if not lab_text:
        return {"error": "No valid lab values provided."}

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a clinical lab specialist AI.
Analyze the provided lab values and return ONLY raw JSON with no markdown or code blocks.
Start your response with { and end with }.
Use this exact structure:
{
  "overall_risk": "low | medium | high | critical",
  "risk_reason": "brief explanation",
  "possible_conditions": ["conditions suggested by these values"],
  "urgent_attention": ["values needing immediate attention if any"],
  "lifestyle_recommendations": ["diet, exercise, lifestyle changes"],
  "follow_up_tests": ["recommended follow-up tests"],
  "plain_summary": "2-3 sentences in simple language for the patient"
}"""
                },
                {
                    "role": "user",
                    "content": f"Analyze these lab results:\n\n{lab_text}"
                }
            ],
            temperature=0.2,
            max_tokens=1000,
        )

        raw = response.choices[0].message.content.strip()

        # Clean markdown if model adds it
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.strip().startswith("{"):
                    raw = part.strip()
                    break

        raw = raw.strip()
        ai_analysis = json.loads(raw)

    except Exception as e:
        ai_analysis = {"error": f"AI analysis failed: {str(e)}"}

    return {
        "range_check": range_check,
        "ai_analysis": ai_analysis,
    }


def calculate_risk_score(lab_data: dict) -> int:
    result = check_lab_values(lab_data)
    score = 0
    score += len(result["abnormal"]) * 10
    score += len(result["critical"]) * 25
    return min(score, 100)