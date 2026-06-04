"""
Basic tests for MediScan AI pipeline.
Run with: pytest tests/test_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ner_extractor import extract_medical_entities, get_entity_summary
from lab_analyzer import check_lab_values, calculate_risk_score


# ─── NER Tests ────────────────────────────────────────────────────────────────

def test_ner_extracts_symptoms():
    text = "Patient has fever, cough, and headache since 3 days."
    result = extract_medical_entities(text)
    assert "fever" in result["symptoms"]
    assert "cough" in result["symptoms"]
    assert "headache" in result["symptoms"]


def test_ner_extracts_diseases():
    text = "Patient has a history of diabetes and hypertension."
    result = extract_medical_entities(text)
    assert "diabetes" in result["diseases"]
    assert "hypertension" in result["diseases"]


def test_ner_extracts_medications():
    text = "Currently on Paracetamol 500mg and Metformin 1000mg."
    result = extract_medical_entities(text)
    assert "paracetamol" in result["medications"] or "metformin" in result["medications"]


def test_ner_empty_text():
    result = extract_medical_entities("")
    assert result["symptoms"] == []
    assert result["diseases"] == []


def test_entity_summary_not_empty():
    text = "Patient has fever and is taking aspirin for pain."
    entities = extract_medical_entities(text)
    summary = get_entity_summary(entities)
    assert isinstance(summary, str)
    assert len(summary) > 0


# ─── Lab Tests ────────────────────────────────────────────────────────────────

def test_normal_glucose():
    result = check_lab_values({"glucose": 85})
    assert any(item["status"] == "NORMAL" for item in result["normal"])


def test_high_glucose_flagged():
    result = check_lab_values({"glucose": 250})
    statuses = [item["status"] for item in result["abnormal"] + result["critical"]]
    assert "HIGH" in statuses


def test_low_hemoglobin_flagged():
    result = check_lab_values({"hemoglobin": 8.0})
    statuses = [item["status"] for item in result["abnormal"] + result["critical"]]
    assert "LOW" in statuses


def test_risk_score_zero_for_normal():
    score = calculate_risk_score({"glucose": 85, "heart_rate": 75})
    assert score == 0


def test_risk_score_increases_with_abnormal():
    score_normal = calculate_risk_score({"glucose": 85})
    score_high   = calculate_risk_score({"glucose": 350})
    assert score_high > score_normal


def test_risk_score_capped_at_100():
    score = calculate_risk_score({
        "glucose": 500, "cholesterol": 400, "creatinine": 10,
        "systolic_bp": 200, "heart_rate": 180,
    })
    assert score <= 100