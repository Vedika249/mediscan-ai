import streamlit as st
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from groq_analyzer import analyze_medical_report, ask_medical_question
from ner_extractor import extract_medical_entities, get_entity_summary
from lab_analyzer import analyze_lab_results, calculate_risk_score

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MediScan AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .risk-low    { color: #27ae60; font-weight: bold; font-size: 1.2rem; }
    .risk-medium { color: #f39c12; font-weight: bold; font-size: 1.2rem; }
    .risk-high   { color: #e74c3c; font-weight: bold; font-size: 1.2rem; }
    .risk-critical { color: #8e0000; font-weight: bold; font-size: 1.2rem; }
    .entity-tag {
        display: inline-block;
        background: #e8f4fd;
        color: #1a73e8;
        padding: 3px 10px;
        border-radius: 12px;
        margin: 3px;
        font-size: 0.85rem;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🏥 MediScan AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Medical Report Analyzer & Disease Risk Predictor — Powered by LLaMA 3 via Groq</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="warning-box">⚠️ <b>Disclaimer:</b> This tool is for educational purposes only. Always consult a qualified doctor for medical advice.</div>', unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📄 Report Analyzer", "🧪 Lab Results", "💬 Ask MediScan"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Report Analyzer
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Paste Your Medical Report")

    sample_report = """Patient: John Doe, Age 45, Male
Chief Complaint: Persistent cough, fever (101°F), and fatigue for 5 days.
History: Patient reports shortness of breath, chest pain, and loss of appetite.
Medications: Currently taking Paracetamol 500mg twice daily.
Vitals: BP 140/90 mmHg, Heart Rate 95 bpm, Temperature 101.2°F, SpO2 94%.
Assessment: Possible pneumonia or upper respiratory infection. Patient also has a history of hypertension."""

    report_text = st.text_area(
        "Medical Report Text",
        value=sample_report,
        height=220,
        placeholder="Paste doctor's notes, discharge summary, clinical report...",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analyze Report", type="primary", use_container_width=True)

    if analyze_btn:
        if not report_text.strip():
            st.warning("Please enter a medical report.")
        else:
            with st.spinner("Analyzing report with LLaMA 3..."):
                analysis = analyze_medical_report(report_text)
                entities = extract_medical_entities(report_text)

            if "error" in analysis:
                st.error(f"Analysis failed: {analysis['error']}")
            else:
                st.success("✅ Analysis Complete!")

                # Severity Badge
                severity = analysis.get("severity", "unknown").lower()
                color_map = {
                    "low": "risk-low",
                    "medium": "risk-medium",
                    "high": "risk-high",
                    "critical": "risk-critical"
                }
                css_class = color_map.get(severity, "risk-medium")
                st.markdown(
                    f"**Overall Severity:** <span class='{css_class}'>{severity.upper()}</span> — {analysis.get('severity_reason', '')}",
                    unsafe_allow_html=True
                )

                st.markdown("---")
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("#### 🤒 Extracted Symptoms")
                    symptoms = analysis.get("extracted_symptoms", [])
                    if symptoms:
                        st.markdown(
                            " ".join([f'<span class="entity-tag">{s}</span>' for s in symptoms]),
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("None detected")

                    st.markdown("#### 💊 Medications Mentioned")
                    meds = analysis.get("medications_mentioned", [])
                    if meds:
                        st.markdown(
                            " ".join([f'<span class="entity-tag">{m}</span>' for m in meds]),
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("None detected")

                with col_b:
                    st.markdown("#### 🦠 Possible Diseases")
                    diseases = analysis.get("possible_diseases", [])
                    for d in diseases:
                        risk = d.get("risk_percentage", 0)
                        color = "#e74c3c" if risk > 70 else "#f39c12" if risk > 40 else "#27ae60"
                        st.markdown(
                            f"**{d.get('name', 'Unknown')}** — <span style='color:{color}'>{risk}% risk</span>",
                            unsafe_allow_html=True
                        )
                        st.caption(d.get("reason", ""))
                        st.progress(risk / 100)

                st.markdown("#### ✅ Recommendations")
                for rec in analysis.get("recommendations", []):
                    st.markdown(f"- {rec}")

                st.markdown("#### 📝 Summary")
                st.info(analysis.get("summary", ""))

                # NER Section
                with st.expander("🔬 NER Entity Extraction (spaCy)"):
                    st.markdown(f"**Entity Summary:** {get_entity_summary(entities)}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Symptoms:**", entities.get("symptoms", []))
                        st.write("**Diseases:**", entities.get("diseases", []))
                    with col2:
                        st.write("**Medications:**", entities.get("medications", []))
                        st.write("**Measurements:**", entities.get("measurements", []))
                    with col3:
                        st.write("**Dates:**", entities.get("dates", []))
                        st.write("**Total Entities:**", entities.get("total_entities_found", 0))

                with st.expander("🛠️ Raw JSON Output"):
                    st.json(analysis)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Lab Results
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Enter Your Lab Values")
    st.caption("Leave fields as 0 if not available.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🩸 Blood Tests**")
        glucose       = st.number_input("Glucose (mg/dL)",     min_value=0.0, value=0.0, step=0.1)
        hba1c         = st.number_input("HbA1c (%)",           min_value=0.0, value=0.0, step=0.1)
        hemoglobin    = st.number_input("Hemoglobin (g/dL)",   min_value=0.0, value=0.0, step=0.1)
        wbc           = st.number_input("WBC Count (K/uL)",    min_value=0.0, value=0.0, step=0.1)
        platelets     = st.number_input("Platelets (K/uL)",    min_value=0.0, value=0.0, step=1.0)

    with col2:
        st.markdown("**❤️ Cardiac & Vitals**")
        systolic_bp   = st.number_input("Systolic BP (mmHg)",  min_value=0.0, value=0.0, step=1.0)
        diastolic_bp  = st.number_input("Diastolic BP (mmHg)", min_value=0.0, value=0.0, step=1.0)
        heart_rate    = st.number_input("Heart Rate (bpm)",    min_value=0.0, value=0.0, step=1.0)
        temperature   = st.number_input("Temperature (°F)",    min_value=0.0, value=0.0, step=0.1)
        oxygen_saturation = st.number_input("SpO2 (%)",        min_value=0.0, value=0.0, step=0.1)

    with col3:
        st.markdown("**🫀 Lipid & Kidney**")
        cholesterol   = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0, value=0.0, step=1.0)
        hdl           = st.number_input("HDL (mg/dL)",               min_value=0.0, value=0.0, step=1.0)
        ldl           = st.number_input("LDL (mg/dL)",               min_value=0.0, value=0.0, step=1.0)
        triglycerides = st.number_input("Triglycerides (mg/dL)",     min_value=0.0, value=0.0, step=1.0)
        creatinine    = st.number_input("Creatinine (mg/dL)",        min_value=0.0, value=0.0, step=0.1)
        tsh           = st.number_input("TSH (mIU/L)",               min_value=0.0, value=0.0, step=0.1)

    lab_btn = st.button("🧪 Analyze Lab Results", type="primary")

    if lab_btn:
        lab_data = {
            k: v for k, v in {
                "glucose": glucose, "hba1c": hba1c, "hemoglobin": hemoglobin,
                "wbc": wbc, "platelets": platelets, "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp, "heart_rate": heart_rate,
                "temperature": temperature, "oxygen_saturation": oxygen_saturation,
                "cholesterol": cholesterol, "hdl": hdl, "ldl": ldl,
                "triglycerides": triglycerides, "creatinine": creatinine, "tsh": tsh,
            }.items()
            if v > 0
        }

        if not lab_data:
            st.warning("Please enter at least one lab value greater than 0.")
        else:
            with st.spinner("Analyzing lab values..."):
                result = analyze_lab_results(lab_data)
                risk_score = calculate_risk_score(lab_data)

            # Risk Score
            st.markdown("### Overall Risk Score")
            risk_color = "#27ae60" if risk_score < 30 else "#f39c12" if risk_score < 60 else "#e74c3c"
            st.markdown(f"<h2 style='color:{risk_color}'>{risk_score}/100</h2>", unsafe_allow_html=True)
            st.progress(risk_score / 100)

            range_check = result.get("range_check", {})
            ai = result.get("ai_analysis", {})

            col_a, col_b = st.columns(2)

            with col_a:
                if range_check.get("critical"):
                    st.markdown("#### 🚨 Critical Values")
                    for item in range_check["critical"]:
                        st.error(f"**{item['parameter']}**: {item['value']} {item['unit']} (Normal: {item['normal_range']}) — {item['status']}")

                if range_check.get("abnormal"):
                    st.markdown("#### ⚠️ Abnormal Values")
                    for item in range_check["abnormal"]:
                        st.warning(f"**{item['parameter']}**: {item['value']} {item['unit']} (Normal: {item['normal_range']}) — {item['status']}")

                if range_check.get("normal"):
                    st.markdown("#### ✅ Normal Values")
                    for item in range_check["normal"]:
                        st.success(f"**{item['parameter']}**: {item['value']} {item['unit']}")

            with col_b:
                if "error" not in ai:
                    st.markdown("#### 🤖 AI Interpretation")
                    overall = ai.get("overall_risk", "unknown")
                    st.markdown(f"**Overall Risk:** `{overall.upper()}`")
                    st.caption(ai.get("risk_reason", ""))

                    if ai.get("possible_conditions"):
                        st.markdown("**Possible Conditions:**")
                        for c in ai["possible_conditions"]:
                            st.markdown(f"- {c}")

                    if ai.get("lifestyle_recommendations"):
                        st.markdown("**Lifestyle Tips:**")
                        for tip in ai["lifestyle_recommendations"]:
                            st.markdown(f"- {tip}")

                    if ai.get("follow_up_tests"):
                        st.markdown("**Recommended Tests:**")
                        for test in ai["follow_up_tests"]:
                            st.markdown(f"- {test}")

                    st.info(ai.get("plain_summary", ""))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ask MediScan
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("💬 Ask a Medical Question")
    st.caption("Ask anything about symptoms, medications, diseases, or health tips.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("E.g. What does high creatinine mean?")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_medical_question(question)
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 MediScan AI")
    st.markdown("**Version:** 1.0.0")
    st.markdown("**Model:** LLaMA 3.3 70B via Groq")
    st.markdown("**NER:** spaCy en_core_web_sm")
    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown("- 🐍 Python")
    st.markdown("- 🤗 Groq API (LLaMA 3)")
    st.markdown("- 🔤 spaCy NLP")
    st.markdown("- ⚡ FastAPI")
    st.markdown("- 📊 Streamlit")
    st.markdown("- 🔬 scikit-learn")
    st.markdown("---")
    st.warning("⚠️ For educational use only.")