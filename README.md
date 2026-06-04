# 🏥 MediScan AI — Medical Report Analyzer & Disease Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![spaCy](https://img.shields.io/badge/spaCy-NLP-yellow)
![License](https://img.shields.io/badge/License-MIT-purple)

An end-to-end **Healthcare AI** web application that analyzes medical
reports, extracts clinical entities, predicts disease risks, and
interprets lab results — all powered by **LLaMA 3 via Groq API**
and **spaCy NLP**.

---

## 🚀 Live Demo

> Run locally using steps below ⬇️

---
## 📸 Features

### 📄 Tab 1 — Medical Report Analyzer
- Paste any doctor's note or clinical report
- AI extracts **symptoms, diseases, medications**
- Predicts **disease risk %** for each condition
- Gives **severity level** (Low / Medium / High / Critical)
- Provides **recommendations** and plain English summary

### 🧪 Tab 2 — Lab Results Analyzer
- Enter blood test values (glucose, BP, cholesterol, etc.)
- **Rule-based range checker** flags abnormal values
- **AI interprets** results and suggests possible conditions
- Generates **0-100 risk score**

### 💬 Tab 3 — Medical Chatbot
- Ask any health question
- Powered by LLaMA 3.3 70B via Groq
- Gives clear, simple answers

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| LLM / AI | Groq API (LLaMA 3.3 70B) |
| NLP / NER | spaCy (en_core_web_sm) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| ML Utilities | scikit-learn, NumPy, pandas |
| Environment | python-dotenv |
| Testing | pytest |

---

## 📁 Project Structure
mediscan-ai/
├── src/
│   ├── init.py
│   ├── groq_analyzer.py     # LLaMA 3 report analysis + Q&A
│   ├── ner_extractor.py     # spaCy NER pipeline
│   ├── lab_analyzer.py      # Lab value checker + AI interpretation
│   └── api.py               # FastAPI REST endpoints
├── tests/
│   └── test_pipeline.py     # pytest unit tests
├── data/
├── models/
├── app.py                   # Streamlit UI
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mediscan-ai.git
cd mediscan-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Add Groq API Key
```bash
# Create .env file
echo GROQ_API_KEY=your_key_here > .env
```
Get your free API key at: https://console.groq.com

---

## 🏃 Running the App

```bash
# Streamlit UI
streamlit run app.py

# FastAPI backend (optional, separate terminal)
cd src
uvicorn api:app --reload
# Docs at: http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/analyze` | Analyze medical report |
| POST | `/entities` | Extract NER entities |
| POST | `/lab` | Analyze lab values |
| POST | `/ask` | Ask medical question |
| POST | `/full-analysis` | Report + NER combined |

---

## 🧪 Running Tests

```bash
pytest tests/test_pipeline.py -v
```

Expected output:test_ner_extracts_symptoms        PASSED ✅
test_ner_extracts_diseases        PASSED ✅
test_ner_extracts_medications     PASSED ✅
test_normal_glucose               PASSED ✅
test_high_glucose_flagged         PASSED ✅
test_risk_score_zero_for_normal   PASSED ✅
---

## 📊 Sample Medical Report to Test
Patient: John Doe, Age 52, Male
Symptoms: chest pain, shortness of breath, fatigue
Vitals: BP 158/98 mmHg, Heart Rate 102 bpm, SpO2 93%
History: Type 2 Diabetes, Hypertension
Medications: Metformin 1000mg, Amlodipine 5mg, Aspirin 75mg
Lab: Glucose 218 mg/dL, HbA1c 8.9%, Cholesterol 242 mg/dL
Assessment: Poorly controlled diabetes, suspected coronary artery disease
---

## 🔑 Key Concepts Demonstrated

- **LLM API Integration** — Groq API with prompt engineering
- **NLP Pipeline** — spaCy Named Entity Recognition
- **REST API Design** — FastAPI with Pydantic validation
- **Frontend Development** — Streamlit interactive dashboard
- **Rule-based ML** — Lab value range classification
- **Software Engineering** — Modular code, unit tests, .env security

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**.
It is NOT a substitute for professional medical diagnosis or treatment.
Always consult a qualified healthcare provider for medical decisions.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@Vedika249](https://github.com/Vedika249)


---

## 📄 License

This project is licensed under the MIT License.
