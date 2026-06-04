from fastapi import FastAPI
from pydantic import BaseModel
from src.groq_analyzer import analyze_medical_report

app = FastAPI(title="MediScan AI API")

class ReportRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze(request: ReportRequest):
    result = analyze_medical_report(request.text)
    return {"analysis": result}