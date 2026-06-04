import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def clean_json_response(raw: str) -> str:
    """Aggressively clean AI response to extract pure JSON."""
    raw = raw.strip()
    
    # Remove markdown code blocks
    raw = re.sub(r'```json', '', raw)
    raw = re.sub(r'```', '', raw)
    raw = raw.strip()
    
    # Find the first { and last } and extract only that
    start = raw.find('{')
    end = raw.rfind('}')
    
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    
    return raw.strip()


def analyze_medical_report(report_text: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert medical AI assistant.
Analyze the given medical report and return ONLY a raw JSON object.
DO NOT use markdown. DO NOT use code blocks. DO NOT add any explanation.
Your entire response must start with { and end with }.
Use exactly this structure:
{
  "extracted_symptoms": ["symptom1", "symptom2"],
  "possible_diseases": [{"name": "disease", "risk_percentage": 75, "reason": "reason"}],
  "medications_mentioned": ["med1", "med2"],
  "severity": "low",
  "severity_reason": "reason here",
  "recommendations": ["rec1", "rec2"],
  "summary": "Plain English summary here"
}"""
                },
                {
                    "role": "user",
                    "content": f"Analyze this medical report and return only JSON:\n\n{report_text}"
                }
            ],
            temperature=0.1,
            max_tokens=1500,
        )

        raw = response.choices[0].message.content
        cleaned = clean_json_response(raw)
        return json.loads(cleaned)

    except json.JSONDecodeError as e:
        return {"error": f"Json Parse Error: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}


def ask_medical_question(question: str, context: str = "") -> str:
    try:
        messages = [
            {
                "role": "system",
                "content": """You are a helpful medical AI assistant.
Answer medical questions clearly in simple language.
Always remind users to consult a real doctor for actual medical decisions."""
            }
        ]

        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
            messages.append({"role": "assistant", "content": "Understood, I have the context."})

        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4,
            max_tokens=800,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"