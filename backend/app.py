import json
import os
from datetime import date

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from groq import Groq


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")


app = Flask(__name__)
client = Groq(api_key=api_key)

SYSTEM_PROMPT = (
    "You are an advanced AI assistant that analyzes spoken conversations and converts them into structured productivity outputs. "
    "Return valid JSON only. Do not wrap the response in markdown or code fences. "
    "Normalize all dates to ISO format using the provided current date for relative expressions. "
    "If a task has no explicit date but is important, suggest a reasonable reminder time. "
    "Keep transcript_cleaned readable, summary concise but detailed, and tasks, reminders, calendar events, deadlines, shopping items, and important notes separated clearly."
)


def analyze_transcript(transcript: str) -> dict:
    """Analyzes a transcript using Groq and returns structured JSON output."""
    today = date.today().isoformat()
    
    user_prompt = f"""Current date: {today}

Analyze this speech transcript and output JSON matching this schema:
{{
  "transcript_cleaned": "",
  "summary": "",
  "sentiment": "",
  "category": "",
  "tasks": [
    {{
      "title": "",
      "description": "",
      "date": "",
      "time": "",
      "priority": "",
      "recurring": false,
      "location": "",
      "participants": [],
      "reminder_before_minutes": 30
    }}
  ],
  "shopping_list": [],
  "deadlines": [],
  "calendar_events": [],
  "important_notes": []
}}

Rules:
- Understand informal and multilingual speech.
- Extract all date and time references, including relative ones such as tomorrow, next Monday, in 2 hours, or this evening.
- Detect recurring events when mentioned.
- Use priority values high, medium, or low.
- Keep all output valid JSON.
- Do not invent facts that are not supported by the transcript.

Transcript:
{transcript}
"""
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )
    
    raw_output = chat_completion.choices[0].message.content.strip()
    
    try:
        parsed_output = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {raw_output}") from exc
    
    return parsed_output


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "AI Transcript Analyzer API"})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyzes a speech transcript and returns structured JSON."""
    try:
        data = request.get_json()
        
        if not data or "transcript" not in data:
            return jsonify({"error": "Missing 'transcript' field in request body"}), 400
        
        transcript = data["transcript"].strip()
        
        if not transcript:
            return jsonify({"error": "Transcript cannot be empty"}), 400
        
        result = analyze_transcript(transcript)
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)