import os
from dotenv import load_dotenv
import requests


# Load the environment variable
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def fact_check_with_llm(news_text):
    """
    Sends the news text to Gemini API for fact-checking style analysis.
    Returns the model's justification or label suggestion.
    """

    prompt = f"""You are an AI assistant helping detect misinformation in news articles. 
Here is a news claim:

\"{news_text}\"

Is this information factually accurate or fake? 
Respond with one of these labels: Real or Fake.
Then explain why in 2-3 lines."""

    headers = {
        "Content-Type": "application/json"
    }
    params = {
        "key": GEMINI_API_KEY
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        # Extract the response text from Gemini's structure
        message = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        return message
    except Exception as e:
        return f"Error during LLM check: {str(e)}"
