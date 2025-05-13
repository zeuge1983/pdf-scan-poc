import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() # This line executes immediately when the module is imported, loading variables from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Retrieves the API key

if not GOOGLE_API_KEY:
    print("API Key not found in environment variables!")
    exit()

print(f"Using API Key starting with: {GOOGLE_API_KEY[:5]}... and ending with: {GOOGLE_API_KEY[-5:]}")
genai.configure(api_key=GOOGLE_API_KEY)

try:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # Settings.llm = Gemini(api_key=GOOGLE_API_KEY, model_name="models/gemini-1.5-pro-latest")

    response = model.generate_content("What is the capital of France?")
    print("Response from Gemini:")
    print(response.text)
except Exception as e:
    print(f"Error connecting to Gemini: {e}")