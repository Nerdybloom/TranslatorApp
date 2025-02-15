from fastapi import FastAPI
import requests
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)
API_URL = "https://api.mymemory.translated.net/get"

@app.get("/translate/")
def translate_text(text: str):
    """
    Translates English text to Spanish using MyMemory Translation API.
    """
    params = {"q": text, "langpair": "en|es"}  
    response = requests.get(API_URL, params=params)  
    data = response.json()  

    if "responseData" in data and "translatedText" in data["responseData"]:
        return {
            "original_text": text,
            "translated_text": data["responseData"]["translatedText"],
            "explanation": "This translation is provided by MyMemory API."
        }
    else:
        return {"error": "Translation failed"}
