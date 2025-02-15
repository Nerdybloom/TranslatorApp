
# Import required libraries
import sqlite3                                              # For SQLite database operations
from transformers import MarianMTModel, MarianTokenizer    # Hugging Face's translation model components
from fastapi import FastAPI, HTTPException                 # FastAPI framework and exception handling
from fastapi.middleware.cors import CORSMiddleware         # For handling Cross-Origin Resource Sharing
import torch                                               # PyTorch library for machine learning operations

# Initialize FastAPI application
app = FastAPI()                                           # Create a new FastAPI application instance

# Configure CORS to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,                                       # Add middleware for handling CORS
    allow_origins=["*"],                                  # Allow requests from all origins
    allow_credentials=True,                               # Allow credentials in requests
    allow_methods=["*"],                                  # Allow all HTTP methods
    allow_headers=["*"]                                   # Allow all HTTP headers
)

# Set up the translation model
model_name = "Helsinki-NLP/opus-mt-en-es"                # Specify the pre-trained model for English to Spanish translation
tokenizer = MarianTokenizer.from_pretrained(model_name)   # Initialize the tokenizer for processing text
model = MarianMTModel.from_pretrained(model_name)         # Load the pre-trained translation model

# Database initialization function
def init_db():
    conn = sqlite3.connect('translations.db')             # Create/connect to SQLite database
    c = conn.cursor()                                     # Create a cursor object to execute SQL commands
    c.execute('''CREATE TABLE IF NOT EXISTS translations  # Create table if it doesn't exist
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,   # Auto-incrementing primary key
                  original_text TEXT NOT NULL,            # Column for original English text
                  translated_text TEXT NOT NULL,          # Column for translated Spanish text
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')  # Timestamp for each translation
    conn.commit()                                         # Save changes to database
    conn.close()                                         # Close database connection

init_db()                                                # Initialize the database when starting the application

@app.get("/translate/")                                  # Define endpoint for translation requests
async def translate_text(text: str):
    try:
        conn = sqlite3.connect('translations.db')         # Connect to the database
        c = conn.cursor()                                # Create database cursor
        
        # Check if translation exists in database
        c.execute("SELECT translated_text FROM translations WHERE original_text=?", (text,))  # Query existing translations
        result = c.fetchone()                            # Get the first matching result
        
        if result:
            translated_text = result[0]                   # Use cached translation if available
        else:
            # Perform new translation using the model
            inputs = tokenizer(text, return_tensors="pt", padding=True)  # Tokenize input text
            translated = model.generate(**inputs)         # Generate translation
            translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]  # Decode translation
            
            # Store new translation in database
            c.execute("INSERT INTO translations (original_text, translated_text) VALUES (?, ?)",  # Save to database
                     (text, translated_text))
            conn.commit()                                # Save changes
        
        conn.close()                                     # Close database connection
        
        return {
            "original_text": text,                       # Return original text
            "translated_text": translated_text,          # Return translated text
            "source": "Helsinki-NLP/opus-mt-en-es model"  # Specify translation source
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Handle and return any errors

@app.get("/history/")                                    # Define endpoint for viewing translation history
async def get_translation_history():
    try:
        conn = sqlite3.connect('translations.db')         # Connect to database
        c = conn.cursor()                                # Create cursor
        c.execute("SELECT original_text, translated_text, timestamp FROM translations ORDER BY timestamp DESC LIMIT 100")  # Get recent translations
        history = [{"original": row[0], "translated": row[1], "timestamp": row[2]} for row in c.fetchall()]  # Format results
        conn.close()                                     # Close database connection
        return history                                   # Return translation history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Handle and return any errors

