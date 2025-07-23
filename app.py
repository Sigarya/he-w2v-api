import os
import json
import traceback
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import requests
from gensim.models import Word2Vec
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

app = FastAPI()
security = HTTPBasic()

# Configuration
ADMIN_PASSWORD = "flotzimbatusik"
GOOGLE_DRIVE_API_KEY = "AIzaSyD0qV86jw1KTYvVxeB-XHOLisaMyOAlmjI"
GOOGLE_DRIVE_FOLDER_ID = None  # You'll need to create a folder and get its ID

# Global variables
model: Optional[Word2Vec] = None
daily_words_cache: Dict[str, Dict] = {}  # Cache for loaded daily word data
background_tasks: Dict[str, threading.Thread] = {}  # Track background calculations

# Load the Word2Vec model
model_path = os.getenv("MODEL_PATH", "model.mdl")
print(f"Attempting to load model from: {model_path}")
try:
    model = Word2Vec.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    traceback.print_exc()
    model = None

def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials for protected endpoints"""
    is_correct_username = secrets.compare_digest(credentials.username, "admin")
    is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

def get_today_date() -> str:
    """Get today's date in dd/mm/yyyy format (UTC)"""
    return datetime.now(timezone.utc).strftime("%d/%m/%Y")

def parse_date(date_str: str) -> datetime:
    """Parse date string in dd/mm/yyyy format to datetime object"""
    return datetime.strptime(date_str, "%d/%m/%Y")

def date_to_filename(date_str: str) -> str:
    """Convert dd/mm/yyyy to filename format YYYY-MM-DD.json"""
    dt = parse_date(date_str)
    return dt.strftime("%Y-%m-%d.json")

def upload_to_google_drive(filename: str, content: str) -> bool:
    """Upload a file to Google Drive"""
    try:
        # Create file metadata
        metadata = {
            'name': filename,
            'parents': [GOOGLE_DRIVE_FOLDER_ID] if GOOGLE_DRIVE_FOLDER_ID else []
        }
        
        # Upload file using Google Drive API
        files = {
            'metadata': (None, json.dumps(metadata), 'application/json; charset=UTF-8'),
            'media': (filename, content, 'application/json')
        }
        
        response = requests.post(
            f"https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&key={GOOGLE_DRIVE_API_KEY}",
            files=files
        )
        
        if response.status_code == 200:
            print(f"Successfully uploaded {filename} to Google Drive")
            return True
        else:
            print(f"Failed to upload {filename}: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error uploading to Google Drive: {str(e)}")
        return False

def download_from_google_drive(filename: str) -> Optional[Dict]:
    """Download a file from Google Drive by filename"""
    try:
        # Search for the file
        search_url = f"https://www.googleapis.com/drive/v3/files?q=name='{filename}'&key={GOOGLE_DRIVE_API_KEY}"
        search_response = requests.get(search_url)
        
        if search_response.status_code != 200:
            print(f"Failed to search for {filename}: {search_response.text}")
            return None
            
        search_data = search_response.json()
        files = search_data.get('files', [])
        
        if not files:
            print(f"File {filename} not found on Google Drive")
            return None
            
        file_id = files[0]['id']
        
        # Download the file content
        download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={GOOGLE_DRIVE_API_KEY}"
        download_response = requests.get(download_url)
        
        if download_response.status_code == 200:
            return json.loads(download_response.text)
        else:
            print(f"Failed to download {filename}: {download_response.text}")
            return None
            
    except Exception as e:
        print(f"Error downloading from Google Drive: {str(e)}")
        return None

def calculate_daily_words_background(date_str: str, daily_word: str):
    """Background task to calculate 1000 most similar words"""
    print(f"Starting background calculation for {date_str} with word '{daily_word}'")
    
    try:
        if model is None:
            print("Model not loaded, cannot calculate similarity")
            return
            
        if daily_word not in model.wv:
            print(f"Daily word '{daily_word}' not in vocabulary")
            return
            
        # Calculate 1000 most similar words
        similar_words = model.wv.most_similar(daily_word, topn=1000)
        
        # Create the data structure
        daily_data = {
            "date": date_str,
            "daily_word": daily_word,
            "similar_words": [
                {
                    "word": word,
                    "similarity": float(similarity),
                    "rank": rank + 1  # rank 1-1000, where 1 is least similar
                }
                for rank, (word, similarity) in enumerate(reversed(similar_words))
            ],
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Cache in memory
        daily_words_cache[date_str] = daily_data
        
        # Save to Google Drive
        filename = date_to_filename(date_str)
        content = json.dumps(daily_data, indent=2, ensure_ascii=False)
        upload_to_google_drive(filename, content)
        
        print(f"Completed calculation for {date_str}")
        
    except Exception as e:
        print(f"Error in background calculation for {date_str}: {str(e)}")
        traceback.print_exc()
    finally:
        # Remove from background tasks tracking
        if date_str in background_tasks:
            del background_tasks[date_str]

def load_daily_words(date_str: str) -> Optional[Dict]:
    """Load daily words data, either from cache or Google Drive"""
    # Check cache first
    if date_str in daily_words_cache:
        return daily_words_cache[date_str]
    
    # Try to load from Google Drive
    filename = date_to_filename(date_str)
    data = download_from_google_drive(filename)
    
    if data:
        # Cache the loaded data
        daily_words_cache[date_str] = data
        return data
    
    return None

def get_word_rank(daily_data: Dict, word: str) -> int:
    """Get the rank of a word in the daily words list (0 if not found)"""
    if word == daily_data["daily_word"]:
        return 1000
    
    for similar_word_data in daily_data["similar_words"]:
        if similar_word_data["word"] == word:
            return 1000 - similar_word_data["rank"]  # Convert to our ranking system
    
    return 0

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Enhanced Word2Vec API is running!"}

@app.get("/similarity")
def get_similarity(word1: str, word2: str):
    """Get similarity between two words with ranking for today's daily word"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not loaded. Please check server logs."}
        )
    
    try:
        print(f"Calculating similarity for: '{word1}' and '{word2}'")
        
        # Check if words exist in vocabulary
        if word1 not in model.wv:
            return JSONResponse(
                status_code=400,
                content={"error": f"Word not found in vocabulary: {word1}"}
            )
        if word2 not in model.wv:
            return JSONResponse(
                status_code=400,
                content={"error": f"Word not found in vocabulary: {word2}"}
            )
        
        # Calculate similarity
        similarity_value = float(model.wv.similarity(word1, word2))
        
        # Get today's date and check if word2 is today's daily word
        today = get_today_date()
        daily_data = load_daily_words(today)
        
        rank = 0
        if daily_data and daily_data["daily_word"] == word2:
            rank = get_word_rank(daily_data, word1)
        
        return {
            "word1": word1,
            "word2": word2,
            "similarity": similarity_value,
            "rank": rank,
            "is_daily_word": daily_data is not None and daily_data["daily_word"] == word2
        }
        
    except Exception as e:
        print(f"Error in /similarity: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.get("/similarity/historical")
def get_historical_similarity(date: str, word1: str, word2: str):
    """Get similarity for a specific historical date"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not loaded. Please check server logs."}
        )
    
    try:
        # Validate date format
        try:
            parse_date(date)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid date format. Use dd/mm/yyyy"}
            )
        
        # Check if words exist in vocabulary
        if word1 not in model.wv:
            return JSONResponse(
                status_code=400,
                content={"error": f"Word not found in vocabulary: {word1}"}
            )
        if word2 not in model.wv:
            return JSONResponse(
                status_code=400,
                content={"error": f"Word not found in vocabulary: {word2}"}
            )
        
        # Load historical data
        daily_data = load_daily_words(date)
        if not daily_data:
            return JSONResponse(
                status_code=404,
                content={"error": f"No daily word data found for {date}"}
            )
        
        # Check if word2 matches the daily word for that date
        if daily_data["daily_word"] != word2:
            return JSONResponse(
                status_code=400,
                content={"error": f"Word2 '{word2}' is not the daily word for {date}. Daily word was '{daily_data['daily_word']}'"}
            )
        
        # Calculate similarity
        similarity_value = float(model.wv.similarity(word1, word2))
        rank = get_word_rank(daily_data, word1)
        
        return {
            "date": date,
            "word1": word1,
            "word2": word2,
            "similarity": similarity_value,
            "rank": rank,
            "daily_word": daily_data["daily_word"]
        }
        
    except Exception as e:
        print(f"Error in /similarity/historical: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.post("/admin/set-daily-word")
def set_daily_word(date: str, word: str, credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    """Set the daily word for a specific date (admin only)"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not loaded. Please check server logs."}
        )
    
    try:
        # Validate date format
        try:
            parse_date(date)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid date format. Use dd/mm/yyyy"}
            )
        
        # Check if word exists in vocabulary
        if word not in model.wv:
            return JSONResponse(
                status_code=400,
                content={"error": f"Word not found in vocabulary: {word}"}
            )
        
        # Check if there's already a background task for this date
        if date in background_tasks and background_tasks[date].is_alive():
            return JSONResponse(
                status_code=409,
                content={"error": f"Already calculating daily words for {date}. Please wait."}
            )
        
        # Start background calculation
        thread = threading.Thread(
            target=calculate_daily_words_background,
            args=(date, word),
            daemon=True
        )
        background_tasks[date] = thread
        thread.start()
        
        return {
            "message": f"Daily word for {date} set to '{word}'. Calculating 1000 most similar words in background.",
            "date": date,
            "daily_word": word,
            "status": "processing"
        }
        
    except Exception as e:
        print(f"Error in /admin/set-daily-word: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.get("/admin/status")
def get_admin_status(credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    """Get status of daily words and background tasks (admin only)"""
    return {
        "cached_dates": list(daily_words_cache.keys()),
        "background_tasks": {
            date: "running" if thread.is_alive() else "completed"
            for date, thread in background_tasks.items()
        },
        "today": get_today_date(),
        "model_loaded": model is not None
    }

@app.get("/daily-word")
def get_daily_word(date: Optional[str] = None):
    """Get the daily word for a specific date (defaults to today)"""
    target_date = date if date else get_today_date()
    
    try:
        if date:
            # Validate date format
            try:
                parse_date(date)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid date format. Use dd/mm/yyyy"}
                )
        
        daily_data = load_daily_words(target_date)
        if not daily_data:
            return JSONResponse(
                status_code=404,
                content={"error": f"No daily word set for {target_date}"}
            )
        
        return {
            "date": target_date,
            "daily_word": daily_data["daily_word"],
            "created_at": daily_data.get("created_at"),
            "total_similar_words": len(daily_data.get("similar_words", []))
        }
        
    except Exception as e:
        print(f"Error in /daily-word: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )
