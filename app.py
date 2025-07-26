# app.py

import os
import json
import traceback
import threading
import base64
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# We will try to import these, but the app should still start if they fail
# This is so we can see logs even if pip install has issues.
try:
    from gensim.models import Word2Vec
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    import secrets
    import io
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
    from google.oauth2 import service_account
    LIBS_LOADED = True
except ImportError as e:
    print(f"!!!!!!!!!!!!!!\nCRITICAL LIBRARY IMPORT ERROR: {e}\n!!!!!!!!!!!!!!")
    LIBS_LOADED = False

def download_model_files_at_runtime():
    """Checks for model files and downloads them using gdown if they are missing."""
    model_files = {
        "model.mdl": "1T9tSdIm-8AEz0c6mJuFfLyBL75_lnsTU",
        "model.mdl.wv.vectors.npy": "1z5n9L-2oS_YEh3qf-nkz3ugMM8oqpxGZ",
        "model.mdl.syn1neg.npy": "1uhu7bevYhCYZNLPdvupSuw_4X42_-smY"
    }

    if all(os.path.exists(f) for f in model_files.keys()):
        print("--- Model files already exist. Skipping download. ---")
        return True

    print("--- Model files not found. Starting download with gdown. ---")
    
    for filename, file_id in model_files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                # Use python -m gdown to be robust. check=True will raise an error if gdown fails.
                subprocess.run(
                    ["python", "-m", "gdown", "--id", file_id, "-O", filename, "--quiet"], 
                    check=True, 
                    timeout=600 # 10 minute timeout
                )
                print(f"✅ Successfully downloaded {filename}.")
            except Exception as e:
                print(f"❌ ERROR: gdown download failed for {filename}: {e}")
                return False
                
    print("--- All model files downloaded successfully. ---")
    return True

# --- GLOBAL VARIABLES ---
model: Optional[Word2Vec] = None
drive_service = None
daily_words_cache: Dict[str, Dict] = {}
background_tasks: Dict[str, threading.Thread] = {}

# --- APP STARTUP SEQUENCE ---
print("--- Starting Application ---")
files_ready = download_model_files_at_runtime()

if files_ready and LIBS_LOADED:
    try:
        print("Attempting to load Word2Vec model...")
        model = Word2Vec.load("model.mdl")
        print("✅✅✅ Word2Vec Model Loaded Successfully! ✅✅✅")
    except Exception as e:
        print(f"❌ Failed to load Word2Vec model after download. Error: {e}")
        model = None
else:
    print("--- Model not loaded due to file download or library import errors. ---")
    model = None

# Create the FastAPI app
app = FastAPI()

# --- HELPER & SETUP FUNCTIONS ---

def setup_google_drive():
    """Initialize Google Drive service using credentials from environment variables."""
    global drive_service
    if not LIBS_LOADED: return
    
    GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        print("⏩ Google Drive integration is disabled (no service account JSON).")
        return
        
    try:
        service_account_info = json.loads(base64.b64decode(GOOGLE_SERVICE_ACCOUNT_JSON).decode('utf-8'))
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        drive_service = build('drive', 'v3', credentials=credentials)
        print("✅ Google Drive service initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize Google Drive service: {e}")
        drive_service = None

# Run setup functions after app is created and variables are defined
setup_google_drive()
if not os.path.exists('data'):
    os.makedirs('data')

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "default_password")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
if LIBS_LOADED:
    security = HTTPBasic()

def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, "admin")
    is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})
    return credentials

def get_today_date() -> str:
    return datetime.now(timezone.utc).strftime("%d/%m/%Y")

def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%d/%m/%Y")

def date_to_filename(date_str: str) -> str:
    return parse_date(date_str).strftime("%Y-%m-%d.json")

def upload_to_google_drive(filename: str, content: str):
    # This function is from your original code, and it's good.
    pass # You would paste your original upload logic here

def download_from_google_drive(filename: str) -> Optional[Dict]:
    # This function is from your original code, and it's good.
    return None # You would paste your original download logic here

def calculate_daily_words_background(date_str: str, daily_word: str):
    # This function is from your original code, and it's good.
    pass # You would paste your original background calculation logic here

def load_daily_words(date_str: str) -> Optional[Dict]:
    if date_str in daily_words_cache: return daily_words_cache[date_str]
    if data := download_from_google_drive(date_to_filename(date_str)):
        daily_words_cache[date_str] = data
        return data
    return None

def get_word_rank(daily_data: Dict, guess_word: str) -> Tuple[int, float]:
    if guess_word == daily_data["daily_word"]: return (1000, 1.0)
    for item in daily_data.get("similar_words", []):
        if item["word"] == guess_word: return (item.get("rank", 0), item.get("similarity", 0.0))
    return (0, 0.0)

# ========== API ENDPOINTS ==========

@app.get("/")
def health_check():
    """Basic health check endpoint."""
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False, "error": "Word2Vec model could not be loaded. Check server logs."}

# Add all your other endpoints here, for example:
@app.get("/rank")
def get_word_rank_endpoint(word: str, date: Optional[str] = None):
    """Gets the rank of a word for a given date."""
    if not model:
        return JSONResponse(status_code=503, content={"error": "Model is not loaded."})
    
    target_date = date or get_today_date()
    
    daily_data = load_daily_words(target_date)
    if not daily_data:
        return JSONResponse(status_code=404, content={"error": f"No daily data for {target_date}"})
        
    rank, similarity = get_word_rank(daily_data, word)
    
    return {
        "date": target_date,
        "word": word,
        "daily_word": daily_data["daily_word"],
        "rank": rank,
        "similarity": similarity
    }

# And your admin endpoints
@app.post("/admin/set-daily-word")
def set_daily_word(date: str, word: str, credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    if not model: return JSONResponse(status_code=503, content={"error": "Model is not loaded."})
    try: parse_date(date)
    except ValueError: return JSONResponse(status_code=400, content={"error": "Invalid date format."})
    if word not in model.wv: return JSONResponse(status_code=400, content={"error": f"Word '{word}' not in vocabulary."})
    if date in background_tasks and background_tasks[date].is_alive(): return JSONResponse(status_code=409, content={"error": f"Calculation running for {date}."})

    thread = threading.Thread(target=calculate_daily_words_background, args=(date, word), daemon=True)
    background_tasks[date] = thread
    thread.start()
    
    return {"message": f"Started calculation for {date} with word '{word}'.", "status": "processing"}
