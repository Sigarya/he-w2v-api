# app.py

import os
import json
import traceback
import threading
import subprocess
import re
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import requests
from gensim.models import Word2Vec
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import io # Make sure io is imported

# Google Drive imports
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    print("Google Drive libraries not available. Using local storage only.")
    GOOGLE_DRIVE_AVAILABLE = False

app = FastAPI()
security = HTTPBasic()

# --- CONFIGURATION (Loaded from Environment Variables) ---
# This is much more secure! We'll set these in the Render dashboard.
# The second value (e.g., "default_password") is a fallback for local testing.
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "default_password") 
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

# --- GLOBAL VARIABLES ---
model: Optional[Word2Vec] = None
daily_words_cache: Dict[str, Dict] = {}  # Cache for loaded daily word data
background_tasks: Dict[str, threading.Thread] = {}  # Track background calculations
drive_service = None

def setup_google_drive():
    """Initialize Google Drive service"""
    global drive_service
    
    if not GOOGLE_DRIVE_AVAILABLE:
        print("Google Drive libraries not available")
        return False
    
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        print("No Google Service Account JSON provided via environment variable.")
        return False
    
    try:
        # The service account JSON is expected to be a base64 encoded string.
        # This is a common way to handle multi-line JSON secrets in environment variables.
        service_account_info = json.loads(base64.b64decode(GOOGLE_SERVICE_ACCOUNT_JSON).decode('utf-8'))
        
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        
        drive_service = build('drive', 'v3', credentials=credentials)
        print("✅ Google Drive service initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Google Drive service: {e}")
        traceback.print_exc()
        return False

def download_model_files():
    """Download model files if they don't exist"""
    model_files = {
        "model.mdl": "1T9tSdIm-8AEz0c6mJuFfLyBL75_lnsTU",
        "model.mdl.wv.vectors.npy": "1z5n9L-2oS_YEh3qf-nkz3ugMM8oqpxGZ", 
        "model.mdl.syn1neg.npy": "1uhu7bevYhCYZNLPdvupSuw_4X42_-smY"
    }
    
    # This entire download logic is complex but necessary for a free hosting service
    # where downloads can be slow or fail. Your approach is very robust!
    
    files_to_download = []
    for filename, file_id in model_files.items():
        if not os.path.exists(filename):
            files_to_download.append((filename, file_id))
        else:
            try:
                size = os.path.getsize(filename)
                if size < 1000: # Check for tiny, likely corrupted files
                    print(f"File {filename} is too small, will re-download.")
                    files_to_download.append((filename, file_id))
                else:
                    with open(filename, 'rb') as f:
                        if b'<html>' in f.read(100).lower(): # Check for HTML error pages
                            print(f"File {filename} seems to be an HTML error page, will re-download.")
                            files_to_download.append((filename, file_id))
            except Exception as e:
                print(f"Error checking file {filename}, will try to re-download. Error: {e}")
                files_to_download.append((filename, file_id))
    
    if not files_to_download:
        print("👍 All model files already exist and look good.")
        return True
    
    print(f"Downloading {len(files_to_download)} model files...")
    
    for filename, file_id in files_to_download:
        print(f"Downloading {filename}...")
        # Using gdown via subprocess is a great fallback for large files on services like Render.
        try:
            subprocess.run([
                "python", "-m", "gdown", 
                "--id", file_id,
                "-O", filename,
                "--quiet"
            ], check=True, timeout=300)
            print(f"Successfully downloaded {filename}.")
        except Exception as e:
            print(f"❌ Failed to download {filename} with gdown: {e}")
            return False
            
    return True

def load_model():
    """Load the Word2Vec model, downloading files if necessary."""
    global model
    if model is not None:
        return model
    
    model_path = "model.mdl"
    print(f"Attempting to load model from: {model_path}")
    
    if not os.path.exists(model_path):
        print("Model file not found. Attempting to download...")
        if not download_model_files():
            print("❌ Critical error: Failed to download model files. The app cannot function.")
            return None
    
    try:
        model = Word2Vec.load(model_path)
        print("✅ Word2Vec Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Critical error loading model: {str(e)}")
        traceback.print_exc()
        return None

# --- APP STARTUP ---
# These run once when the server starts.
setup_google_drive()
model = load_model()
os.makedirs('data', exist_ok=True) # Ensure local backup directory exists

# --- HELPER FUNCTIONS ---

def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verifies admin credentials against the environment variable."""
    # `secrets.compare_digest` is used to prevent timing attacks. It's a secure way to compare strings.
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
    """Gets today's date in dd/mm/yyyy format (UTC)."""
    return datetime.now(timezone.utc).strftime("%d/%m/%Y")

def parse_date(date_str: str) -> datetime:
    """Parses a dd/mm/yyyy string into a datetime object."""
    return datetime.strptime(date_str, "%d/%m/%Y")

def date_to_filename(date_str: str) -> str:
    """Converts a dd/mm/yyyy date to a YYYY-MM-DD.json filename."""
    dt = parse_date(date_str)
    return dt.strftime("%Y-%m-%d.json")

def upload_to_google_drive(filename: str, content: str) -> bool:
    """Uploads data to Google Drive and saves a local backup."""
    # Always save locally first as a reliable backup
    try:
        local_path = os.path.join('data', filename)
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Saved {filename} to local backup.")
    except Exception as e:
        print(f"❌ Error saving file locally: {e}")
    
    if not drive_service or not GOOGLE_DRIVE_FOLDER_ID:
        print("⏩ Skipping Google Drive upload (service not configured).")
        return True # Still a success if local save worked

    try:
        file_metadata = {'name': filename, 'parents': [GOOGLE_DRIVE_FOLDER_ID]}
        media = MediaIoBaseUpload(io.BytesIO(content.encode('utf-8')), mimetype='application/json')
        
        # Check if file exists to update it, otherwise create new
        query = f"name='{filename}' and parents in '{GOOGLE_DRIVE_FOLDER_ID}' and trashed=false"
        results = drive_service.files().list(q=query, fields="files(id)").execute()
        existing_files = results.get('files', [])

        if existing_files:
            file_id = existing_files[0]['id']
            drive_service.files().update(fileId=file_id, media_body=media).execute()
            print(f"✅ Updated {filename} in Google Drive.")
        else:
            drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"✅ Uploaded new {filename} to Google Drive.")
        return True
    except Exception as e:
        print(f"❌ Error uploading to Google Drive: {e}")
        return False

def download_from_google_drive(filename: str) -> Optional[Dict]:
    """Downloads data from Google Drive, with a fallback to local storage."""
    # Try Google Drive first
    if drive_service and GOOGLE_DRIVE_FOLDER_ID:
        try:
            query = f"name='{filename}' and parents in '{GOOGLE_DRIVE_FOLDER_ID}' and trashed=false"
            results = drive_service.files().list(q=query, fields="files(id)").execute()
            files = results.get('files', [])
            
            if files:
                file_id = files[0]['id']
                request = drive_service.files().get_media(fileId=file_id)
                file_content = io.BytesIO()
                downloader = MediaIoBaseDownload(file_content, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                
                print(f"✅ Downloaded {filename} from Google Drive.")
                return json.loads(file_content.getvalue().decode('utf-8'))
        except Exception as e:
            print(f"⚠️ Could not download from Google Drive, trying local. Error: {e}")

    # Fallback to local storage
    try:
        local_path = os.path.join('data', filename)
        if os.path.exists(local_path):
            with open(local_path, 'r', encoding='utf-8') as f:
                print(f"✅ Loaded {filename} from local backup.")
                return json.load(f)
    except Exception as e:
        print(f"❌ Error loading from local storage: {e}")
    
    return None

def calculate_daily_words_background(date_str: str, daily_word: str):
    """
    This is the star of the show! It runs in the background.
    1. Finds the 999 most similar words.
    2. Assigns them a rank from 999 (most similar) down to 1.
    3. Saves the final list to Google Drive and a local file.
    """
    print(f"🚀 BACKGROUND TASK STARTED for {date_str} with word '{daily_word}'")
    try:
        current_model = model
        if current_model is None:
            raise Exception("Model is not loaded. Cannot perform calculation.")
            
        if daily_word not in current_model.wv:
            raise Exception(f"Daily word '{daily_word}' not found in model's vocabulary.")
        
        # This is the heavy lifting part. We ask the model for the 999 closest words.
        # Your foresight to handle MemoryError here is excellent for a free server!
        try:
            print("Calculating 999 most similar words...")
            similar_words_tuples = current_model.wv.most_similar(daily_word, topn=999)
            print(f"✅ Found {len(similar_words_tuples)} similar words.")
        except MemoryError as me:
            print(f"⚠️ MemoryError during calculation: {me}. Trying with a smaller number (500).")
            similar_words_tuples = current_model.wv.most_similar(daily_word, topn=500)
            print(f"✅ Found {len(similar_words_tuples)} similar words with fallback.")

        # This is where we implement the Semantle ranking logic.
        # Rank 1000: The daily word itself.
        # Rank 999: The #1 most similar word.
        # Rank 998: The #2 most similar word.
        # ...and so on.
        ranked_words = []
        for i, (word, similarity) in enumerate(similar_words_tuples):
            semantle_rank = 999 - i
            ranked_words.append({
                "word": word,
                "similarity": float(similarity),
                "rank": semantle_rank
            })

        # This is the final data object we will save.
        daily_data = {
            "date": date_str,
            "daily_word": daily_word,
            "similar_words": ranked_words, # The list we just created
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ranking_system": "semantle_1_to_1000",
            "storage_version": "2.0" # Good practice to version your data format!
        }
        
        # Put the result in our in-memory cache for immediate access.
        daily_words_cache[date_str] = daily_data
        print(f"Saved {date_str} to in-memory cache.")
        
        # Save the result to Google Drive (with local backup).
        filename = date_to_filename(date_str)
        content = json.dumps(daily_data, indent=2, ensure_ascii=False)
        upload_to_google_drive(filename, content)
        
        print(f"✅ BACKGROUND TASK COMPLETED for {date_str}.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR in background task for {date_str}: {e}")
        traceback.print_exc()
        # Cache the error so we can see what went wrong via an API call.
        daily_words_cache[f"{date_str}_error"] = {
            "error": str(e),
            "error_type": str(type(e)),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    finally:
        # Clean up the task from our tracking dictionary.
        if date_str in background_tasks:
            del background_tasks[date_str]

def load_daily_words(date_str: str) -> Optional[Dict]:
    """Loads daily word data, from cache first, then from storage."""
    if date_str in daily_words_cache:
        return daily_words_cache[date_str]
    
    filename = date_to_filename(date_str)
    data = download_from_google_drive(filename)
    
    if data:
        daily_words_cache[date_str] = data # Store in cache for next time
        return data
    
    return None

def get_word_rank(daily_data: Dict, guess_word: str) -> Tuple[int, float]:
    """Finds the rank and similarity for a guessed word."""
    if guess_word == daily_data["daily_word"]:
        return (1000, 1.0) # Rank 1000, perfect similarity
    
    for item in daily_data["similar_words"]:
        if item["word"] == guess_word:
            return (item["rank"], item["similarity"])
    
    return (0, 0.0) # Not in the top 1000

# ========== ALL YOUR API ENDPOINTS GO HERE ==========
# (Your existing endpoints are great! I'll paste them back in without changes,
# except for the admin ones which are already included above)

# ========== PUBLIC ENDPOINTS ==========

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "message": "Hebrew Word2Vec API is running!",
        "model_loaded": model is not None,
        "google_drive_enabled": drive_service is not None
    }

@app.get("/rank")
def get_word_rank_endpoint(word: str, date: Optional[str] = None):
    """Get the semantle rank (1-1000) and similarity of a word for a specific date."""
    target_date = date if date else get_today_date()
    
    daily_data = load_daily_words(target_date)
    if not daily_data:
        return JSONResponse(
            status_code=404,
            content={"error": f"No daily word data found for {target_date}"}
        )
    
    # Let's make sure the word exists in the main model first
    if model and word not in model.wv:
        return {"rank": 0, "similarity": 0.0, "error": "Word not in vocabulary"}

    rank, similarity = get_word_rank(daily_data, word)
    
    return {
        "date": target_date,
        "word": word,
        "daily_word": daily_data["daily_word"],
        "rank": rank,
        "similarity": similarity,
        "is_daily_word": word == daily_data["daily_word"]
    }
    
# ... (Paste all your other endpoints like /similarity, /daily-word, etc. here)
# Your other endpoints from the original file are fine and can be pasted here directly.
# For brevity, I'll focus on the admin/setup endpoints which have been updated.

# ========== ADMIN ENDPOINTS ==========
@app.post("/admin/set-daily-word")
def set_daily_word(date: str, word: str, credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    """Set the daily word and trigger the background calculation (Admin only)."""
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model is not loaded."})
    
    try:
        parse_date(date)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid date format. Use dd/mm/yyyy"})
        
    if word not in model.wv:
        return JSONResponse(status_code=400, content={"error": f"Word '{word}' not found in vocabulary."})
        
    if date in background_tasks and background_tasks[date].is_alive():
        return JSONResponse(status_code=409, content={"error": f"Calculation already running for {date}."})

    # Start the background job!
    thread = threading.Thread(
        target=calculate_daily_words_background,
        args=(date, word),
        daemon=True # Daemon threads exit when the main app exits
    )
    background_tasks[date] = thread
    thread.start()
    
    return {
        "message": f"Successfully started background calculation for {date} with word '{word}'.",
        "status": "processing"
    }

# ... (And here you would paste the rest of your well-designed admin and debug endpoints)
# I'm omitting them here to keep the response focused, but you should add them back in!
# They are very useful for managing your application.
