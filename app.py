# app.py

import os
import json
import traceback
import threading
import base64
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from gensim.models import Word2Vec
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import io

# Google Drive imports
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
    from google.oauth2 import service_account
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    print("Google Drive libraries not available. Using local storage only.")
    GOOGLE_DRIVE_AVAILABLE = False

app = FastAPI()
security = HTTPBasic()

# --- CONFIGURATION (Loaded securely from Environment Variables) ---
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "default_password")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

# --- GLOBAL VARIABLES ---
model: Optional[Word2Vec] = None
daily_words_cache: Dict[str, Dict] = {}
background_tasks: Dict[str, threading.Thread] = {}
drive_service = None


def setup_google_drive():
    """Initialize Google Drive service using credentials from environment variables."""
    global drive_service
    if not GOOGLE_DRIVE_AVAILABLE or not GOOGLE_SERVICE_ACCOUNT_JSON:
        print("⏩ Google Drive integration is disabled.")
        return False
    try:
        service_account_info = json.loads(base64.b64decode(GOOGLE_SERVICE_ACCOUNT_JSON).decode('utf-8'))
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        drive_service = build('drive', 'v3', credentials=credentials)
        print("✅ Google Drive service initialized successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Google Drive service: {e}")
        return False


def load_model():
    """Load the Word2Vec model. Assumes files were downloaded during the build step."""
    global model
    if model is not None:
        return model
    
    model_path = "model.mdl"
    print(f"Attempting to load model from: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ CRITICAL: Model file not found. It should have been downloaded during the build step.")
        print("Please check the build logs for download errors.")
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
setup_google_drive()
model = load_model()
if not os.path.exists('data'):
    os.makedirs('data')

# --- HELPER FUNCTIONS ---

def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)):
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
    return datetime.now(timezone.utc).strftime("%d/%m/%Y")

def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%d/%m/%Y")

def date_to_filename(date_str: str) -> str:
    return parse_date(date_str).strftime("%Y-%m-%d.json")

def upload_to_google_drive(filename: str, content: str) -> bool:
    try:
        with open(os.path.join('data', filename), 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Saved {filename} to local backup.")
    except Exception as e:
        print(f"❌ Error saving file locally: {e}")

    if not drive_service:
        return True

    try:
        file_metadata = {'name': filename, 'parents': [GOOGLE_DRIVE_FOLDER_ID]}
        media = MediaIoBaseUpload(io.BytesIO(content.encode('utf-8')), mimetype='application/json')
        query = f"name='{filename}' and parents in '{GOOGLE_DRIVE_FOLDER_ID}' and trashed=false"
        results = drive_service.files().list(q=query, fields="files(id)").execute()
        existing_files = results.get('files', [])

        if existing_files:
            drive_service.files().update(fileId=existing_files[0]['id'], media_body=media).execute()
            print(f"✅ Updated {filename} in Google Drive.")
        else:
            drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"✅ Uploaded new {filename} to Google Drive.")
        return True
    except Exception as e:
        print(f"❌ Error uploading to Google Drive: {e}")
        return False

def download_from_google_drive(filename: str) -> Optional[Dict]:
    if drive_service:
        try:
            query = f"name='{filename}' and parents in '{GOOGLE_DRIVE_FOLDER_ID}' and trashed=false"
            results = drive_service.files().list(q=query, fields="files(id)").execute()
            if files := results.get('files', []):
                request = drive_service.files().get_media(fileId=files[0]['id'])
                file_content = io.BytesIO()
                downloader = MediaIoBaseDownload(file_content, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                print(f"✅ Downloaded {filename} from Google Drive.")
                return json.loads(file_content.getvalue().decode('utf-8'))
        except Exception as e:
            print(f"⚠️ Could not download from Google Drive, trying local. Error: {e}")

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
    print(f"🚀 BACKGROUND TASK STARTED for {date_str} with word '{daily_word}'")
    try:
        if model is None: raise Exception("Model is not loaded.")
        if daily_word not in model.wv: raise Exception(f"Daily word '{daily_word}' not found in vocabulary.")

        print("Calculating 999 most similar words...")
        similar_words_tuples = model.wv.most_similar(daily_word, topn=999)
        
        ranked_words = [{"word": word, "similarity": float(similarity), "rank": 999 - i} for i, (word, similarity) in enumerate(similar_words_tuples)]

        daily_data = {"date": date_str, "daily_word": daily_word, "similar_words": ranked_words, "created_at": datetime.now(timezone.utc).isoformat(), "ranking_system": "semantle_1_to_1000", "storage_version": "2.0"}
        
        daily_words_cache[date_str] = daily_data
        print(f"Saved {date_str} to in-memory cache.")
        
        upload_to_google_drive(date_to_filename(date_str), json.dumps(daily_data, indent=2, ensure_ascii=False))
        
        print(f"✅ BACKGROUND TASK COMPLETED for {date_str}.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR in background task for {date_str}: {e}")
        traceback.print_exc()
        daily_words_cache[f"{date_str}_error"] = {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
    finally:
        if date_str in background_tasks: del background_tasks[date_str]

def load_daily_words(date_str: str) -> Optional[Dict]:
    if date_str in daily_words_cache: return daily_words_cache[date_str]
    if data := download_from_google_drive(date_to_filename(date_str)):
        daily_words_cache[date_str] = data
        return data
    return None

def get_word_rank(daily_data: Dict, guess_word: str) -> Tuple[int, float]:
    if guess_word == daily_data["daily_word"]: return (1000, 1.0)
    for item in daily_data["similar_words"]:
        if item["word"] == guess_word: return (item["rank"], item["similarity"])
    return (0, 0.0)

# ========== API ENDPOINTS ==========

@app.head("/")
def health_check_head():
    return {}

@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "google_drive_enabled": drive_service is not None}

@app.get("/rank")
def get_word_rank_endpoint(word: str, date: Optional[str] = None):
    target_date = date if date else get_today_date()
    if not model: return JSONResponse(status_code=503, content={"error": "Model is not loaded."})
    if word not in model.wv: return {"rank": 0, "similarity": 0.0, "error": "Word not in vocabulary"}
    if not (daily_data := load_daily_words(target_date)): return JSONResponse(status_code=404, content={"error": f"No data for {target_date}"})
    
    rank, similarity = get_word_rank(daily_data, word)
    return {"date": target_date, "word": word, "daily_word": daily_data["daily_word"], "rank": rank, "similarity": similarity, "is_daily_word": word == daily_data["daily_word"]}

# ========== ADMIN ENDPOINTS ==========

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
