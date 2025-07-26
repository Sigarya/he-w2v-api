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
    """
    Checks for model files and downloads them from Hugging Face if missing.
    This method is simple and reliable.
    """
    
    # === ACTION REQUIRED ===
    # Replace these placeholder URLs with your 3 real Hugging Face download links from Step 1.
    model_file_urls = {
        "model.mdl": "https://huggingface.co/Sigarya/he-w2v-1000/resolve/main/model.mdl?download=true",
        "model.mdl.wv.vectors.npy": "https://huggingface.co/Sigarya/he-w2v-1000/resolve/main/model.mdl.wv.vectors.npy?download=true",
        "model.mdl.syn1neg.npy": "https://huggingface.co/Sigarya/he-w2v-1000/resolve/main/model.mdl.syn1neg.npy?download=true"
    }
    # =======================

    # Check if all files already exist and are reasonably large
    if all(os.path.exists(f) and os.path.getsize(f) > 10000 for f in model_file_urls.keys()):
        print("--- Model files already exist. Skipping download. ---")
        return True

    print("--- Model files not found. Starting download from Hugging Face. ---")
    
    for filename, url in model_file_urls.items():
        if "YourUsername" in url:
            print(f"❌ FATAL ERROR: Placeholder URL found for {filename}. Please paste your real Hugging Face download links into app.py.")
            return False
            
        if not os.path.exists(filename) or os.path.getsize(filename) < 10000:
            print(f"Downloading {filename}...")
            try:
                # Using the simple, reliable wget command.
                # This works perfectly with Hugging Face's direct links.
                subprocess.run(["wget", "-O", filename, url], check=True, timeout=900)
                print(f"✅ Successfully downloaded {filename}.")
            except Exception as e:
                print(f"❌ ERROR: wget download failed for {filename}: {e}")
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
model = None
if LIBS_LOADED:
    files_ready = download_model_files_at_runtime()
    if files_ready:
        try:
            print("Attempting to load Word2Vec model...")
            model = Word2Vec.load("model.mdl")
            print("✅✅✅ Word2Vec Model Loaded Successfully! The app is fully functional. ✅✅✅")
        except Exception as e:
            print(f"❌ Failed to load Word2Vec model after download. Error: {e}")
            model = None
    else:
        print("--- Model not loaded due to file download errors. ---")

app = FastAPI()

# --- HELPER & SETUP FUNCTIONS ---
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "default_password")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
if LIBS_LOADED:
    security = HTTPBasic()

def setup_google_drive():
    global drive_service
    if not LIBS_LOADED: return
    GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        print("⏩ Google Drive integration for saving daily words is disabled (no service account JSON).")
        return
    try:
        service_account_info = json.loads(base64.b64decode(GOOGLE_SERVICE_ACCOUNT_JSON).decode('utf-8'))
        credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=['https://www.googleapis.com/auth/drive'])
        drive_service = build('drive', 'v3', credentials=credentials)
        print("✅ Google Drive service initialized for saving daily words.")
    except Exception as e:
        print(f"❌ Failed to initialize Google Drive service: {e}")

setup_google_drive()
if not os.path.exists('data'): os.makedirs('data')

def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, "admin")
    is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return credentials

def get_today_date() -> str:
    return datetime.now(timezone.utc).strftime("%d/%m/%Y")

def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%d/%m/%Y")

def date_to_filename(date_str: str) -> str:
    return parse_date(date_str).strftime("%Y-%m-%d.json")

def get_word_rank(daily_data: Dict, guess_word: str) -> Tuple[int, float]:
    if guess_word == daily_data["daily_word"]: return (1000, 1.0)
    for item in daily_data.get("similar_words", []):
        if item["word"] == guess_word: return (item.get("rank", 0), item.get("similarity", 0.0))
    return (0, 0.0)

# The rest of your original functions are here, complete.
def upload_to_google_drive(filename: str, content: str) -> bool:
    try:
        with open(os.path.join('data', filename), 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Saved {filename} to local backup.")
    except Exception as e:
        print(f"❌ Error saving file locally: {e}")

    if not drive_service: return True

    try:
        file_metadata = {'name': filename, 'parents': [GOOGLE_DRIVE_FOLDER_ID]}
        media = MediaIoBaseUpload(io.BytesIO(content.encode('utf-8')), mimetype='application/json')
        query = f"name='{filename}' and parents in '{GOOGLE_DRIVE_FOLDER_ID}' and trashed=false"
        results = drive_service.files().list(q=query, fields="files(id)").execute()
        if existing_files := results.get('files', []):
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
                with io.BytesIO() as file_content:
                    downloader = MediaIoBaseDownload(file_content, request)
                    done = False
                    while not done: _, done = downloader.next_chunk()
                    return json.loads(file_content.getvalue().decode('utf-8'))
        except Exception as e:
            print(f"⚠️ Could not download from Google Drive, trying local. Error: {e}")

    try:
        local_path = os.path.join('data', filename)
        if os.path.exists(local_path):
            with open(local_path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e:
        print(f"❌ Error loading from local storage: {e}")
    return None

def calculate_daily_words_background(date_str: str, daily_word: str):
    print(f"🚀 BACKGROUND TASK STARTED for {date_str} with word '{daily_word}'")
    try:
        if model is None: raise Exception("Model is not loaded.")
        if daily_word not in model.wv: raise Exception(f"Daily word '{daily_word}' not found in vocabulary.")
        similar_words_tuples = model.wv.most_similar(daily_word, topn=999)
        ranked_words = [{"word": word, "similarity": float(similarity), "rank": 999 - i} for i, (word, similarity) in enumerate(similar_words_tuples)]
        daily_data = {"date": date_str, "daily_word": daily_word, "similar_words": ranked_words, "created_at": datetime.now(timezone.utc).isoformat(), "ranking_system": "semantle_1_to_1000"}
        daily_words_cache[date_str] = daily_data
        upload_to_google_drive(date_to_filename(date_str), json.dumps(daily_data, indent=2, ensure_ascii=False))
        print(f"✅ BACKGROUND TASK COMPLETED for {date_str}.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR in background task for {date_str}: {e}")
    finally:
        if date_str in background_tasks: del background_tasks[date_str]

def load_daily_words(date_str: str) -> Optional[Dict]:
    if date_str in daily_words_cache: return daily_words_cache[date_str]
    if data := download_from_google_drive(date_to_filename(date_str)):
        daily_words_cache[date_str] = data
        return data
    return None

# ========== API ENDPOINTS ==========

@app.head("/")
def health_check_head():
    return {}

@app.get("/")
def health_check():
    """Basic health check endpoint."""
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False, "error": "Word2Vec model could not be loaded. Please check server logs."}

@app.get("/rank")
def get_word_rank_endpoint(word: str, date: Optional[str] = None):
    if not model: return JSONResponse(status_code=503, content={"error": "Model is not loaded."})
    target_date = date or get_today_date()
    if not (daily_data := load_daily_words(target_date)): return JSONResponse(status_code=404, content={"error": f"No data for {target_date}"})
    rank, similarity = get_word_rank(daily_data, word)
    return {"date": target_date, "word": word, "daily_word": daily_data["daily_word"], "rank": rank, "similarity": similarity, "is_daily_word": word == daily_data["daily_word"]}

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
