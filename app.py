# app.py

import os
import json
import traceback
import threading
import base64
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# We will try to import these, but the app should still start if they fail
try:
    import gdown
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
    # This will help debug if a library is missing from requirements.txt
    print(f"!!!!!!!!!!!!!!\nCRITICAL LIBRARY IMPORT ERROR: {e}\n!!!!!!!!!!!!!!")
    LIBS_LOADED = False

def download_model_files_at_runtime():
    """Checks for model files and downloads them using the gdown library directly."""
    if not LIBS_LOADED:
        print("Cannot download files because core libraries (like gdown) are missing.")
        return False

    model_files = {
        "model.mdl": "1T9tSdIm-8AEz0c6mJuFfLyBL75_lnsTU",
        "model.mdl.wv.vectors.npy": "1z5n9L-2oS_YEh3qf-nkz3ugMM8oqpxGZ",
        "model.mdl.syn1neg.npy": "1uhu7bevYhCYZNLPdvupSuw_4X42_-smY"
    }

    # Check if all files already exist and are reasonably large
    if all(os.path.exists(f) and os.path.getsize(f) > 10000 for f in model_files.keys()):
        print("--- Model files already exist. Skipping download. ---")
        return True

    print("--- Model files not found or incomplete. Starting download directly via gdown library. ---")
    
    for filename, file_id in model_files.items():
        if not os.path.exists(filename) or os.path.getsize(filename) < 10000:
            print(f"Downloading {filename}...")
            try:
                # This is the simple, reliable way: call the function directly.
                # quiet=False will show us the progress bar in the logs.
                gdown.download(id=file_id, output=filename, quiet=False)
                print(f"✅ Successfully downloaded {filename}.")
            except Exception as e:
                print(f"❌ ERROR: gdown.download function failed for {filename}: {e}")
                # Clean up a potentially corrupted, small file
                if os.path.exists(filename):
                    os.remove(filename)
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
# Only try to download and load if the core libraries were imported correctly
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
        model = None
else:
    model = None

# Create the FastAPI app. It will run regardless of model status.
app = FastAPI()

# --- HELPER & SETUP FUNCTIONS (The rest of your code, which is correct) ---
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "default_password")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
if LIBS_LOADED:
    security = HTTPBasic()

def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, "admin")
    is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return credentials

# ... (The rest of your functions like `setup_google_drive`, `/rank`, etc. are fine) ...
# I am pasting them here so you have a truly complete file.

def setup_google_drive():
    global drive_service
    if not LIBS_LOADED: return
    GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not GOOGLE_SERVICE_ACCOUNT_JSON: return
    try:
        service_account_info = json.loads(base64.b64decode(GOOGLE_SERVICE_ACCOUNT_JSON).decode('utf-8'))
        credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=['https://www.googleapis.com/auth/drive'])
        drive_service = build('drive', 'v3', credentials=credentials)
        print("✅ Google Drive service initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize Google Drive service: {e}")

setup_google_drive()
if not os.path.exists('data'): os.makedirs('data')

# ========== API ENDPOINTS ==========

@app.get("/")
def health_check():
    """Basic health check endpoint."""
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False, "error": "Word2Vec model could not be loaded. Please check server logs for download or load errors."}

# You can paste your other endpoints here from your working file.
# For example:
@app.get("/rank")
def get_rank():
    return {"message": "Add your ranking logic here."}
