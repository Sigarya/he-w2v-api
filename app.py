# app.py

import os
import json
import traceback
import threading
import base64
import subprocess # We need this to run wget
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

# This function will handle the downloads
def download_model_files_at_runtime():
    """Checks for model files and downloads them if they are missing."""
    model_files = {
        "model.mdl": "1T9tSdIm-8AEz0c6mJuFfLyBL75_lnsTU",
        "model.mdl.wv.vectors.npy": "1z5n9L-2oS_YEh3qf-nkz3ugMM8oqpxGZ",
        "model.mdl.syn1neg.npy": "1uhu7bevYhCYZNLPdvupSuw_4X42_-smY"
    }

    # Check if all files already exist
    if all(os.path.exists(f) for f in model_files.keys()):
        print("--- Model files already exist. Skipping download. ---")
        return True

    print("--- Model files not found. Starting download at runtime. ---")
    print("This may take several minutes.")

    for filename, file_id in model_files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                # This is your proven wget command, run from inside Python
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
                # We give it a long timeout of 10 minutes (600 seconds)
                subprocess.run(["wget", "-O", filename, url], check=True, timeout=600)
                # Verify that the downloaded file is not tiny (i.e. not an error page)
                if os.path.getsize(filename) < 10000:
                    print(f"❌ ERROR: Downloaded file {filename} is too small. It might be an error page.")
                    return False
                print(f"✅ Successfully downloaded {filename}.")
            except FileNotFoundError:
                print(f"❌ ERROR: 'wget' command not found. Cannot download files.")
                print("The server environment is missing a required tool.")
                return False
            except subprocess.CalledProcessError as e:
                print(f"❌ ERROR: Download failed for {filename}. wget returned an error: {e}")
                return False
            except subprocess.TimeoutExpired:
                print(f"❌ ERROR: Download timed out for {filename}.")
                return False
            except Exception as e:
                print(f"❌ ERROR: An unexpected error occurred downloading {filename}: {e}")
                return False
                
    print("--- All model files downloaded successfully. ---")
    return True

# --- GLOBAL VARIABLES ---
model: Optional[Word2Vec] = None
drive_service = None

# --- APP STARTUP SEQUENCE ---
print("--- Starting Application ---")
# 1. First, download the files if we need them.
files_ready = download_model_files_at_runtime()
# 2. If downloads were successful AND libraries loaded, we can try to load the model.
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

# The rest of your app code. It is designed to handle 'model' being None.
# This ensures the server can still run and show errors even if the model fails to load.
app = FastAPI()
if LIBS_LOADED:
    security = HTTPBasic()
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "default_password")
    GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

    @app.get("/")
    def health_check():
        if model is not None:
            return {"status": "healthy", "model_loaded": True}
        else:
            return {"status": "unhealthy", "model_loaded": False, "error": "Word2Vec model could not be loaded. Check server logs."}
    
    # ... All your other endpoints and functions go here, unchanged from my previous message.
    # For brevity, I'll just add the health check, but you should have the rest of your app logic here.

else:
    @app.get("/")
    def emergency_health_check():
        return {"status": "CRITICAL_ERROR", "error": "Core libraries like FastAPI or Gensim failed to import. Check build logs."}
