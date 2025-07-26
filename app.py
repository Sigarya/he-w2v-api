# app.py

import os
import json
import traceback
import threading
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

try:
    from gensim.models import Word2Vec, KeyedVectors
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    import secrets
    from supabase import create_client, Client # Import the Supabase library
    LIBS_LOADED = True
except ImportError as e:
    print(f"!!!!!!!!!!!!!!\nCRITICAL LIBRARY IMPORT ERROR: {e}\n!!!!!!!!!!!!!!")
    LIBS_LOADED = False

def download_model_files_at_runtime():
    """Checks for model files and downloads them from Hugging Face if missing."""
    model_file_urls = {
        "model.mdl": "https://huggingface.co/Sigarya/he-w2v-1000/resolve/main/model.mdl?download=true",
        "model.mdl.wv.vectors.npy": "https://huggingface.co/Sigarya/he-w2v-1000/resolve/main/model.mdl.wv.vectors.npy?download=true",
        "model.mdl.syn1neg.npy": "https://huggingface.co/Sigarya/he-w2v-1000/resolve/main/model.mdl.syn1neg.npy?download=true"
    }

    if all(os.path.exists(f) and os.path.getsize(f) > 10000 for f in model_file_urls.keys()):
        return True
    print("--- Model files not found. Starting download from Hugging Face. ---")
    for filename, url in model_file_urls.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                subprocess.run(["wget", "-O", filename, url], check=True, timeout=900)
                print(f"✅ Successfully downloaded {filename}.")
            except Exception as e:
                print(f"❌ ERROR: wget download failed for {filename}: {e}")
                return False
    return True

# --- GLOBAL VARIABLES ---
model: Optional[KeyedVectors] = None
daily_words_cache: Dict[str, Dict] = {}
background_tasks: Dict[str, threading.Thread] = {}
supabase: Optional[Client] = None

# --- APP STARTUP SEQUENCE ---
print("--- Starting Application ---")
if LIBS_LOADED:
    # Set up Supabase connection
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if url and key:
        supabase = create_client(url, key)
        print("✅ Supabase client initialized.")
    else:
        print("⏩ Supabase integration is disabled (URL or Key not set).")

    files_ready = download_model_files_at_runtime()
    if files_ready:
        try:
            print("Attempting to load Word2Vec model into memory...")
            temp_model = Word2Vec.load("model.mdl")
            model = temp_model.wv
            del temp_model
            print("✅✅✅ Word2Vec Model Loaded Successfully (Memory Efficient)! ✅✅✅")
        except Exception as e:
            print(f"❌ Failed to load Word2Vec model. Error: {e}")
            model = None
    else:
        print("--- Model not loaded due to file download errors. ---")

app = FastAPI()

# --- HELPER FUNCTIONS ---
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "default_password")
if LIBS_LOADED:
    security = HTTPBasic()

def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    # ... (This function remains unchanged)
    is_correct_username = secrets.compare_digest(credentials.username, "admin")
    is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return credentials

def get_today_date() -> str: return datetime.now(timezone.utc).strftime("%d/%m/%Y")
def parse_date(date_str: str) -> datetime: return datetime.strptime(date_str, "%d/%m/%Y")
def date_to_filename(date_str: str) -> str: return parse_date(date_str).strftime("%Y-%m-%d.json")

def calculate_daily_words_background(date_str: str, daily_word: str):
    print(f"🚀 BACKGROUND TASK STARTED for {date_str}")
    try:
        if model is None: raise Exception("Model is not loaded.")
        if daily_word not in model: raise Exception(f"Daily word '{daily_word}' not in vocabulary.")
        
        similar_words_tuples = model.most_similar(daily_word, topn=999)
        ranked_words = [{"word": word, "similarity": float(similarity), "rank": 999 - i} for i, (word, similarity) in enumerate(similar_words_tuples)]
        daily_data = {"date": date_str, "daily_word": daily_word, "similar_words": ranked_words, "ranking_system": "semantle_1_to_1000", "created_at": datetime.now(timezone.utc).isoformat()}
        
        # This is the new, reliable way to save the file
        if supabase:
            filename = date_to_filename(date_str)
            file_content = json.dumps(daily_data, ensure_ascii=False).encode('utf-8')
            try:
                # Upsert=True means it will overwrite the file if it already exists.
                supabase.storage.from_("daily-words").upload(filename, file_content, {"content-type": "application/json", "upsert": "true"})
                print(f"✅ Successfully uploaded {filename} to Supabase Storage.")
            except Exception as e:
                print(f"❌ Error uploading to Supabase Storage: {e}")
        
        daily_words_cache[date_str] = daily_data
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in background task: {e}")
        traceback.print_exc()
    finally:
        if date_str in background_tasks: del background_tasks[date_str]

def load_daily_words(date_str: str) -> Optional[Dict]:
    if date_str in daily_words_cache:
        return daily_words_cache[date_str]
    
    if supabase:
        filename = date_to_filename(date_str)
        try:
            print(f"Attempting to download {filename} from Supabase...")
            response = supabase.storage.from_("daily-words").download(filename)
            data = json.loads(response)
            daily_words_cache[date_str] = data
            print(f"✅ Successfully downloaded and cached {filename} from Supabase.")
            return data
        except Exception as e:
            # It's normal for this to fail if the file doesn't exist.
            print(f"Info: Could not find {filename} in Supabase Storage. {e}")
            
    return None

def get_word_rank(daily_data: Dict, guess_word: str) -> Tuple[int, float]:
    # ... (This function remains unchanged)
    if guess_word == daily_data["daily_word"]: return (1000, 1.0)
    for item in daily_data.get("similar_words", []):
        if item.get("word") == guess_word:
            return (item.get("rank", 0), item.get("similarity", 0.0))
    return (0, 0.0)

# ========== API ENDPOINTS ==========
@app.head("/")
def health_check_head(): return {}

@app.get("/")
def health_check():
    return {"status": "healthy" if model is not None else "unhealthy", "model_loaded": model is not None}

# ... (All your other endpoints like /rank and /admin/set-daily-word remain unchanged)
@app.post("/admin/set-daily-word")
def set_daily_word(date: str, word: str, credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    if not model: return JSONResponse(status_code=503, content={"error": "Model not loaded."})
    # ... (rest of the endpoint is the same)
    try: parse_date(date)
    except ValueError: return JSONResponse(status_code=400, content={"error": "Invalid date format."})
    if word not in model: return JSONResponse(status_code=400, content={"error": f"Word '{word}' not in vocabulary."})
    if date in background_tasks and background_tasks[date].is_alive(): return JSONResponse(status_code=409, content={"error": "Calculation running."})
    
    thread = threading.Thread(target=calculate_daily_words_background, args=(date, word), daemon=True)
    background_tasks[date] = thread
    thread.start()
    return {"message": f"Started calculation for {date}."}
