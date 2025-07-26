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
    import boto3 # The industry-standard library for S3-compatible storage like R2
    from botocore.exceptions import ClientError
    LIBS_LOADED = True
except ImportError as e:
    print(f"!!!!!!!!!!!!!!\nCRITICAL LIBRARY IMPORT ERROR: {e}\n!!!!!!!!!!!!!!")
    LIBS_LOADED = False

def download_model_files_at_runtime():
    """Downloads model files from Hugging Face."""
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
s3_client = None
R2_BUCKET_NAME = "semantle-daily-words" # The name you gave your bucket in Step 1

# --- APP STARTUP SEQUENCE ---
print("--- Starting Application ---")
if LIBS_LOADED:
    # Set up Cloudflare R2 connection
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key_id = os.getenv("R2_ACCESS_KEY_ID")
    secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
    
    if all([account_id, access_key_id, secret_access_key]):
        try:
            s3_client = boto3.client(
                's3',
                endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            )
            print("✅ Cloudflare R2 client initialized.")
        except Exception as e:
            print(f"❌ Failed to initialize R2 client: {e}")
    else:
        print("⏩ R2 integration is disabled (environment variables not set).")

    files_ready = download_model_files_at_runtime()
    if files_ready:
        try:
            temp_model = Word2Vec.load("model.mdl")
            model = temp_model.wv
            del temp_model
            print("✅✅✅ Word2Vec Model Loaded Successfully (Memory Efficient)! ✅✅✅")
        except Exception as e:
            print(f"❌ Failed to load Word2Vec model. Error: {e}")

app = FastAPI()

# --- HELPER FUNCTIONS ---
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "default_password")
if LIBS_LOADED: security = HTTPBasic()

def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    # ... (This function is correct)
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
        
        # New, reliable way to save the file to R2
        if s3_client:
            filename = date_to_filename(date_str)
            file_content = json.dumps(daily_data, ensure_ascii=False).encode('utf-8')
            try:
                s3_client.put_object(Bucket=R2_BUCKET_NAME, Key=filename, Body=file_content, ContentType='application/json')
                print(f"✅ Successfully uploaded {filename} to Cloudflare R2.")
            except ClientError as e:
                print(f"❌ Error uploading to R2: {e}")
        
        daily_words_cache[date_str] = daily_data
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in background task: {e}")
    finally:
        if date_str in background_tasks: del background_tasks[date_str]

def load_daily_words(date_str: str) -> Optional[Dict]:
    if date_str in daily_words_cache: return daily_words_cache[date_str]
    
    if s3_client:
        filename = date_to_filename(date_str)
        try:
            response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=filename)
            file_content = response['Body'].read()
            data = json.loads(file_content)
            daily_words_cache[date_str] = data
            print(f"✅ Successfully downloaded and cached {filename} from R2.")
            return data
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"Info: File {filename} not found in R2 bucket.")
            else:
                print(f"❌ Error downloading from R2: {e}")
    return None

def get_word_rank(daily_data: Dict, guess_word: str) -> Tuple[int, float]:
    # ... (This function remains unchanged)
    pass 

# ========== API ENDPOINTS ==========
@app.head("/")
def health_check_head(): return {}

@app.get("/")
def health_check():
    return {"status": "healthy" if model is not None else "unhealthy", "model_loaded": model is not None}

# ... (All your other endpoints like /rank and /admin/set-daily-word are fine)
@app.post("/admin/set-daily-word")
def set_daily_word(date: str, word: str, credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    # ... (This function is fine)
    pass
