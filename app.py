import os
import json
import traceback
import threading
import subprocess
import re
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
GOOGLE_DRIVE_FOLDER_ID = None  # Will be disabled due to permission issues - using local storage instead

# Global variables
model: Optional[Word2Vec] = None
daily_words_cache: Dict[str, Dict] = {}  # Cache for loaded daily word data
background_tasks: Dict[str, threading.Thread] = {}  # Track background calculations

def download_model_files():
    """Download model files if they don't exist"""
    model_files = {
        "model.mdl": "1T9tSdIm-8AEz0c6mJuFfLyBL75_lnsTU",
        "model.mdl.wv.vectors.npy": "1z5n9L-2oS_YEh3qf-nkz3ugMM8oqpxGZ", 
        "model.mdl.syn1neg.npy": "1uhu7bevYhCYZNLPdvupSuw_4X42_-smY"
    }
    
    files_to_download = []
    for filename, file_id in model_files.items():
        if not os.path.exists(filename):
            files_to_download.append((filename, file_id))
        else:
            # Check if file is too small or contains HTML (corrupted download)
            try:
                size = os.path.getsize(filename)
                if size < 1000:
                    print(f"{filename} exists but is too small ({size} bytes), will re-download")
                    files_to_download.append((filename, file_id))
                else:
                    # Check if file starts with HTML (corrupted download)
                    with open(filename, 'rb') as f:
                        first_bytes = f.read(100)
                        if b'<html>' in first_bytes.lower() or b'<!doctype' in first_bytes.lower():
                            print(f"{filename} contains HTML content, will re-download")
                            files_to_download.append((filename, file_id))
            except:
                files_to_download.append((filename, file_id))
    
    if not files_to_download:
        print("All model files exist and appear valid")
        return True
    
    print(f"Need to download {len(files_to_download)} model files...")
    
    for filename, file_id in files_to_download:
        print(f"Downloading {filename}...")
        success = False
        
        # Method 1: Try gdown with different options
        try:
            # Remove existing file first
            if os.path.exists(filename):
                os.remove(filename)
            
            result = subprocess.run([
                "python", "-m", "gdown", 
                "--id", file_id,
                "-O", filename,
                "--quiet"
            ], capture_output=True, text=True, timeout=300)
            
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                # Check if it's not HTML
                with open(filename, 'rb') as f:
                    first_bytes = f.read(100)
                    if size > 1000 and b'<html>' not in first_bytes.lower():
                        print(f"Successfully downloaded {filename} with gdown (method 1) - {size} bytes")
                        success = True
                    else:
                        print(f"gdown method 1 returned HTML or small file for {filename}")
                        os.remove(filename)
        except Exception as e:
            print(f"gdown method 1 failed for {filename}: {e}")
        
        if success:
            continue
            
        # Method 2: Try gdown with fuzzy matching
        try:
            if os.path.exists(filename):
                os.remove(filename)
                
            result = subprocess.run([
                "python", "-m", "gdown", 
                "--fuzzy",
                f"https://drive.google.com/file/d/{file_id}/view?usp=sharing",
                "-O", filename,
                "--quiet"
            ], capture_output=True, text=True, timeout=300)
            
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                with open(filename, 'rb') as f:
                    first_bytes = f.read(100)
                    if size > 1000 and b'<html>' not in first_bytes.lower():
                        print(f"Successfully downloaded {filename} with gdown (method 2) - {size} bytes")
                        success = True
                    else:
                        print(f"gdown method 2 returned HTML or small file for {filename}")
                        os.remove(filename)
        except Exception as e:
            print(f"gdown method 2 failed for {filename}: {e}")
            
        if success:
            continue
        
        # Method 3: Try direct download with session to handle redirects
        try:
            if os.path.exists(filename):
                os.remove(filename)
                
            session = requests.Session()
            
            # First request to get the redirect
            url1 = f"https://drive.google.com/uc?export=download&id={file_id}"
            response1 = session.get(url1, stream=True, timeout=30)
            
            # Look for the download warning bypass
            if 'confirm=' in response1.text:
                # Extract the confirm token
                import re
                token_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', response1.text)
                if token_match:
                    token = token_match.group(1)
                    url2 = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
                    response2 = session.get(url2, stream=True, timeout=300)
                    
                    if response2.status_code == 200:
                        with open(filename, 'wb') as f:
                            for chunk in response2.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        size = os.path.getsize(filename)
                        with open(filename, 'rb') as f:
                            first_bytes = f.read(100)
                            if size > 1000 and b'<html>' not in first_bytes.lower():
                                print(f"Successfully downloaded {filename} with direct method - {size} bytes")
                                success = True
                            else:
                                print(f"Direct method returned HTML or small file for {filename}")
                                os.remove(filename)
            
        except Exception as e:
            print(f"Direct download failed for {filename}: {e}")
        
        if not success:
            print(f"All methods failed for {filename}")
            return False
    
    return True

def load_model():
    """Load the Word2Vec model with retry logic"""
    global model
    if model is not None:
        return model
    
    model_path = os.getenv("MODEL_PATH", "model.mdl")
    print(f"Attempting to load model from: {model_path}")
    
    # Check if model file exists, if not try to download
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        print("Current directory contents:")
        try:
            print(os.listdir("."))
        except Exception as e:
            print(f"Could not list directory: {e}")
        
        print("Attempting to download model files...")
        if download_model_files():
            print("Model files downloaded successfully")
        else:
            print("Failed to download model files")
            return None
    
    try:
        model = Word2Vec.load(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        traceback.print_exc()
        return None

# Try to load model at startup, but don't fail if it's not available yet
model = load_model()

# Create data directory for local storage if it doesn't exist
try:
    if not os.path.exists('data'):
        os.makedirs('data')
except:
    pass

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
    """Save file locally (Google Drive disabled due to API permissions)"""
    try:
        os.makedirs('data', exist_ok=True)
        local_path = os.path.join('data', filename)
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved {filename} locally to {local_path}")
        return True
    except Exception as e:
        print(f"Error saving locally: {str(e)}")
        return False

def download_from_google_drive(filename: str) -> Optional[Dict]:
    """Load file from local storage (Google Drive disabled due to API permissions)"""
    try:
        local_path = os.path.join('data', filename)
        if os.path.exists(local_path):
            with open(local_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Local file {local_path} not found")
            return None
    except Exception as e:
        print(f"Error loading from local storage: {str(e)}")
        return None

def calculate_daily_words_background(date_str: str, daily_word: str):
    """Background task to calculate 999 most similar words with improved error handling"""
    print(f"=== BACKGROUND TASK STARTED ===")
    print(f"Date: {date_str}, Word: {daily_word}")
    print(f"Thread ID: {threading.get_ident()}")
    
    try:
        # Ensure model is loaded
        current_model = model if model is not None else load_model()
        if current_model is None:
            print("ERROR: Model not loaded in background task, cannot calculate similarity")
            return
            
        print(f"Model confirmed loaded in background task")
        print(f"Model vocabulary size: {len(current_model.wv.key_to_index)}")
        
        if daily_word not in current_model.wv:
            print(f"ERROR: Daily word '{daily_word}' not in vocabulary in background task")
            return
            
        print(f"Daily word '{daily_word}' confirmed in vocabulary")
        print(f"Starting similarity calculation...")
        
        # Calculate top 999 most similar words (excluding the daily word itself)
        similar_words = None
        try:
            # First, let's try with a smaller number to test
            print("Testing with top 10 words first...")
            test_similar = current_model.wv.most_similar(daily_word, topn=10)
            print(f"Test successful - found {len(test_similar)} words")
            print(f"Sample results: {test_similar[:3]}")
            
            # Now try with 100
            print("Testing with top 100 words...")
            test_similar_100 = current_model.wv.most_similar(daily_word, topn=100)
            print(f"100 words test successful - found {len(test_similar_100)} words")
            
            # Now try the full 999 (we need 999 because the daily word itself gets rank 1000)
            print("Calculating 999 most similar words...")
            similar_words = current_model.wv.most_similar(daily_word, topn=999)
            print(f"Similarity calculation completed. Found {len(similar_words)} similar words")
            
        except MemoryError as me:
            print(f"MEMORY ERROR during similarity calculation: {str(me)}")
            print("Trying with smaller batch size...")
            # Fallback to smaller number
            try:
                similar_words = current_model.wv.most_similar(daily_word, topn=500)
                print(f"Fallback successful with 500 words: {len(similar_words)} similar words")
            except Exception as e2:
                print(f"Even fallback to 500 failed: {str(e2)}")
                try:
                    similar_words = current_model.wv.most_similar(daily_word, topn=250)
                    print(f"Fallback to 250 words successful: {len(similar_words)} similar words")
                except Exception as e3:
                    print(f"Even fallback to 250 failed: {str(e3)}")
                    return
                
        except Exception as calc_error:
            print(f"ERROR during similarity calculation: {str(calc_error)}")
            print(f"Error type: {type(calc_error)}")
            traceback.print_exc()
            return
        
        if similar_words is None:
            print("ERROR: similar_words is None after calculation")
            return
            
        print(f"Processing {len(similar_words)} similar words into data structure...")
        
        # Create the data structure with semantle-style ranking
        # Rank 1000 = daily word (target)
        # Rank 999 = most similar word
        # Rank 998 = second most similar word
        # ...
        # Rank 1 = least similar word (or words not in top list get rank 0)
        
        similar_words_data = []
        for i, (word, similarity) in enumerate(similar_words):
            semantle_rank = 999 - i  # 999, 998, 997, ..., down to (999 - len + 1)
            similar_words_data.append({
                "word": word,
                "similarity": float(similarity),
                "rank": semantle_rank
            })
            
            # Progress indicator
            if (i + 1) % 200 == 0:
                print(f"Processed {i + 1} words...")
        
        daily_data = {
            "date": date_str,
            "daily_word": daily_word,
            "similar_words": similar_words_data,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_words": len(similar_words_data),
            "calculation_method": "word2vec_most_similar_semantle_ranking",
            "model_vocab_size": len(current_model.wv.key_to_index),
            "ranking_system": "semantle_1_to_1000"
        }
        
        print(f"Data structure created with {len(daily_data['similar_words'])} words")
        
        # Cache in memory
        daily_words_cache[date_str] = daily_data
        print(f"Data cached in memory for {date_str}")
        
        # Save to local storage with better error handling
        filename = date_to_filename(date_str)
        try:
            content = json.dumps(daily_data, indent=2, ensure_ascii=False)
            print(f"JSON serialization successful, size: {len(content)} characters")
            
            success = upload_to_google_drive(filename, content)
            
            if success:
                print(f"Data saved to local storage as {filename}")
                # Verify the file was saved correctly
                local_path = os.path.join('data', filename)
                if os.path.exists(local_path):
                    file_size = os.path.getsize(local_path)
                    print(f"File verified: {local_path} ({file_size} bytes)")
                else:
                    print(f"WARNING: File not found after save: {local_path}")
            else:
                print(f"ERROR: Failed to save data to local storage")
                
        except Exception as save_error:
            print(f"ERROR during save process: {str(save_error)}")
            traceback.print_exc()
        
        print(f"=== BACKGROUND TASK COMPLETED SUCCESSFULLY ===")
        print(f"Final stats: {len(similar_words_data)} words processed for '{daily_word}' on {date_str}")
        
    except Exception as e:
        print(f"!!! CRITICAL ERROR in background calculation for {date_str}: {str(e)} !!!")
        print(f"Error type: {type(e)}")
        traceback.print_exc()
        
        # Try to save error info to cache so we know what happened
        try:
            error_data = {
                "date": date_str,
                "daily_word": daily_word,
                "error": str(e),
                "error_type": str(type(e)),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "thread_id": threading.get_ident()
            }
            daily_words_cache[f"{date_str}_error"] = error_data
            print(f"Error info cached for debugging")
        except:
            print("Could not even cache error info")
            
    finally:
        # Remove from background tasks tracking
        if date_str in background_tasks:
            del background_tasks[date_str]
            print(f"Removed {date_str} from background_tasks tracking")
        else:
            print(f"WARNING: {date_str} was not in background_tasks tracking")

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
    """Get the semantle rank of a word (0 if not found, 1000 for daily word, 1-999 for similar words)"""
    if word == daily_data["daily_word"]:
        return 1000  # The daily word always gets rank 1000
    
    # Look for the word in the similar words list
    for similar_word_data in daily_data["similar_words"]:
        if similar_word_data["word"] == word:
            return similar_word_data["rank"]  # This is already in semantle format (999, 998, 997, etc.)
    
    return 0  # Word not found in top similar words

def get_similarity_score(daily_data: Dict, word: str) -> float:
    """Get the similarity score of a word to the daily word"""
    if word == daily_data["daily_word"]:
        return 100.0  # Perfect similarity
    
    # Look for the word in the similar words list
    for similar_word_data in daily_data["similar_words"]:
        if similar_word_data["word"] == word:
            return similar_word_data["similarity"]
    
    return 0.0  # Word not found, no similarity

# ========== API ENDPOINTS ==========

@app.head("/")
def health_check_head():
    """Handle HEAD requests for health check"""
    return {}

@app.get("/")
def health_check():
    """Health check endpoint that also attempts to load model if needed"""
    if model is None:
        load_model()
    
    return {
        "status": "healthy", 
        "message": "Enhanced Word2Vec API with Semantle Scoring is running!",
        "model_loaded": model is not None,
        "current_time": datetime.now(timezone.utc).isoformat(),
        "cached_dates": list(daily_words_cache.keys()),
        "active_calculations": len([t for t in background_tasks.values() if t.is_alive()]),
        "scoring_system": "semantle_1_to_1000"
    }

@app.get("/similarity")
def get_similarity(word1: str, word2: str):
    """Get similarity between two words with semantle ranking for today's daily word"""
    # Try to load model if it's not loaded
    if model is None:
        load_model()
    
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
        percentile = 0.0
        if daily_data and daily_data["daily_word"] == word2:
            rank = get_word_rank(daily_data, word1)
            if rank > 0:
                # Calculate percentile (what percentage of players would rank lower)
                percentile = (rank / 1000.0) * 100
        
        return {
            "word1": word1,
            "word2": word2,
            "similarity": similarity_value,
            "rank": rank,
            "percentile": round(percentile, 1),
            "rank_display": f"{rank}/1000" if rank > 0 else "Not ranked",
            "is_daily_word": daily_data is not None and daily_data["daily_word"] == word2,
            "date": today,
            "scoring_system": "semantle"
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
    """Get similarity for a specific historical date with semantle ranking"""
    # Try to load model if it's not loaded
    if model is None:
        load_model()
    
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
        percentile = (rank / 1000.0) * 100 if rank > 0 else 0.0
        
        return {
            "date": date,
            "word1": word1,
            "word2": word2,
            "similarity": similarity_value,
            "rank": rank,
            "percentile": round(percentile, 1),
            "rank_display": f"{rank}/1000" if rank > 0 else "Not ranked",
            "daily_word": daily_data["daily_word"],
            "scoring_system": "semantle"
        }
        
    except Exception as e:
        print(f"Error in /similarity/historical: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

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
            "total_similar_words": len(daily_data.get("similar_words", [])),
            "calculation_method": daily_data.get("calculation_method", "unknown"),
            "ranking_system": daily_data.get("ranking_system", "semantle_1_to_1000"),
            "scoring_system": "semantle"
        }
        
    except Exception as e:
        print(f"Error in /daily-word: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.get("/rank")
def get_word_rank_endpoint(word: str, date: Optional[str] = None):
    """Get the semantle rank (1-1000) of a word for a specific date"""
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
        
        # Try to load model if it's not loaded
        if model is None:
            load_model()
        
        if model is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Model is not loaded. Please check server logs."}
            )
        
        # Check if word exists in vocabulary
        if word not in model.wv:
            return JSONResponse(
                status_code=400,
                content={"error": f"Word not found in vocabulary: {word}"}
            )
        
        daily_data = load_daily_words(target_date)
        if not daily_data:
            return JSONResponse(
                status_code=404,
                content={"error": f"No daily word data found for {target_date}"}
            )
        
        rank = get_word_rank(daily_data, word)
        similarity = get_similarity_score(daily_data, word)
        percentile = (rank / 1000.0) * 100 if rank > 0 else 0.0
        
        return {
            "date": target_date,
            "word": word,
            "daily_word": daily_data["daily_word"],
            "rank": rank,
            "percentile": round(percentile, 1),
            "rank_display": f"{rank}/1000" if rank > 0 else "Not ranked",
            "similarity": similarity,
            "is_daily_word": word == daily_data["daily_word"],
            "scoring_system": "semantle"
        }
        
    except Exception as e:
        print(f"Error in /rank: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

# ========== ADMIN ENDPOINTS ==========

@app.head("/admin/set-daily-word")
def head_set_daily_word_info(credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    """Handle HEAD requests for set daily word"""
    return {}

@app.get("/admin/set-daily-word")
def get_set_daily_word_info(date: Optional[str] = None, word: Optional[str] = None, credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    """Get info about setting daily words or actually set if params provided (admin only)"""
    print(f"GET /admin/set-daily-word called with date={date}, word={word}")
    
    if date and word:
        # If both parameters are provided, actually set the daily word
        print(f"Setting daily word: {word} for date: {date}")
        result = set_daily_word_internal(date, word)
        print(f"Result: {result}")
        return result
    else:
        # Otherwise, return usage info
        return {
            "message": "Use POST method to set daily word, or provide date and word parameters",
            "usage": "POST /admin/set-daily-word?date=dd/mm/yyyy&word=yourword",
            "alternative": "GET /admin/set-daily-word?date=dd/mm/yyyy&word=yourword",
            "example": "GET /admin/set-daily-word?date=24/07/2025&word=שלום",
            "current_cached_dates": list(daily_words_cache.keys()),
            "today": get_today_date(),
            "model_loaded": model is not None,
            "scoring_system": "semantle_1_to_1000",
            "active_calculations": {
                date: "running" if thread.is_alive() else "finished"
                for date, thread in background_tasks.items()
            }
        }

def set_daily_word_internal(date: str, word: str):
    """Internal function to set daily word"""
    print(f"set_daily_word_internal called with date={date}, word={word}")
    
    # Try to load model if it's not loaded
    if model is None:
        print("Model is None, attempting to load...")
        load_model()
    
    if model is None:
        print("Model still None after load attempt")
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not loaded. Please check server logs."}
        )
    
    print(f"Model loaded successfully, vocabulary size: {len(model.wv.key_to_index)}")
    
    try:
        # Validate date format
        try:
            parse_date(date)
            print(f"Date {date} parsed successfully")
        except ValueError as e:
            print(f"Date parsing failed: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid date format. Use dd/mm/yyyy"}
            )
        
        # Check if word exists in vocabulary
        if word not in model.wv:
            print(f"Word '{word}' not found in vocabulary")
            # Show some similar words for debugging
            sample_words = list(model.wv.key_to_index.keys())[:10]
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Word not found in vocabulary: {word}",
                    "sample_vocabulary": sample_words
                }
            )
        
        print(f"Word '{word}' found in vocabulary")
        
        # Check if there's already a background task for this date
        if date in background_tasks and background_tasks[date].is_alive():
            print(f"Background task already running for {date}")
            return JSONResponse(
                status_code=409,
                content={"error": f"Already calculating daily words for {date}. Please wait."}
            )
        
        # Start background calculation
        print(f"Starting background calculation for {date} with word '{word}'")
        thread = threading.Thread(
            target=calculate_daily_words_background,
            args=(date, word),
            daemon=True
        )
        background_tasks[date] = thread
        thread.start()
        print(f"Background thread started for {date}")
        
        return {
            "message": f"Daily word for {date} set to '{word}'. Calculating semantle rankings in background.",
            "date": date,
            "daily_word": word,
            "status": "processing",
            "thread_id": thread.ident,
            "scoring_system": "semantle_1_to_1000",
            "expected_ranks": "999 most similar words will get ranks 999-1, daily word gets rank 1000"
        }
        
    except Exception as e:
        print(f"Error in set_daily_word_internal: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.post("/admin/set-daily-word")
def set_daily_word(date: str, word: str, credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    """Set the daily word for a specific date (admin only)"""
    return set_daily_word_internal(date, word)

@app.get("/admin/calculation-status")
def get_calculation_status(date: str, credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    """Check if background calculation is complete for a date (admin only)"""
    try:
        parse_date(date)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid date format. Use dd/mm/yyyy"}
        )
    
    is_running = date in background_tasks and background_tasks[date].is_alive()
    is_cached = date in daily_words_cache
    has_error = f"{date}_error" in daily_words_cache
    
    status = "unknown"
    if is_running:
        status = "running"
    elif is_cached and not has_error:
        status = "completed"
    elif has_error:
        status = "error"
    else:
        status = "not_started"
    
    result = {
        "date": date,
        "status": status,
        "is_running": is_running,
        "is_cached": is_cached,
        "has_error": has_error,
        "scoring_system": "semantle_1_to_1000"
    }
    
    if is_cached:
        data = daily_words_cache[date]
        result["daily_word"] = data.get("daily_word")
        result["word_count"] = len(data.get("similar_words", []))
        result["created_at"] = data.get("created_at")
        result["calculation_method"] = data.get("calculation_method")
        result["ranking_system"] = data.get("ranking_system", "semantle_1_to_1000")
    
    if has_error:
        error_data = daily_words_cache[f"{date}_error"]
        result["error_info"] = {
            "error": error_data.get("error"),
            "error_type": error_data.get("error_type"),
            "created_at": error_data.get("created_at"),
            "thread_id": error_data.get("thread_id")
        }
    
    return result

@app.post("/admin/retry-calculation")
def retry_calculation(date: str, word: str, force: bool = False, credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    """Retry or force restart the background calculation for a specific date (admin only)"""
    
    try:
        parse_date(date)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid date format. Use dd/mm/yyyy"}
        )
    
    # Try to load model if it's not loaded
    if model is None:
        load_model()
    
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not loaded. Please check server logs."}
        )
    
    # Check if word exists in vocabulary
    if word not in model.wv:
        return JSONResponse(
            status_code=400,
            content={"error": f"Word not found in vocabulary: {word}"}
        )
    
    # Check current status
    is_running = date in background_tasks and background_tasks[date].is_alive()
    is_cached = date in daily_words_cache
    
    if is_running and not force:
        return JSONResponse(
            status_code=409,
            content={
                "error": f"Calculation is already running for {date}. Use force=true to restart.",
                "current_status": "running"
            }
        )
    
    if is_cached and not force:
        data = daily_words_cache[date]
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Calculation already completed for {date}. Use force=true to recalculate.",
                "current_status": "completed",
                "daily_word": data.get("daily_word"),
                "word_count": len(data.get("similar_words", [])),
                "created_at": data.get("created_at"),
                "scoring_system": "semantle_1_to_1000"
            }
        )
    
    # Clean up any existing background task
    if date in background_tasks:
        old_thread = background_tasks[date]
        if old_thread.is_alive():
            print(f"WARNING: Forcing restart while thread is still alive for {date}")
        del background_tasks[date]
    
    # Clean up any cached data if forcing
    if force:
        if date in daily_words_cache:
            del daily_words_cache[date]
            print(f"Cleared cached data for {date}")
        if f"{date}_error" in daily_words_cache:
            del daily_words_cache[f"{date}_error"]
            print(f"Cleared cached error data for {date}")
    
    # Start new background calculation
    print(f"Starting{'(FORCED)' if force else ''} background calculation for {date} with word '{word}'")
    thread = threading.Thread(
        target=calculate_daily_words_background,
        args=(date, word),
        daemon=True
    )
    background_tasks[date] = thread
    thread.start()
    
    return {
        "message": f"{'Restarted' if force else 'Started'} calculation for {date} with word '{word}'",
        "date": date,
        "daily_word": word,
        "status": "processing",
        "forced": force,
        "thread_id": thread.ident,
        "scoring_system": "semantle_1_to_1000"
    }

@app.get("/admin/clear-cache")
def clear_cache(date: Optional[str] = None, credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    """Clear cached data for a specific date or all dates (admin only)"""
    
    if date:
        try:
            parse_date(date)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid date format. Use dd/mm/yyyy"}
            )
        
        cleared = []
        if date in daily_words_cache:
            del daily_words_cache[date]
            cleared.append(date)
        
        if f"{date}_error" in daily_words_cache:
            del daily_words_cache[f"{date}_error"]
            cleared.append(f"{date}_error")
        
        return {
            "message": f"Cleared cache for {date}",
            "cleared_entries": cleared,
            "remaining_entries": list(daily_words_cache.keys())
        }
    else:
        # Clear all cache
        cache_size = len(daily_words_cache)
        cleared_entries = list(daily_words_cache.keys())
        daily_words_cache.clear()
        
        return {
            "message": "Cleared all cached data",
            "cleared_count": cache_size,
            "cleared_entries": cleared_entries
        }

@app.post("/admin/download-model")
def download_model_endpoint(credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    """Force download model files (admin only)"""
    global model
    model = None  # Reset the model
    
    print("Starting manual model download...")
    success = download_model_files()
    
    if success:
        model = load_model()
        return {
            "message": "Model download completed",
            "model_loaded": model is not None,
            "files_in_directory": os.listdir(".") if os.path.exists(".") else []
        }
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to download model files"}
        )

@app.get("/admin/status")
def get_admin_status(credentials: HTTPBasicCredentials = Depends(verify_admin_credentials)):
    """Get status of daily words and background tasks (admin only)"""
    return {
        "cached_dates": list(daily_words_cache.keys()),
        "cached_errors": [key for key in daily_words_cache.keys() if "_error" in key],
        "background_tasks": {
            date: {
                "status": "running" if thread.is_alive() else "completed",
                "thread_id": thread.ident,
                "is_alive": thread.is_alive()
            }
            for date, thread in background_tasks.items()
        },
        "today": get_today_date(),
        "model_loaded": model is not None,
        "model_vocab_size": len(model.wv.key_to_index) if model else 0,
        "current_time": datetime.now(timezone.utc).isoformat(),
        "scoring_system": "semantle_1_to_1000"
    }

# ========== DEBUG ENDPOINTS ==========

@app.get("/debug/background-status")
def debug_background_status():
    """Debug endpoint to check background task status"""
    return {
        "active_background_tasks": {
            date: {
                "status": "running" if thread.is_alive() else "finished",
                "thread_id": thread.ident,
                "is_daemon": thread.daemon
            }
            for date, thread in background_tasks.items()
        },
        "cached_dates": list(daily_words_cache.keys()),
        "cached_errors": [key for key in daily_words_cache.keys() if "_error" in key],
        "model_loaded": model is not None,
        "current_time": datetime.now(timezone.utc).isoformat(),
        "total_threads": threading.active_count(),
        "scoring_system": "semantle_1_to_1000"
    }

@app.get("/debug/test-model")
def test_model():
    """Test if the model is working by checking a common word"""
    if model is None:
        load_model()
    
    if model is None:
        return {"error": "Model not loaded", "model_loaded": False}
    
    try:
        # Test with a common Hebrew word
        test_word = "בית"  # house
        if test_word in model.wv:
            similar = model.wv.most_similar(test_word, topn=5)
            return {
                "model_loaded": True,
                "test_word": test_word,
                "similar_words": [{"word": word, "similarity": float(sim)} for word, sim in similar],
                "vocabulary_size": len(model.wv.key_to_index),
                "scoring_system": "semantle_1_to_1000"
            }
        else:
            # Try another common word
            test_words = ["אני", "את", "של", "על", "כל", "לא", "יש"]
            found_word = None
            for tw in test_words:
                if tw in model.wv:
                    found_word = tw
                    break
            
            if found_word:
                similar = model.wv.most_similar(found_word, topn=5)
                return {
                    "model_loaded": True,
                    "test_word": found_word,
                    "original_test_word": test_word,
                    "original_test_word_found": False,
                    "similar_words": [{"word": word, "similarity": float(sim)} for word, sim in similar],
                    "vocabulary_size": len(model.wv.key_to_index),
                    "scoring_system": "semantle_1_to_1000"
                }
            else:
                return {
                    "model_loaded": True,
                    "test_word": test_word,
                    "error": f"None of the test words found in vocabulary",
                    "vocabulary_size": len(model.wv.key_to_index),
                    "sample_words": list(model.wv.key_to_index.keys())[:20],
                    "scoring_system": "semantle_1_to_1000"
                }
    except Exception as e:
        return {
            "model_loaded": True,
            "error": f"Error testing model: {str(e)}",
            "error_type": str(type(e))
        }

@app.get("/debug/files")
def debug_files():
    """Debug endpoint to check what files are available"""
    try:
        current_dir = os.getcwd()
        files = os.listdir(".")
        
        model_files_info = {}
        model_files = ["model.mdl", "model.mdl.wv.vectors.npy", "model.mdl.syn1neg.npy"]
        
        for filename in model_files:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                model_files_info[filename] = {
                    "exists": True,
                    "size": size,
                    "size_mb": round(size / (1024*1024), 2)
                }
            else:
                model_files_info[filename] = {"exists": False}
        
        return {
            "current_directory": current_dir,
            "all_files": files,
            "model_files": model_files_info,
            "model_loaded": model is not None,
            "data_directory_exists": os.path.exists("data"),
            "data_files": os.listdir("data") if os.path.exists("data") else [],
            "scoring_system": "semantle_1_to_1000"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/cache-contents")
def debug_cache_contents():
    """Debug endpoint to see what's in the cache"""
    cache_info = {}
    
    for key, value in daily_words_cache.items():
        if isinstance(value, dict):
            cache_info[key] = {
                "type": "error" if "_error" in key else "data",
                "daily_word": value.get("daily_word"),
                "created_at": value.get("created_at"),
                "word_count": len(value.get("similar_words", [])) if "similar_words" in value else 0,
                "has_error": "error" in value,
                "status": value.get("status"),
                "keys": list(value.keys()),
                "ranking_system": value.get("ranking_system", "unknown")
            }
        else:
            cache_info[key] = {"type": "unknown", "value_type": str(type(value))}
    
    return {
        "cache_entries": cache_info,
        "total_entries": len(daily_words_cache),
        "error_entries": len([k for k in daily_words_cache.keys() if "_error" in k]),
        "scoring_system": "semantle_1_to_1000"
    }

@app.get("/debug/test-ranking")
def debug_test_ranking():
    """Debug endpoint to test the ranking system with sample data"""
    if model is None:
        load_model()
    
    if model is None:
        return {"error": "Model not loaded", "model_loaded": False}
    
    try:
        # Use a common Hebrew word for testing
        test_words = ["בית", "אני", "את", "של", "על"]
        found_word = None
        
        for word in test_words:
            if word in model.wv:
                found_word = word
                break
        
        if not found_word:
            return {"error": "No test words found in vocabulary"}
        
        # Get top 10 similar words to demonstrate ranking
        similar_words = model.wv.most_similar(found_word, topn=10)
        
        # Create sample ranking data like our system would
        ranking_demo = []
        for i, (word, similarity) in enumerate(similar_words):
            semantle_rank = 999 - i  # 999, 998, 997, etc.
            ranking_demo.append({
                "word": word,
                "similarity": float(similarity),
                "rank": semantle_rank,
                "rank_display": f"{semantle_rank}/1000"
            })
        
        # Add the target word (always rank 1000)
        ranking_demo.insert(0, {
            "word": found_word,
            "similarity": 1.0,
            "rank": 1000,
            "rank_display": "1000/1000",
            "note": "This is the daily word (target)"
        })
        
        return {
            "test_word": found_word,
            "ranking_system": "semantle_1_to_1000",
            "explanation": {
                "daily_word_rank": 1000,
                "most_similar_rank": 999,
                "second_most_similar_rank": 998,
                "least_similar_in_top_999": 1,
                "not_in_top_999": 0
            },
            "sample_rankings": ranking_demo,
            "model_loaded": True,
            "vocabulary_size": len(model.wv.key_to_index)
        }
        
    except Exception as e:
        return {
            "error": f"Error testing ranking: {str(e)}",
            "error_type": str(type(e))
        }
