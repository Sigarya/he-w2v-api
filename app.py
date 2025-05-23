import os
from gensim.models import Word2Vec
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import traceback # הוסף ייבוא זה אם עדיין לא הוספת

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "API is running!"}

model_path = os.getenv("MODEL_PATH", "model.mdl")
print(f"Attempting to load model from: {model_path}")
try:
    model = Word2Vec.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    traceback.print_exc() # כדאי להוסיף גם כאן למקרה של שגיאה בטעינה
    model = None


@app.get("/similarity")
def get_similarity(word1: str, word2: str):
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not loaded. Please check server logs."}
        )
    try:
        print(f"Calculating similarity for: '{word1}' and '{word2}'")
        
        # בדיקה אם המילים קיימות באוצר המילים
        if word1 not in model.wv:
            print(f"Word '{word1}' not in vocabulary.")
            return JSONResponse(
                status_code=400,
                content={"error": f"Word not found in vocabulary: {word1}"}
            )
        if word2 not in model.wv:
            print(f"Word '{word2}' not in vocabulary.")
            return JSONResponse(
                status_code=400,
                content={"error": f"Word not found in vocabulary: {word2}"}
            )
        
        similarity_value = model.wv.similarity(word1, word2)
        print(f"Similarity calculated (numpy.float32): {similarity_value}")
        
        # !!! התיקון הקריטי כאן !!!
        # המר את הערך ל-float רגיל של פייתון
        final_similarity = float(similarity_value)
        print(f"Similarity converted to Python float: {final_similarity}")

        return {"word1": word1, "word2": word2, "similarity": final_similarity}

    except KeyError as e: # זה לרוב ייתפס על ידי הבדיקות למעלה
        print(f"KeyError in similarity for '{word1}' or '{word2}': {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Word not found in vocabulary: {e.args[0]}"}
        )
    except Exception as e:
        print(f"!!! UNEXPECTED ERROR in /similarity for '{word1}', '{word2}' !!!")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected internal server error. Check server logs for details."}
        )