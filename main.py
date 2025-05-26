import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import whisper
import logging
import tempfile
import os

from starlette.middleware.cors import CORSMiddleware

import backend
from starlette.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


@app.on_event("startup")
async def startup_event():
    global _model
    logging.info("Loading Whisper model...")
    _model = whisper.load_model("base")
    logging.info("Model loaded.")


class IngredientsRequest(BaseModel):
    ingredients: list[str]
    language: str = "en"



@app.post("/find_recipes")
def find_recipes(request: IngredientsRequest):
    matching_recipes = backend.find_recipes_from_api(request.ingredients)
    if request.language != "en":
        matching_recipes = backend.translate_recipes_to_english({"recipes": matching_recipes}, request.language)
    return {"recipes": matching_recipes}

class IngredientsTranslateRequest(BaseModel):
    ingredients: list[str]
    language: str

@app.post("/translate")
async def translate_ingredients(request: IngredientsTranslateRequest):
    translated = backend.translate_to_english(request.ingredients, request.language)
    return {"translated_ingredients": translated}


ALLOWED_EXTENSIONS = {"mp3", "m4a"}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Only mp3 and m4a are allowed.")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + file.filename.rsplit('.', 1)[1]) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        result = backend.transcribe_audio_file(_model, temp_file_path)
        return {
            "text": result["text"],
            "language": result["language"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
