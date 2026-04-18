from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uuid
import os
import edge_tts
import requests
import httpx
from groq import Groq
from google import genai
from fastapi.middleware.cors import CORSMiddleware
import time
from dotenv import load_dotenv
from fastapi import Request
import asyncio
import hashlib
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi import BackgroundTasks
from google.genai import types
from fastapi.responses import StreamingResponse

load_dotenv()  # loads .env file

CACHE = {}
CACHE_TTL = 60 * 60  # 1 hour
def clean_cache():
    now = time.time()
    for k in list(CACHE.keys()):
        if now - CACHE[k]["time"] > CACHE_TTL:
            del CACHE[k]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# 1. This is the new function that handles the disk write
async def save_audio_file(text, output_path):
    """Saves the full audio to disk for caching."""
    communicate = edge_tts.Communicate(text, "en-GB-RyanNeural")
    await communicate.save(output_path)

# 2. Update safe_tts to call it
async def safe_tts(text, output_file, retries=2):
    for attempt in range(retries):
        try:
            # REPLACE the old call with save_audio_file
            await save_audio_file(text, output_file)
            return True
        except Exception as e:
            print(f"TTS attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(0.5)
    return False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Audio storage
AUDIO_DIR = "/tmp/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# 🌐 Serve audio files publicly
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

    
    
def ask_groq(user_text, search_context=""):
    models = [
        "llama-3.1-8b-instant",      # Fast, lightweight, conversational
        "llama-3.3-70b-versatile"    # Slower, heavyweight fallback
    ]

    for model_name in models:
        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Alan, a voice AI assistant. "
                            "Use provided context if available. "
                            "Be natural, short, and spoken."
                            "If the user asks about current events, news, weather, or something you "
                            "don't know for sure, respond ONLY with the tag: [SEARCH_REQUIRED]. "
                            "Otherwise, answer naturally and briefly."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"""
User question:
{user_text}

Context (may be empty):
{search_context}

Respond naturally and clearly.
"""
                    }
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Groq model failed ({model_name}):", e)

    return "Sorry, I couldn't generate a response right now."
    
def get_cache_key(text: str):
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def is_cache_valid(entry):
    return time.time() - entry["time"] < CACHE_TTL
    
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        transcription = groq_client.audio.transcriptions.create(
            file=("audio.wav", f.read()),
            model="whisper-large-v3"
        )
    return transcription.text    
    

class SpeakRequest(BaseModel):
    text: str
    
# 🔊 EDGE TTS (fallback engine)
async def stream_audio_generator(text):
    communicate = edge_tts.Communicate(text, "en-GB-RyanNeural")
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

# 2. Keep this for BackgroundTasks (Saves to file)
async def save_audio_file(text, output_path):
    communicate = edge_tts.Communicate(text, "en-GB-RyanNeural")
    await communicate.save(output_path)


# 🧠 QWEN TTS (placeholder for now)
HF_TOKEN = os.getenv("HF_TOKEN")

async def try_qwen_tts(text, output_path):
    try:
        API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen3-TTS-12Hz-0.6B"
        
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}"
        }

        payload = {
            "inputs": text
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(API_URL, headers=headers, json=payload)

        content_type = response.headers.get("content-type", "")

        if response.status_code != 200 or not content_type.startswith("audio/"):
         print("Qwen failed:", response.text)
         return False

        # Save audio
        with open(output_path, "wb") as f:
            f.write(response.content)

        return True

    except Exception as e:
        print("Qwen error:", e)
        return False

# 🎤 Upload voice (for STT later)
@app.post("/upload-voice")
async def upload_voice(file: UploadFile = File(...)):
    # 1. Create a unique ID and path
    file_id = str(uuid.uuid4())
    file_path = f"{AUDIO_DIR}/{file_id}_{file.filename}"

    try:
        # 2. Write the uploaded file to disk
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 3. Transcribe (Run in thread to avoid blocking the event loop)
        # Ensure transcribe_audio is the function using Groq Whisper
        text = await asyncio.to_thread(transcribe_audio, file_path)

        return {
            "status": "success",
            "text": text
        }

    except Exception as e:
        print(f"❌ STT Error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": "Failed to process audio"}
        )

    finally:
        # 4. 🔥 THE CLEANUP: Always delete the file, even if it fails
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🗑️ Cleaned up temporary file: {file_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Failed to delete {file_path}: {cleanup_error}")


def needs_search(text):
    keywords = ["who", "what", "latest", "news", "price", "weather", "when", "current", "where","update"]
    return any(k in text.lower() for k in keywords)

def get_search_context(user_text):
    try:
        # 1. Define the Grounding Tool
        # This tells Gemini: "You have permission to use Google Search"
        search_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        # 2. Call the model with the tool enabled
        # Using gemini-2.0-flash (the 2026 standard) for maximum speed
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=user_text,
            config=types.GenerateContentConfig(
                tools=[search_tool]
            )
        )

        # 3. Extra Safety: Log if grounding actually happened
        if response.candidates[0].grounding_metadata:
             print(f"✅ Grounded with Google Search: {user_text}")

        return response.text
    except Exception as e:
        print(f"❌ Gemini Search failed: {e}")
        return ""
    

        
# 🔊 MAIN TTS ENDPOINT (Qwen → Edge fallback) uvicorn main:app --host 0.0.0.0 --port 8000
@app.post("/speak")
async def speak(request: Request, data: SpeakRequest, background_tasks: BackgroundTasks):
    user_text = data.text
    key = get_cache_key(user_text)

    # ⚡ CACHE HIT
    if key in CACHE and is_cache_valid(CACHE[key]):
        return JSONResponse(CACHE[key]["data"])
    # 🔎 STEP 1: ONLY if needed
    # STEP 1: Ask Groq if it knows the answer
    initial_response = await asyncio.to_thread(ask_groq, user_text)

    if "[SEARCH_REQUIRED]" in initial_response:
        print("🔍 Alan is searching Google...")
        # Get real-time facts from Gemini
        search_context = await asyncio.to_thread(get_search_context, user_text)
        
        # Get final spoken answer based on facts
        # Note: Make sure you have 'ask_groq_final' defined or just reuse ask_groq with context
        ai_response = await asyncio.to_thread(ask_groq, user_text, search_context)
    else:
        ai_response = initial_response


    # 🔊 TTS
    file_id = str(uuid.uuid4())
    output_file = f"{AUDIO_DIR}/{file_id}.mp3"


    #success = await safe_tts(ai_response, output_file)

   # if not success:
   #  return JSONResponse({
    #    "audio_url": None,
    #    "text": ai_response
   # })

    # This sends the response to the user WITHOUT waiting for the audio to finish
    background_tasks.add_task(save_audio_file, ai_response, output_file)
    
    base_url = str(request.base_url)

    response_data = {
    "audio_url": f"{base_url}audio/{file_id}.mp3",
    "text": ai_response
}

    CACHE[key] = {
    "data": response_data,
    "time": time.time()
}
    clean_cache()
  
    return StreamingResponse(
        stream_audio_generator(ai_response),
        media_type="audio/mpeg",
        headers={
            "X-AI-Text": ai_response.replace("\n", " "), # No newlines in headers
            "Access-Control-Expose-Headers": "X-AI-Text" # Crucial for Frontend
        }
    )

