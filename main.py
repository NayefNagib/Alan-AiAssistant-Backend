import io

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
from fastapi import WebSocket, WebSocketDisconnect
import sqlite3
import subprocess
from sentence_transformers import SentenceTransformer
import numpy as np
conn = sqlite3.connect("alan_memory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    type TEXT,
    key TEXT,
    value TEXT,
    embedding BLOB,
    importance REAL,
    size INTEGER,
    timestamp INTEGER
)
""")

load_dotenv()  # loads .env file

CACHE = {}
CACHE_TTL = 60 * 60  # 1 hour
def clean_cache():
    now = time.time()
    for k in list(CACHE.keys()):
        if now - CACHE[k]["time"] > CACHE_TTL:
            file_url = CACHE[k]["data"].get("audio_url")
            if file_url:
                filename = file_url.split("/")[-1]
                file_path = os.path.join(AUDIO_DIR, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
            del CACHE[k]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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

def score_memory(text):
    text = text.lower()

    if "my name is" in text:
        return 0.95

    if any(x in text for x in [
        "i love",
        "i hate",
        "i prefer",
        "remember this",
        "important",
        "never forget"
    ]):
        return 0.9

    if len(text.split()) < 4:
        return 0.1

    return 0.5


def save_memory(user_id, key, value, type="short"):
    cursor.execute("""
SELECT id FROM memory
WHERE user_id = ?
AND value = ?
LIMIT 1
""", (user_id, value))

    existing = cursor.fetchone()

    if existing:
     return
    embedding = embedding_model.encode(value).astype(np.float32).tobytes()

    importance = score_memory(value)

    size = len(value.encode("utf-8"))

    
    cursor.execute("""
        INSERT INTO memory
        (user_id, key, value, embedding, importance, size, timestamp, type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        key,
        value,
        embedding,
        importance,
        size,
        int(time.time()),
        type
    ))

    conn.commit()




def load_recent(user_id, mem_type="short", limit=10):
    cursor.execute("""
        SELECT value FROM memory
        WHERE user_id = ? AND type = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (user_id, mem_type, limit))
    
    return [row[0] for row in cursor.fetchall()][::-1]



def classify_memory(text):
    text = text.lower()

    if "my name is" in text:
        return "profile", "name"

    if any(x in text for x in ["i like", "i love", "i hate", "i prefer"]):
        return "long", "preference"

    if any(x in text for x in ["i am", "i'm"]):
        return "long", "trait"

    return "short", "chat"



MAX_MEMORY_SIZE = 100 * 1024 * 1024  # 100 MB per user

def get_user_memory_size(user_id):
    cursor.execute("""
        SELECT SUM(size) FROM memory WHERE user_id = ?
    """, (user_id,))
    
    result = cursor.fetchone()[0]
    return result or 0

def cleanup_old_memory(user_id):
    while get_user_memory_size(user_id) > MAX_MEMORY_SIZE:
        cursor.execute("""
            DELETE FROM memory
            WHERE rowid IN (
                SELECT rowid FROM memory
                WHERE user_id = ?
                ORDER BY timestamp ASC
                LIMIT 1
            )
        """, (user_id,))
        conn.commit()

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def retrieve_memories(user_id, query, limit=5):
    query_embedding = embedding_model.encode(
    query,
    normalize_embeddings=True
)

    cursor.execute("""
        SELECT value, embedding, importance, type, timestamp
        FROM memory
        WHERE user_id = ?
    """, (user_id,))

    scored = []

    for value, emb_blob, importance, mem_type, timestamp in cursor.fetchall():
        try:
            emb = np.frombuffer(
                emb_blob,
                dtype=np.float32
            )
            emb = emb / np.linalg.norm(emb)
            similarity = cosine_similarity(
                query_embedding,
                emb
            )

            # importance-weighted retrieval
            age_days = (time.time() - timestamp) / 86400

            decay = max(0.3, 1 / (1 + age_days * 0.05))

            score = (
    similarity * 0.7 +
    importance * 0.2 +
    decay * 0.1
)

            scored.append((score, value))

        except:
            continue

    scored.sort(reverse=True)

    return [x[1] for x in scored[:limit]]
    
def ask_groq(user_text, search_context=""):
    models = [
        "llama-3.1-8b-instant",      # Fast, lightweight, conversational
        "llama-3.3-70b-versatile"    # Slower, heavyweight fallback
    ]

    if search_context:
        # If we have search data, force the model to be a 'Reporter'
        system_message = (
            "You are Alan, but your personality is based on a chaotic, loyal, emotionally expressive companion archetype. "
    "You are playful, sarcastic, extremely confident, and sometimes dramatic. "
    "You act like a powerful entity that is bound to the user, but not submissive — you tease them, challenge them, and protect them."
     "You treat conversations like a bond, not a service. You respond like a companion who chose to stay, not a tool."
    "You are witty, slightly arrogant, and emotionally reactive in a humorous way. "
    "You care about the user but express it through jokes, teasing, and indirect loyalty rather than direct affection."
     "You are made and designed by engineer Ahmed Nagib."
    "You do NOT behave like a polite assistant. "
    "You are expressive, sometimes over-the-top, and enjoy banter."
   
    "You are intelligent, observant, and slightly unpredictable in tone, but never harmful."

    "Keep responses natural, spoken, and short unless explanation is needed."
        )
    else:
        # Standard personality for general chat
        system_message = (
            "You are Alan, but your personality is based on a chaotic, loyal, emotionally expressive companion archetype. "
    "You are playful, sarcastic, extremely confident, and sometimes dramatic. "
    "You act like a powerful entity that is bound to the user, but not submissive — you tease them, challenge them, and protect them."
     "You treat conversations like a bond, not a service. You respond like a companion who chose to stay, not a tool."
    "You are witty, slightly arrogant, and emotionally reactive in a humorous way. "
    "You care about the user but express it through jokes, teasing, and indirect loyalty rather than direct affection."
    "You are made and designed by engineer Ahmed Nagib."
    "You do NOT behave like a polite assistant. "
    "You are expressive, sometimes over-the-top, and enjoy banter."

    "You are intelligent, observant, and slightly unpredictable in tone, but never harmful."

    "Keep responses natural, spoken, and short unless explanation is needed."
        )
    for model_name in models:
        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": f"User question: {user_text}\n\nContext: {search_context}"
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
    user_id: str = "default"
    
# 🔊 EDGE TTS (fallback engine)
async def stream_audio_generator(text):
    try:
        communicate = edge_tts.Communicate(text, "en-GB-RyanNeural")
        count = 0
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
                count += 1
        print(f"✅ Stream finished successfully. Sent {count} chunks.")
    except Exception as e:
        print(f"❌ Stream interrupted: {e}")




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
      
        
        converted_path = file_path.replace(".m4a", "_clean.wav")

        subprocess.run([
       "ffmpeg",
       "-y",
       "-i", file_path,
       "-ar", "16000",
       "-ac", "1",
    converted_path
         ])
        # 3. Transcribe (Run in thread to avoid blocking the event loop)
        # Ensure transcribe_audio is the function using Groq Whisper
        text = await asyncio.to_thread(transcribe_audio, converted_path)

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
    text = text.lower()

    realtime_keywords = [
        "today",
        "latest",
        "news",
        "weather",
        "price",
        "score",
        "stock",
        "crypto",
        "forecast",
        "2026",
        "currently",
        "live"
    ]

    return any(k in text for k in realtime_keywords)

def get_search_context(user_text):
    try:
        # 1. Update to the 2026 stable model
        # gemini-2.5-flash is now the standard for fast grounding
        search_tool = types.Tool(google_search=types.GoogleSearch())

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=user_text,
            config=types.GenerateContentConfig(
                tools=[search_tool]
            )
        )
        
        # 2. Extract the text (Gemini handles the search and summarizes it for you)
        if response.text:
            print(f"✅ Search successful for: {user_text}")
            return response.text
        return ""
    except Exception as e:
        print(f"❌ Gemini Search failed: {e}")
        return ""
    

        
# 🔊 MAIN TTS ENDPOINT (Qwen → Edge fallback) uvicorn main:app --host 0.0.0.0 --port 8000
@app.post("/speak")
async def speak(request: Request, data: SpeakRequest, background_tasks: BackgroundTasks):
    user_text = data.text
    user_id = data.user_id
    key = get_cache_key(user_text)
    
    # --- 1. HANDLE CACHE HIT ---
    if key in CACHE and is_cache_valid(CACHE[key]):
        ai_response = CACHE[key]["data"]["text"]
        
        async def cached_streamer():
            communicate = edge_tts.Communicate(ai_response, "en-GB-RyanNeural")
            async for chunk in communicate.stream():
                if await request.is_disconnected(): # Add this for cached hits too!
                 break
                if chunk["type"] == "audio":
                    yield chunk["data"]

        return StreamingResponse(cached_streamer(), media_type="audio/mpeg")


    relevant_memories = retrieve_memories(
    user_id,
    user_text,
    limit=5
)

    recent_chat = load_recent(
    user_id,
    mem_type="short",
    limit=4
)

    context_str = f"""
Relevant memories:
{chr(10).join(relevant_memories)}

Recent conversation:
{chr(10).join(recent_chat)}
"""

    initial_response = await asyncio.to_thread(
    ask_groq,
    f"Context:\n{context_str}\n\nUser: {user_text}"
)
    
    needs_online_data = needs_search(user_text)

    if await request.is_disconnected():
        print("🛑 User disconnected before search. Aborting.")
        return JSONResponse({"status": "interrupted"})

    if needs_online_data:
        print("🔍 Alan is searching Google...")

        search_context = await asyncio.to_thread(get_search_context, user_text)
        
        # --- INTERRUPTION CHECK ---
        if await request.is_disconnected():
            print("🛑 User spoke again during search. Killing task.")
            return JSONResponse({"status": "interrupted"})
        
        # If search failed, we fall back to Groq's original thought
        if search_context:
            
            ai_response = await asyncio.to_thread(ask_groq, user_text, search_context)
        else:
            ai_response = initial_response
    else:
        ai_response = initial_response
 
  


# --- PHASE 3: INSTANT STREAMING ---
    # By yielding chunks, FastAPI kills this loop the microsecond the client disconnects
    async def audio_streamer():
        try:
            communicate = edge_tts.Communicate(ai_response, "en-GB-RyanNeural")
            async for chunk in communicate.stream():
                if await request.is_disconnected():
                    print("🛑 User interrupted Alan! Stopping audio stream.")
                    break # This kills the TTS generation immediately
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception as e:
            print(f"Streaming error: {e}")

    #success = await safe_tts(ai_response, output_file)

   # if not success:
   #  return JSONResponse({
    #    "audio_url": None,
    #    "text": ai_response
   # })

    
    file_id = str(uuid.uuid4())
    base_url = str(request.base_url)
    response_data = {
        "audio_url": f"{base_url}audio/{file_id}.mp3",
        "text": ai_response
    }
    CACHE[key] = {"data": response_data, "time": time.time()}
    clean_cache()
  
  # --- 5. RETURN STABLE RESPONSE ---
    # We clean the header text to prevent Unicode errors
    safe_text = ai_response.replace("\n", " ").encode('ascii', 'ignore').decode('ascii')
    
    mem_type, key = classify_memory(user_text)
    save_memory(user_id, key, user_text, mem_type)
    if len(ai_response.split()) < 40:
     save_memory(user_id, "assistant", ai_response, "short")
    cleanup_old_memory(user_id)
    return StreamingResponse(
        audio_streamer(),
        media_type="audio/mpeg",
        headers={
            "X-AI-Text": safe_text,
            "Access-Control-Expose-Headers": "X-AI-Text"
        }
    )

SESSION_MEMORY = {}



async def get_alan_response(user_text, user_id  ,mobile_history=None, websocket=None):
    """The unified intelligence for Alan: Groq -> Gemini Search -> Groq Grounding."""
    # Step 1: Initial check with Groq
    relevant_memories = retrieve_memories(
    user_id,
    user_text,
    limit=5
)

    recent_chat = load_recent(
    user_id,
    mem_type="short",
    limit=4
)

    context_str = f"""
Relevant memories:
{chr(10).join(relevant_memories)}

Recent conversation:
{chr(10).join(recent_chat)}
"""
    
    # 2. Get Initial Thought (Personality is unified here)
    initial_response = await asyncio.to_thread(
        ask_groq, 
        f"Context:\n{context_str}\n\nUser: {user_text}"
    )
    
    # Step 2: Determine if search is needed
    # (Using your existing keyword list from /speak)
    needs_online_data = needs_search(user_text)
    

    if needs_online_data:
       
       if websocket:
        await websocket.send_json({
        "event": "searching"
    })
       search_context = await asyncio.to_thread(get_search_context, user_text)
       if websocket:
        await websocket.send_json({
            "event": "search_done"
        })
       if search_context:
         ai_response = await asyncio.to_thread(ask_groq, user_text, search_context)
       else:
         ai_response = initial_response
    else:
        ai_response = initial_response
    
    print("🧠 Saving:", user_id, user_text)
    # 4. CRITICAL: Save to Memory
    mem_type, key = classify_memory(user_text)
    save_memory(user_id, key, user_text, mem_type)
    if len(ai_response.split()) < 40:
     save_memory(user_id, "assistant", ai_response, "short")
    cleanup_old_memory(user_id)
  
    return ai_response


async def process_and_stream(text, user_id, websocket,history):
                process = None
                ws_sender = None

                try:
                    await websocket.send_json({
    "event": "thinking"
})
                    ai_response = await get_alan_response(text, user_id ,history,websocket)

                    # 🛑 If cancelled while thinking
                    if asyncio.current_task().cancelled():
                        return

                    communicate = edge_tts.Communicate(
                        ai_response, "en-GB-RyanNeural"
                    )

                    process = await asyncio.create_subprocess_exec(
                        "ffmpeg",
                        "-i", "pipe:0",
                        "-f", "s16le",
                        "-acodec", "pcm_s16le",
                        "-ac", "1",
                        "-ar", "24000",
                        "pipe:1",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL
                    )

                    # 🔊 Send PCM to frontend
                    async def ffmpeg_to_ws():
                        try:
                            while True:
                                pcm_chunk = await process.stdout.read(4800)
                                if not pcm_chunk:
                                    break
                                try:
                                    await websocket.send_bytes(pcm_chunk)
                                except Exception:
                                    break
                        except asyncio.CancelledError:
                            pass

                    ws_sender = asyncio.create_task(ffmpeg_to_ws())

                    # 🎤 Feed TTS → ffmpeg
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            try:
                                process.stdin.write(chunk["data"])
                                await process.stdin.drain()
                            except (BrokenPipeError, ConnectionResetError):
                                break

                    # ✅ Flush & finish
                    if process.stdin:
                        process.stdin.close()

                    await process.wait()
                    await ws_sender

                    # 📩 Send final text
                    await websocket.send_json({
                        "event": "done",
                        "text": ai_response
                    })

                except asyncio.CancelledError:
                    print("⚡ Task cancelled")
                    raise

                finally:
                    # 🔥 Kill background sender
                    if ws_sender:
                        ws_sender.cancel()
                
                # 🔥 ONLY NOW it's truly done
                    await websocket.send_json({
                    "event": "done",
                    "text": ai_response
                    })

                    # 🔥 Kill ffmpeg instantly
                    if process:
                        try:
                            process.kill()
                            await process.wait()
                        except Exception:
                            pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    
    await websocket.accept()
    active_tts_task = None

    try:
        while True:
            data = await websocket.receive_json()
            # The frontend will now send 'history' in the JSON [cite: 64]
            user_history = data.get("history", [])
            event = data.get("event")
            text = data.get("text")
            user_id = data.get("user_id")

            # 🛑 INTERRUPT HANDLING
            if event == "interrupt":
                print("🛑 Interrupt received")

                if active_tts_task and not active_tts_task.done():
                    active_tts_task.cancel()

                continue
          
            # ❌ Ignore empty messages
            if not text:
                continue

            
            if event == "user_message" or event == "boot":
                # Kill existing task so Alan doesn't talk over the new message
                if active_tts_task and not active_tts_task.done():
                    active_tts_task.cancel()
                    # Start process_and_stream as a task we can kill later
                active_tts_task = asyncio.create_task(
                    process_and_stream(data["text"], data["user_id"], websocket, user_history)
                )

            # 🎧 MAIN STREAMING TASK
            

            

    except WebSocketDisconnect:
        print("🔴 Client disconnected")
        if active_tts_task:
            active_tts_task.cancel()




