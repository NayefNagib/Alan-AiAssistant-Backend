# Alan AI Assistant Backend

![Python](https://img.shields.io/badge/Python-AI_Backend-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Realtime_API-green)
![WebSocket](https://img.shields.io/badge/WebSocket-Streaming-orange)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20Gemini-red)

Real-time AI voice assistant backend featuring streaming speech synthesis, speech-to-text transcription, hybrid LLM orchestration, memory systems, and live conversational interaction.

---

## Features

- Real-time AI voice conversations
- WebSocket low-latency streaming architecture
- Speech-to-Text transcription using Whisper
- Text-to-Speech streaming with Edge TTS
- Hybrid LLM orchestration (Groq + Gemini)
- Live Google grounding/search integration
- Persistent conversational memory using SQLite
- Interruptible AI speech streaming
- Async processing pipeline
- Audio streaming via FFmpeg PCM conversion
- Intelligent response routing system
- Context-aware memory retrieval
- Response caching system

---

## AI Architecture

User Speech  
→ Whisper STT  
→ Memory Retrieval  
→ Groq LLM Reasoning  
→ Gemini Search Grounding (if needed)  
→ Response Generation  
→ Edge TTS Streaming  
→ Real-Time Audio Playback

---

## Tech Stack

### Backend
- Python
- FastAPI
- WebSockets
- Uvicorn
- SQLite

### AI / NLP
- Groq API
- Google Gemini
- Whisper Large v3
- Edge TTS

### Audio Processing
- FFmpeg
- PCM audio streaming

### Infrastructure
- AsyncIO
- HTTPX
- StreamingResponse

---

## Core Systems

### Conversational Memory
- Profile memory
- Long-term memory
- Short-term contextual memory
- Memory cleanup & size management

### Real-Time Streaming
- Interruptible TTS playback
- Live PCM audio streaming
- WebSocket event handling
- Async task cancellation

### Hybrid Intelligence Routing
- Groq for conversational reasoning
- Gemini for live grounded search
- Automatic online-data detection

---

## API Endpoints

### REST API
- `POST /speak`
- `POST /upload-voice`

### WebSocket
- `/ws`

---

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
HF_TOKEN=your_key
```

---

## Engineering Notes

This project focuses on building a production-style AI assistant backend capable of low-latency conversational interaction, real-time streaming audio, persistent memory, and grounded AI responses.

---

## Disclaimer

This project is intended for educational and research purposes.