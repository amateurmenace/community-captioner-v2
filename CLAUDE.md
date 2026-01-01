# Community Captioner v2

## Project Overview

**Community Captioner v2** is a free, open-source live captioning system for community media organizations. It provides real-time speech-to-text with local AI-powered corrections for proper nouns (names, places, organizations).

**Primary User:** Brookline Interactive Group (BIG) - a community media organization in Brookline, MA that broadcasts town meetings, events, and local programming.

**Problem Solved:** Commercial captioning encoders cost $30K+. This provides a zero-cost alternative using browser APIs and optional local Whisper AI.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (index.html)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Web Speech  │  │   React     │  │  Settings/Controls  │  │
│  │    API      │  │    UI       │  │                     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼───────────────────┼──────────────┘
          │                │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                 Python Server (start-server.py)              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Whisper   │  │   Caption   │  │     Session         │  │
│  │   Engine    │  │   Engine    │  │     Manager         │  │
│  │ (optional)  │  │   (RAG)     │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
          │                │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                      File System                             │
│  context-data/     sessions/          engines/               │
│  (engine state)    (recordings)       (portable engines)     │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Captioning Modes
- **Web Speech API** (browser) - Real-time ~200ms, requires Chrome/Edge + internet
- **Whisper** (local) - Accurate, 2-4s latency, works offline, requires `faster-whisper`

### 2. Caption Engine (RAG)
Corrects ASR errors for local proper nouns:
- "Brooklyn" → "Brookline" (common ASR error)
- "bernard green" → "Bernard Greene" (capitalization + spelling)
- "selectboard" → "Select Board" (formatting)

**Data structure:**
```python
terms = {
    "bernard greene": {
        "term": "Bernard Greene",
        "category": "person",  # person|place|organization
        "aliases": ["bernard green", "bernie greene"],
        "source": "default"  # default|manual|document
    }
}
```

### 3. Session Manager
Records captioning sessions with:
- Timestamps for each caption segment
- Raw and corrected text
- All corrections applied
- Export to SRT, VTT, TXT, JSON

### 4. Portable Engines
Engines can be exported/imported as JSON files to share correction rules between organizations.

## File Structure

```
community-captioner/
├── start-server.py      # Main Python server (FastAPI-style with http.server)
├── index.html           # React SPA control panel
├── overlay.html         # OBS browser source overlay
├── cloud-server.py      # Optional cloud relay server
├── start.sh             # Mac/Linux launcher
├── CLAUDE.md            # This file
├── README.md            # User documentation
├── context-data/        # Engine state persistence
│   └── engine_state.json
├── sessions/            # Recorded session data
│   └── {session_id}.json
└── engines/             # Portable engine files
    └── brookline_ma.json
```

## API Endpoints

### Caption
- `GET /api/caption` - Get current caption state
- `POST /api/caption` - Send caption text + settings

### Engine
- `GET /api/engine/status` - Engine stats and recent corrections
- `GET /api/engine/terms` - List all terms
- `GET /api/engine/corrections` - Recent corrections log
- `GET /api/engine/test?text=...` - Test correction on text
- `GET /api/engine/export` - Download engine as JSON
- `POST /api/engine/enable` - Enable/disable engine
- `POST /api/engine/term` - Add term `{term, category, aliases}`
- `POST /api/engine/term/remove` - Remove term
- `POST /api/engine/import` - Upload engine JSON
- `POST /api/engine/defaults` - Load Brookline defaults

### Whisper
- `GET /api/whisper/status` - Availability and model status
- `GET /api/whisper/devices` - List audio input devices
- `POST /api/whisper/load` - Load model `{model: "base"}`
- `POST /api/whisper/start` - Start listening `{device_id}`
- `POST /api/whisper/stop` - Stop listening

### Session
- `GET /api/session/status` - Recording status and stats
- `GET /api/session/captions` - All captions in session
- `GET /api/session/summary` - Generate summary
- `GET /api/session/export/srt` - Export SRT
- `GET /api/session/export/vtt` - Export VTT
- `GET /api/session/export/txt` - Export transcript
- `GET /api/session/export/json` - Export full JSON
- `POST /api/session/start` - Start recording `{name}`
- `POST /api/session/stop` - Stop recording

## Key Technical Decisions

1. **No framework for server** - Using Python's built-in `http.server` for zero dependencies
2. **React via CDN** - No build step, single HTML file
3. **Whisper optional** - Core functionality works without it
4. **Browser-first** - Web Speech API handles most use cases
5. **File-based persistence** - JSON files, no database needed
6. **Portable engines** - JSON export/import for sharing

## Brookline-Specific Context

**Key People:**
- Bernard Greene - Select Board Chair
- Heather Hamilton - Select Board member
- John VanScoyoc - Town Administrator
- Melissa Goff - Deputy Town Administrator
- Todd Kirrane - DPW Commissioner

**Key Places:**
- Coolidge Corner - Main commercial district
- Brookline Village - Historic area
- Larz Anderson Park - Large park
- Town Hall - 333 Washington Street

**Key Organizations:**
- Select Board - Town executive body (5 members)
- Town Meeting - Legislative body (240+ members)
- BIG (Brookline Interactive Group) - Community media
- Advisory Committee - Finance oversight

## Current State (v3.0)

### Completed Features
- [x] Dual captioning modes (Web Speech / Whisper)
- [x] Caption Engine with term management
- [x] Real-time corrections dashboard
- [x] Session recording with timestamps
- [x] Export to SRT, VTT, TXT, JSON
- [x] Portable engines (download/upload)
- [x] OBS overlay integration
- [x] Basic summary generation

### Known Issues
- Summary generation is basic (just stats, no AI)
- No audio recording for post-session Whisper reprocessing
- No document ingestion UI (API exists)
- Cloud server not tested recently

## Planned Features (Priority Order)

### High Priority
1. **Audio Recording** - Record session audio for Whisper reprocessing
2. **Accuracy Pass** - Re-run Whisper on recorded audio for cleaner export
3. **AI Summaries** - Use LLM to generate meeting summaries
4. **Document Ingestion UI** - Upload agendas/documents to extract terms

### Medium Priority
5. **Speaker Diarization** - Identify different speakers
6. **Custom Vocabulary** - Feed Whisper domain-specific words
7. **Live Corrections UI** - Edit corrections in real-time
8. **Searchable History** - Search past sessions

### Lower Priority
9. **Multi-language Support** - Spanish, Portuguese for Brookline
10. **Remote Control** - Control from phone/tablet
11. **Auto-punctuation** - Better sentence detection
12. **Confidence Scores** - Show ASR confidence

## Development Notes

### Running Locally
```bash
python3 start-server.py
# Opens http://localhost:8080
```

### Testing Whisper
```bash
pip3 install faster-whisper sounddevice numpy
# Mac: brew install portaudio
# Linux: sudo apt install portaudio19-dev
```

### Testing Corrections
Say or type: "Welcome to the Brooklyn Select Board meeting with Chair Bernard Green"
Should correct to: "Welcome to the Brookline Select Board meeting with Chair Bernard Greene"

### Adding New Terms
```python
# Via API
POST /api/engine/term
{
    "term": "New Person Name",
    "category": "person",
    "aliases": ["new person", "n. person"]
}
```

## Code Style

- Python: Standard library preferred, type hints appreciated
- JavaScript: Modern ES6+, React hooks
- Keep it simple - this is for community use
- Comments for non-obvious logic
- Error handling with user-friendly messages

## Contact

- **Organization:** Brookline Interactive Group
- **Developer:** Stephen Walter (weirdmachine.org)
- **License:** CC BY-SA 4.0
