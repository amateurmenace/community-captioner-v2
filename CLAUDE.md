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

### Completed Features - January 2026 Update

#### Landing Page (Major Redesign - January 2026)
- [x] **Animated blob background** - 5 circular sage/purple globs floating with smooth animations
- [x] **Warm beige color scheme** - Professional #F5F3EE background replacing pure white
- [x] **Repositioned demo** - AI Correction Engine demo directly under "AI for Accessibility" headline
- [x] **Emoji-free design** - Clean, professional interface without decorative emojis
- [x] **Legal requirements section** - FCC CVAA and state captioning law information
- [x] **Modern About section** - Gradient cards with left borders, uppercase headings
- [x] **Numbered How-To section** - Circular step badges with background gradient
- [x] **Interactive FAQ section** - Q badges, hover animations, accordion-style
- [x] **Professional footer** - Partner logos (BIG, CC), links to community partners
- [x] **Live interactive demo** - Shows Caption Engine corrections in real-time with animation
- [x] **Three demo examples** - Brooklyn→Brookline, bernard green→Bernard Greene, coolidge corner corrections

#### Control Dashboard
- [x] **Dual captioning modes** (Web Speech / Whisper) - Fully functional
- [x] **Browser Speech Recognition** - Working with Web Speech API, auto-restart on error
- [x] **Whisper Integration** - Device selection, model loading (tiny/base/small/medium), start/stop
- [x] **Live caption styling controls** - Font size, line height, max width, font family, colors, opacity
- [x] **Real-time preview** - Shows captions with live styling updates
- [x] **Caption Engine toggle** - Enable/disable with visual feedback
- [x] **Session recording** - Start/stop with session naming
- [x] **Export formats** - SRT, VTT, TXT, JSON with session stats display
- [x] **Live Overlay access** - Button to open overlay + network URL display
- [x] **Sage green color scheme** - Consistent throughout, purple accent for special features only

#### AI Caption Engine Wizard
- [x] **7-step guided workflow** with visual progress indicator
- [x] **Step 1: Welcome** - Explains what the engine does and why it's unique
- [x] **Step 2: Choose template** - Empty, Brookline defaults, or import existing
- [x] **Step 3: Manual term entry** - Form to add terms with categories and aliases
- [x] **Step 4: Document upload** - UI placeholder for PDF/DOCX/TXT upload (server implementation pending)
- [x] **Step 5: Web scraping** - UI placeholder for URL-based term extraction (server implementation pending)
- [x] **Step 6: Test engine** - Real-time correction testing with examples
- [x] **Step 7: Save & download** - Export engine as JSON, return to dashboard

#### About Page
- [x] **Comprehensive documentation** - Project overview, architecture, use cases
- [x] **ASCII architecture diagram** - Visual system representation
- [x] **Feature comparison table** - Community Captioner vs Commercial Systems
- [x] **Installation instructions** - Basic and full (with Whisper)
- [x] **Six detailed use cases** - Town meetings, events, news, education, accessibility, archival
- [x] **Credits section** - Developer, powered by, special thanks
- [x] **CC BY-SA 4.0 license** - Full license details and terms

#### Live Overlay (overlay.html)
- [x] **Caption display** with customizable styling
- [x] **Line height support** - Added lineHeight setting
- [x] **Real-time updates** - Polls /api/caption every 100ms
- [x] **Settings inheritance** - Applies all styling from dashboard
- [x] **OBS/vMix compatible** - Browser source ready

#### Caption Engine (Backend)
- [x] **32 Brookline default terms** loaded
- [x] **47 correction rules** generated (including aliases)
- [x] **Pattern-based matching** with regex
- [x] **Position tracking** to prevent double corrections
- [x] **Sorted by pattern length** - Longest matches first
- [x] **Export/import** as portable JSON

### Known Issues & Limitations
- **Double correction edge case** - When "coolidge corner" becomes "Coolidge Corner", the standalone alias "coolidge" still matches (by design, acceptable for template)
- Summary generation is basic (stats only, no AI)
- No audio recording for post-session Whisper reprocessing
- Document upload feature has UI but needs server implementation
- Web scraping feature has UI but needs server implementation
- Cloud server not tested recently

### Technical Improvements
- index.html: 1,014 → 3,717 lines (major redesign + enhancements)
- All styling controls functional with live updates
- Improved correction algorithm with overlap detection
- Better state management with React hooks throughout
- Unique visual identities for About/FAQ/How-To sections
- Professional, AI-designed look removed in favor of bold, clean design

### Design Philosophy - January 2026
- **No emojis** - Professional interface for government and media organizations
- **Bold typography** - Clear hierarchy with strong heading styles
- **Distinct sections** - Each major section has unique visual treatment
- **Warm color palette** - Beige backgrounds with sage green accents
- **Legal compliance** - Prominent messaging about captioning requirements

## Planned Features (Priority Order)

### High Priority
1. **Engine Wizard Redesign** - Convert from 7-step wizard to interactive dashboard
2. **Engine Upload** - Allow users to upload saved engines to dashboard sessions
3. **Multilanguage Translation** - Live and post-session caption translation
4. **Typing Animations** - Character-by-character text animations on page load
5. **Audio Recording** - Record session audio for Whisper reprocessing
6. **Accuracy Pass** - Re-run Whisper on recorded audio for cleaner export

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
