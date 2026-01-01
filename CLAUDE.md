# Community Captioner v4.0 - Advanced RAG Engine

## Project Overview

**Community Captioner v4.0** is a free, open-source live captioning system for community media organizations. It provides real-time speech-to-text with an **Advanced RAG Caption Engine** that uses semantic similarity matching, fuzzy matching, and real-time learning for near-human accuracy correction of proper nouns.

**Primary User:** Brookline Interactive Group (BIG) - a community media organization in Brookline, MA that broadcasts town meetings, events, and local programming.

**Problem Solved:** Commercial captioning encoders cost $30K+. This provides a zero-cost alternative using browser APIs, optional local Whisper AI, and advanced AI-powered corrections.

## Architecture (v4.0)

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (index.html)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Web Speech  │  │   React     │  │  Settings/Controls  │  │
│  │    API      │  │    UI       │  │  Analytics Dashboard│  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼───────────────────┼──────────────┘
          │                │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                 Python Server (start-server.py)              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Whisper   │  │  Advanced   │  │   Session Manager   │  │
│  │   Engine    │  │ RAG Engine  │  │  + Audio Recording  │  │
│  │ (optional)  │  │ (v4.0)      │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Knowledge  │  │    GPT      │  │     Video           │  │
│  │    Base     │  │  Processor  │  │  Intelligence       │  │
│  │ (PDF/DOCX)  │  │             │  │   (ffmpeg)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
          │                │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                      File System                             │
│  context-data/     sessions/          knowledge/             │
│  (engine state)    (recordings)       (documents)            │
│  embeddings/       audio/             videos/                │
│  (vector cache)    (audio files)      (clips/reels)          │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Captioning Modes
- **Web Speech API** (browser) - Real-time ~200ms, requires Chrome/Edge + internet
- **Whisper** (local) - Accurate, 2-4s latency, works offline, requires `faster-whisper`

### 2. Advanced RAG Caption Engine (v4.0) - THE CORE FEATURE
The v4.0 engine uses a hybrid approach combining:

**Real-Time Correction Methods:**
1. **Regex Pattern Matching** - Fast, exact matching for known terms
2. **Fuzzy Matching** - Levenshtein distance for typos (threshold: 0.80)
3. **Semantic Similarity** - Vector embeddings via sentence-transformers or OpenAI
4. **Learned Patterns** - Auto-learns from observed ASR mistakes

**Confidence Thresholds:**
- `>= 0.85` - Auto-correct immediately
- `0.60 - 0.85` - Add to suggestions for review
- `< 0.60` - Ignore

**Context-Aware Disambiguation:**
- "Brooklyn Select Board" → "Brookline Select Board" (local context)
- "Brooklyn Nets game" → unchanged (NYC sports context)

**Data structure:**
```python
terms = {
    "bernard greene": {
        "term": "Bernard Greene",
        "category": "person",  # person|place|organization
        "aliases": ["bernard green", "bernie greene"],
        "source": "default",  # default|manual|document|knowledge_base
        "context": "Brookline Select Board member",
        "metadata": {"title": "Chair", "organization": "Select Board"}
    }
}
```

### 3. Session Manager (v4.0)
Records captioning sessions with:
- Timestamps for each caption segment
- Raw and corrected text
- All corrections applied
- **Audio recording** for post-session Whisper reprocessing
- Export to SRT, VTT, TXT, JSON

### 4. Knowledge Base
- Upload PDFs, DOCX, or text documents
- AI-powered entity extraction (people, places, organizations)
- Semantic search across documents
- Auto-sync extracted entities to caption engine

### 5. GPT Post-Processor
- Real-time grammar and punctuation fixes
- Named entity recognition
- Sentence boundary detection

### 6. Session Analytics
- Word cloud generation
- Sentiment timeline analysis
- Topic modeling and segmentation
- Correction frequency heatmap
- Quality metrics dashboard

### 7. Video Intelligence
- Video upload (MP4/MOV)
- AI-powered highlight detection from transcript
- Clip extraction with ffmpeg
- Automated highlight reel generation

### 8. Portable Engines
Engines can be exported/imported as JSON files including:
- All terms with aliases and metadata
- Learned ASR patterns
- Custom correction rules

## File Structure

```
community-captioner/
├── start-server.py      # Main Python server (v4.0 with all AI features)
├── index.html           # React SPA control panel
├── overlay.html         # OBS browser source overlay
├── cloud-server.py      # Optional cloud relay server
├── start.sh             # Mac/Linux launcher
├── CLAUDE.md            # This file
├── README.md            # User documentation
├── context-data/        # Engine state persistence
│   ├── engine_state.json
│   ├── asr_patterns.json    # Learned ASR corrections
│   └── embeddings/          # Cached vector embeddings
├── sessions/            # Recorded session data
│   └── {session_id}.json
├── audio/               # Session audio recordings
│   └── {session_id}.wav
├── knowledge/           # Uploaded documents
│   ├── knowledge_index.json
│   └── {doc_id}_{filename}
├── videos/              # Video files and clips
│   └── clips/           # Generated highlight clips
├── analytics/           # Analytics data
└── engines/             # Portable engine files
    └── brookline_ma.json
```

## API Endpoints

### Caption
- `GET /api/caption` - Get current caption state
- `POST /api/caption` - Send caption text + settings

### Engine (Core)
- `GET /api/engine/status` - Engine stats, RAG status, recent corrections
- `GET /api/engine/terms` - List all terms with metadata
- `GET /api/engine/corrections` - Recent corrections log
- `GET /api/engine/test?text=...` - Test correction on text
- `GET /api/engine/export` - Download engine as JSON
- `POST /api/engine/enable` - Enable/disable engine
- `POST /api/engine/term` - Add term `{term, category, aliases, context, metadata}`
- `POST /api/engine/term/remove` - Remove term
- `POST /api/engine/import` - Upload engine JSON
- `POST /api/engine/defaults` - Load Brookline defaults

### RAG Engine (v4.0)
- `GET /api/engine/suggestions` - Pending low-confidence suggestions
- `GET /api/engine/learned` - Learned ASR patterns
- `GET /api/engine/embeddings/status` - Embeddings availability
- `POST /api/engine/rag/enable` - Enable/disable RAG features
- `POST /api/engine/suggestion/accept` - Accept a suggestion
- `POST /api/engine/suggestion/reject` - Reject a suggestion
- `POST /api/engine/learned/clear` - Clear learned patterns
- `POST /api/engine/aliases/generate` - Auto-generate aliases for term
- `POST /api/session/refine` - Post-session consistency analysis
- `POST /api/session/refine/apply` - Apply a refinement

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
- `GET /api/session/analytics` - Full analytics dashboard data
- `GET /api/session/analytics/sentiment` - Sentiment timeline
- `GET /api/session/export/srt` - Export SRT
- `GET /api/session/export/vtt` - Export VTT
- `GET /api/session/export/txt` - Export transcript
- `GET /api/session/export/json` - Export full JSON
- `POST /api/session/start` - Start recording `{name, record_audio, audio_device}`
- `POST /api/session/stop` - Stop recording
- `POST /api/session/reprocess` - Reprocess with Whisper
- `POST /api/session/reprocess/apply` - Apply reprocessed captions

### Knowledge Base
- `GET /api/knowledge/status` - Documents and entity counts
- `GET /api/knowledge/entities` - All extracted entities
- `GET /api/knowledge/search?q=...` - Semantic search
- `POST /api/knowledge/upload` - Upload document (base64 or text)
- `POST /api/knowledge/remove` - Remove document
- `POST /api/knowledge/sync` - Sync entities to caption engine
- `POST /api/knowledge/search` - Search with query

### GPT Post-Processor
- `GET /api/gpt/status` - Processor status
- `POST /api/gpt/enable` - Enable/disable processing
- `POST /api/gpt/process` - Process text
- `POST /api/gpt/extract-entities` - Extract entities from text
- `POST /api/gpt/clear` - Clear buffer

### Video Intelligence
- `GET /api/video/status` - Video and ffmpeg status
- `POST /api/video/upload` - Upload video (base64)
- `POST /api/video/highlights` - Generate highlight moments
- `POST /api/video/clip` - Extract single clip
- `POST /api/video/highlight-reel` - Generate full highlight reel

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

## Current State (v4.0)

### v4.0 Backend Implementation - COMPLETED

#### Advanced RAG Caption Engine - IMPLEMENTED
- [x] **Vector Embeddings** - sentence-transformers integration with fallback to OpenAI
- [x] **Semantic Similarity Matching** - cosine similarity for context-aware corrections
- [x] **Fuzzy Matching** - Levenshtein distance with 0.80 threshold
- [x] **Confidence Thresholds** - 0.85+ auto-correct, 0.60-0.85 suggest
- [x] **Real-time Learning** - ASRLearner class tracks and learns from corrections
- [x] **Context-Aware Disambiguation** - NYC vs Brookline context detection
- [x] **Automatic Alias Generation** - AI-powered and rule-based

#### Post-Session Refinement - IMPLEMENTED
- [x] **Consistency Enforcement** - Detects spelling variations across session
- [x] **Bulk Refinement API** - `/api/session/refine` and `/api/session/refine/apply`
- [x] **Quality Metrics** - Correction rate, proper noun analysis

#### Knowledge Base - IMPLEMENTED
- [x] **PDF/DOCX Ingestion** - Extracts text from uploaded documents
- [x] **Entity Extraction** - AI-powered extraction of people/places/orgs
- [x] **Semantic Search** - Vector-based search across documents
- [x] **Engine Sync** - One-click sync of entities to caption engine

#### GPT Post-Processor - IMPLEMENTED
- [x] **Real-time Enhancement** - Grammar, punctuation fixes
- [x] **Entity Recognition** - Extract entities from caption stream
- [x] **Buffer Context** - Uses recent context for better processing

#### Session Analytics - IMPLEMENTED
- [x] **Word Cloud Data** - Frequency analysis for visualization
- [x] **Sentiment Timeline** - AI-powered sentiment over time
- [x] **Topic Analysis** - Identifies main discussion topics
- [x] **Quality Metrics** - Corrections heatmap, pace analysis

#### Audio Recording - IMPLEMENTED
- [x] **Session Audio Capture** - WAV format recording
- [x] **Whisper Reprocessing** - Post-session accuracy pass
- [x] **Side-by-side Comparison** - Compare real-time vs reprocessed

#### Video Intelligence - IMPLEMENTED
- [x] **Video Upload** - MP4/MOV file handling
- [x] **AI Highlight Detection** - Identifies key moments from transcript
- [x] **Clip Extraction** - ffmpeg-based clip generation
- [x] **Highlight Reel** - Automated compilation of clips

### Completed Features - January 2026 Update (v3.1)

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
- [x] **Typing Animations** - Character-by-character typing effect on hero title and subtitle with blinking cursor

#### Internationalization
- [x] **Multilanguage Support** - 4 languages: English, Spanish (Español), Portuguese (Português), French (Français)
- [x] **Language Selector** - Dropdown in header for easy switching
- [x] **Translation System** - 100+ UI strings translated across all pages
- [x] **Automatic Re-rendering** - Components update instantly when language changes
- [x] **TypeWriter Animation** - Re-animates text when language switches for smooth UX

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
- [x] **Engine Upload** - Upload saved engine JSON directly from dashboard

#### AI Caption Engine - Interactive Dashboard (January 2026 Redesign)
- [x] **Dashboard layout** - Replaced 7-step linear wizard with flexible card-based dashboard
- [x] **Real-time stats** - Live display of total terms, corrections made, engine status
- [x] **Quick Start card** - Load Brookline template, upload engine, download engine
- [x] **Manual Term Entry** - Fine-tune engine with individual term additions
- [x] **AI Document Extraction** - Paste document text, AI extracts names (primary method)
- [x] **AI Web Scraping** - Enter URLs, AI scrapes and extracts terms (primary method)
- [x] **Test Engine** - Side-by-side before/after correction testing
- [x] **Current Terms display** - Shows first 10 terms with categories
- [x] **Work in any order** - No forced workflow, all tools accessible simultaneously
- [x] **Auto-refreshing** - Stats update every 2 seconds

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
- Frontend UI needs updates to use new v4.0 API endpoints
- Cloud server not tested recently
- Translation features deprioritized (leave to last)

### Technical Improvements
- index.html: 3,717 → 3,600 lines (wizard redesign simplified code)
- Engine Wizard: Reduced from 447 lines to 286 lines (-36% code reduction)
- All styling controls functional with live updates
- Improved correction algorithm with overlap detection
- Better state management with React hooks throughout
- Unique visual identities for About/FAQ/How-To sections
- Professional, AI-designed look removed in favor of bold, clean design
- Translation system with centralized dictionary and prop drilling
- TypeWriter component with startDelay and cursor animation support

### Design Philosophy - January 2026
- **No emojis** - Professional interface for government and media organizations
- **Bold typography** - Clear hierarchy with strong heading styles
- **Distinct sections** - Each major section has unique visual treatment
- **Warm color palette** - Beige backgrounds with sage green accents
- **Legal compliance** - Prominent messaging about captioning requirements

## Remaining Tasks (v4.0)

### High Priority - Frontend Updates
1. **Update Dashboard UI** - Add RAG controls (enable/disable, suggestions review)
2. **Analytics Dashboard View** - Display word cloud, sentiment, topics from `/api/session/analytics`
3. **Knowledge Base UI** - Document upload, entity viewer, sync button
4. **Video Panel** - Upload video, generate highlights, download reel
5. **Suggestions Panel** - Review and accept/reject low-confidence suggestions

### Medium Priority
1. **Live Corrections UI** - Edit corrections in real-time during session
2. **Searchable History** - Search across all past sessions
3. **Remote Control** - Control from phone/tablet
4. **YouTube Integration** - Paste URL to download and sync video

### Lower Priority (Translation - Deprioritized)
1. **Real-time translation** during live captioning
2. **Post-session batch translation** of entire transcript
3. **Multiple simultaneous languages**
4. **Context-aware translation** using full transcript

## Development Notes

### Running Locally
```bash
python3 start-server.py
# Opens http://localhost:8080
```

### Installing AI Features
```bash
# For Whisper mode:
pip3 install faster-whisper sounddevice numpy
# Mac: brew install portaudio
# Linux: sudo apt install portaudio19-dev

# For vector embeddings (optional, recommended):
pip3 install sentence-transformers

# For PDF ingestion (optional):
pip3 install PyPDF2  # or pdfplumber

# For DOCX ingestion (optional):
pip3 install python-docx

# For video processing:
brew install ffmpeg  # Mac
# or: sudo apt install ffmpeg  # Linux
```

### Testing Corrections
Say or type: "Welcome to the Brooklyn Select Board meeting with Chair Bernard Green"
Should correct to: "Welcome to the Brookline Select Board meeting with Chair Bernard Greene"

### Testing RAG Features
```bash
# Check embeddings status
curl http://localhost:8080/api/engine/embeddings/status

# Check learned patterns
curl http://localhost:8080/api/engine/learned

# Get suggestions
curl http://localhost:8080/api/engine/suggestions
```

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
