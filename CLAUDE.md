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

## Current State (v3.1)

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
- Summary generation is basic (stats only, no AI)
- No audio recording for post-session Whisper reprocessing
- Document upload feature has UI but needs server implementation
- Web scraping feature has UI but needs server implementation
- Cloud server not tested recently

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

## Planned Features (Priority Order)

### High Priority - Core Functionality Fixes
1. **Fix Browser Speech Recognition** - Debug why mic recognition not generating captions
2. **Fix Whisper Mode** - Debug why Whisper not working
3. **Fix Caption Engine** - Brooklyn should become Brookline in test function
4. **Audio Recording** - Record session audio for Whisper reprocessing
5. **Accuracy Pass** - Re-run Whisper on recorded audio for cleaner export

### High Priority - Major AI Enhancements (v4.0)

#### 1. Advanced RAG Caption Engine - **THE CORE FEATURE**

**What Makes This Different:**
Current caption engine is basic pattern matching. v4.0 will be a TRUE RAG system that revolutionizes caption correction.

**Real-Time Correction Architecture:**
- **Vector embeddings** for all terms using sentence-transformers
- **Semantic similarity matching** instead of regex patterns
- **Context-aware corrections** that understand "Brookline Select Board" vs "Brooklyn Nets"
- **Multi-word entity recognition** ("Bernard Greene" not "bernard" + "greene")
- **Fuzzy matching** with confidence thresholds (0.8+ = auto-correct, 0.5-0.8 = suggest)
- **Real-time learning** that observes ASR mistakes and builds correction rules automatically

**Post-Session Refinement:**
- **Second-pass correction** using full transcript context
- **Consistency enforcement** (if "Bernard Greene" appears 10x, fix the 2 "Bernard Green" errors)
- **Acronym expansion** based on first usage (DPW → Department of Public Works)
- **Cross-reference validation** against knowledge base
- **Bulk find-and-replace** across entire session with preview

**Knowledge Base Integration:**
- **Document ingestion**: Upload PDFs, meeting minutes, org charts
- **Web scraping**: Pull names from town website, LinkedIn, public records
- **Entity extraction**: GPT-4 extracts people, places, orgs with relationships
- **Automatic alias generation**: "Select Board" → ["selectboard", "board", "select bored"]
- **Contextual metadata**: Store job titles, locations, affiliations for disambiguation

**Why This Matters:**
This isn't just spell-check. It's an AI that learns your organization's unique language and corrects captions with near-human accuracy in BOTH real-time AND post-session.

#### 2. AI-Enhanced Real-Time Captions
- **GPT-4 post-processing** of ASR output for grammar, punctuation, capitalization
- **Real-time translation** to multiple languages
- **Sentiment analysis** and topic detection
- **Named entity recognition** to auto-populate caption engine

#### 3. Post-Session Analytics Dashboard
Comprehensive data visualization and analysis page:

**Transcript Analytics:**
- Word cloud of most frequent terms
- Sentiment timeline (positive/negative/neutral over time)
- Topic modeling and segmentation
- Speaker participation charts
- Pace/speed analysis (words per minute over time)

**Interactive Search:**
- Full-text search with highlighting
- Jump to timestamp in transcript
- Filter by speaker
- Filter by topic/sentiment
- Export search results

**Quality Metrics:**
- Correction frequency heatmap
- Confidence score distribution
- ASR accuracy estimation
- Engine effectiveness metrics

#### 4. AI-Powered Whisper Enhancement
- **Post-session reprocessing** using Whisper on recorded audio
- **Side-by-side comparison** of real-time vs. reprocessed
- **Automatic merge** of best results from both
- **Batch processing** for multiple sessions
- **Quality improvement metrics**

#### 5. Video Intelligence Integration

**Video Upload & Sync:**
- Upload video file or paste YouTube URL
- Auto-sync transcript timestamps with video
- Frame-accurate seeking via transcript search
- Visual preview of search results

**AI Video Highlights:**
- GPT-4 analyzes full transcript
- Identifies 5-10 key moments with justification
- Extracts exact quotes and timestamps
- Creates highlight clips using ffmpeg
- Auto-generates title cards
- Compiles into downloadable highlight reel (MP4)

**Advanced Video Features:**
- Chapter generation from topic segmentation
- Automatic B-roll suggestions
- Visual sentiment matching
- Quote overlays for social media clips
- Multi-clip compilation editor

#### 6. Smart Caption Translation
- **Real-time translation** during live captioning
- **Post-session batch translation** of entire transcript
- **Multiple simultaneous languages**
- **Context-aware translation** using full transcript
- **Cultural adaptation** not just literal translation

### Medium Priority
1. **Custom Vocabulary** - Feed Whisper domain-specific words
2. **Live Corrections UI** - Edit corrections in real-time during session
3. **Searchable History** - Search across all past sessions
4. **Remote Control** - Control from phone/tablet
5. **Collaborative Editing** - Multiple users edit transcript simultaneously
6. **API Access** - RESTful API for integration with other tools

### Lower Priority
1. **Auto-punctuation** - Better sentence detection
2. **Confidence Scores** - Show ASR confidence in UI
3. **Dark Mode** - Optional dark theme for operators
4. **Mobile App** - Native iOS/Android apps
5. **Cloud Sync** - Optional cloud backup/sync
6. **Monetization** - Optional paid tiers for cloud AI features

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
