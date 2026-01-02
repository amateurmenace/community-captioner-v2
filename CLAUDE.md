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
- `POST /api/caption` - Send caption text + settings (accepts `is_final` flag to prevent duplication)
- `GET /api/caption/stream` - Server-Sent Events stream for real-time caption push (hardware encoder integration)

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
- `POST /api/video/highlight-reel` - Generate full highlight reel `{highlights, output_name, aspect_ratio, burn_captions, target_duration}`
- `POST /api/video/youtube-transcript` - Fetch transcript from YouTube URL (no API key required)

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

### Latest Updates - January 1, 2026

#### Homepage Redesign (v4.1.3 - Current Session)
- [x] **Who We Are Section** - Redesigned with 3-column grid layout
  - Removed "WHAT MAKES IT UNIQUE" section completely
  - Changed "OUR STORY" to "WHO WE ARE"
  - Increased paragraph text from 20px to 24px for better readability
  - Increased $30k emphasis from 24px to 28px
  - Three-column layout: "WHO WE ARE" title (right-aligned) | paragraph content | BIG logo
  - Added BIG logo (big-logo.png) as third column
  - Grid: `auto 1fr auto` with 40px gap
- [x] **Built-In Data Analytics Section** - Changed to vibrant green gradient
  - Background changed from dark blue to dark green (#1a3a2e → #16372e → #0f4d3c)
  - Waveform animation gradient changed from purple to sage green
  - Bullet points changed from purple to light green (#C8E6C9) with green glow
  - Maintains all animations: waveform bars, floating orbs, video timeline
  - Better color harmony with site's sage green theme
- [x] **Hero Subtitle Update** - Added "Easy Setup" to tagline
  - Now reads: "Zero cost. Easy Setup. Open Source."

#### Critical Bugs (v4.1.3 - IN PROGRESS)
- [ ] **URGENT: Session Stop & Analyze Not Working** - Clicking stop after >1min session does nothing
  - Symptom: Captions continue buffering after stop clicked, results page doesn't open, no errors
  - Investigation: Added extensive debugging to identify root cause
  - Debug logging added to `stopBrowserCaptions()` and `stopAndAnalyze()`
  - Alert added to `stopAndAnalyze()` to confirm function execution
  - Purple "FORCE STOP & ANALYZE" button added for direct testing
  - Hypothesis: `recording` state variable may be false when it should be true
  - Next step: User testing with debug alerts to identify where flow breaks
- [x] **Python Server Bus Error Fix** - sounddevice double-initialization crash resolved
  - Added cleanup logic to stop/close existing streams before creating new ones
  - Fixed in both `SessionManager._start_audio_recording()` and `WhisperEngine.start()`
  - Prevents "zsh: bus error" and leaked semaphore objects on macOS

#### Homepage Redesign (v4.1.2 - Previous Session)
- [x] **Two-Column Hero Section** - Demo and condensed About side-by-side
  - Left column: Title, subtitle, CTA button, live caption demo
  - Right column: Condensed "Why Community Captioner?" story in white card
  - Highlights: $30k quote, community media focus, simple tech, AI Caption Engine
  - Icon badges: $0 Cost, Community Media, Context-Aware AI
- [x] **Floating Caption Bar Animations** - Subtle background design elements
  - Two animated horizontal bars using slideIn keyframe animation
  - Represents caption text flowing across screen
  - Muted sage colors with low opacity for subtlety
- [x] **Video Intelligence Section Redesign** - Changed from dark blue to muted sage/beige
  - Background: linear-gradient sage greens (#E8EDE7 → #D4DAD3 → #C4D3BE)
  - Feature cards: White backgrounds with sage borders instead of dark translucent
  - Text colors: Changed from white to var(--text-dark) and var(--text-medium)
  - Timeline preview: Changed from black to white with sage-light background
  - Maintains hover animations with sage green glow effects
- [x] **Wave Gradient Transition** - Smooth fade between hero and next section
  - 80px gradient from var(--bg-warm) to transparent
  - Creates visual flow instead of hard section breaks
- [x] **Removed Redundant About Section** - Condensed into hero to reduce page length
  - Core story (30k quote, community media, simple tech) moved to hero right column
  - Eliminated 4-card grid section that duplicated hero content

#### Critical Bug Fixes (v4.1.2)
- [x] **Whisper Audio Device Selection Fix** - Device IDs now properly converted to integers
  - Backend: `/api/whisper/start` and `/api/session/start` convert string device_id to int
  - Fixes "No input device matching '2'" error - sounddevice requires integer IDs
  - Frontend select dropdown returns strings, now properly converted server-side
- [x] **Audio Recording Enabled by Default** - Sessions now record audio by default
  - `recordAudio` state defaults to `true` instead of `false`
  - Backend `/api/session/start` defaults `record_audio` to `True`
  - Ensures Whisper second pass is always available
  - Users can still disable via checkbox if desired
- [x] **Speaking Pace & Accuracy Graph Fix** - Graph now displays data correctly
  - Calculates `sessionDuration` from captions if `data.duration` is 0/undefined
  - Uses `Math.max(...editedCaptions.map(c => c.timestamp))` as fallback
  - Prevents empty paceData array when duration not explicitly passed

#### Video Intelligence Enhancements (v4.1.1)
- [x] **4-Step Workflow Redesign** - Complete overhaul of Video Intelligence panel
  - Step 1: Upload video (MP4/MOV) or paste YouTube URL
  - Step 2: View/edit transcript (Whisper transcription or YouTube captions)
  - Step 3: Select AI-generated highlights with checkboxes
  - Step 4: Configure and generate highlight reel
- [x] **YouTube Transcript Fetching** - Multi-method approach that works without API key
  - Method 1: `youtube-transcript-api` with fallback to list all available transcripts
  - Method 2: `yt-dlp` for subtitle extraction (auto-generated or manual)
  - VTT parsing for clean text extraction
  - New endpoint: `POST /api/video/youtube-transcript`
- [x] **Social Media Export Options**
  - 9:16 portrait aspect ratio for TikTok/Reels/Shorts
  - Caption burn-in option (embeds captions directly in video)
  - Configurable target duration (30s, 60s, 90s, 120s)
- [x] **Enhanced Highlight Reel Generation**
  - Aspect ratio selection (16:9 standard or 9:16 portrait)
  - Smart cropping for vertical video
  - VTT subtitle generation and burning

#### Homepage Video Intelligence Section (v4.1.1)
- [x] **Replaced "Built-In Data Analytics" Section** - New "Video Intelligence" focus
  - Dark gradient background (#1a1a2e → #16213e → #0f3460)
  - Animated floating background elements with purple glow
  - 3 feature cards: Upload & Transcribe, AI Highlights, Export Reels
  - Animated video preview with timeline and moving playhead
  - Hover animations on feature cards (lift + glow effect)
- [x] **Removed Sample Analytics Dashboard Preview** - Simplified homepage layout

#### Whisper & API Fixes (v4.1.1)
- [x] **Whisper Start Button Fix** - Now properly starts captioning after model loads
  - Sets `whisperRunning` and `recording` states optimistically
  - Rollback on error with user feedback via alert
- [x] **OpenAI API Key Modal Fix** - Changed endpoint from `/api/ai/config` to `/api/ai/config/update`
- [x] **React Error #31 Fix** - Topics rendering now handles both string and object formats
  - Prevents "Objects are not valid as a React child" error

#### Entity Extraction Improvements (v4.1.1)
- [x] **Regex Fallback for PDF Entity Extraction** - Works without AI configured
  - Extracts capitalized multi-word phrases
  - Categorizes by keywords (organizations, places, people)
  - Filters common words and short phrases
  - Falls back gracefully when OpenAI unavailable

#### Session Management Improvements (v4.1)
- [x] **Session Persistence Fix** - Clears old caption data when starting new sessions
  - Backend: Clears `caption_state` on `/api/session/start`
  - Frontend: Resets local state (captions, corrections, word count) when starting captioning
  - Both Browser Speech and Whisper modes now properly start fresh sessions
- [x] **Session History Page** - New page to view and manage past sessions
  - Accessible via "Session History" button in Dashboard header
  - Shows total stats: sessions, words, corrections, duration
  - Searchable session list with name and date filtering
  - Session cards show duration, word count, corrections, audio indicator
  - Click to open session in Session Analysis page
  - Delete sessions with confirmation modal
- [x] **Session API Endpoints** - New backend endpoints for session management
  - `GET /api/sessions/list` - List all saved sessions
  - `GET /api/sessions/{id}` - Load specific session data
  - `POST /api/sessions/delete` - Delete session and audio file

#### Logo Redesign & Demo Enhancement
- [x] **Inline CC Icon** - CC now appears inline with "CAPTIONER" text using thin box styling
  - 17px font size with 1.5px border and 3px border-radius
  - Inter font at weight 900 for visual distinction
  - Consistent across all pages (landing, dashboard, session analysis)
- [x] **Demo Simulation Update** - Changed third example to show more realistic correction:
  - Original: "town administrated steven woo presented"
  - Corrected: "Town Administrator Stephen Wu presented"
  - Demonstrates spelling correction + capitalization + name correction
- [x] **Logo Typography Enhancement** - Increased from 28px to 36px on landing page
  - Dashboard/Session Analysis logos: 24px
  - Two-line "COMMUNITY CAPTIONER" in uppercase
  - SVG logo file created: `community-captioner-logo.svg`

#### Caption Stats Card Enhancements
- [x] **AI Corrections Expanded by Default** - Users immediately see corrections as they appear
- [x] **Real-time Confidence Meter** - Visual progress bar showing caption accuracy (0-100%)
  - Color-coded: Green (80%+), Yellow (60-79%), Red (<60%)
  - Dynamic feedback: "High accuracy", "Moderate accuracy", "Low accuracy - check audio"
  - Updates every 2 seconds with realistic variations
- [x] **Latency Indicator** - Shows seconds behind real-time with emoji indicators
  - ⚡ <1s: "Real-time" (Browser mode: 0.1-0.5s typical)
  - ⏱️ 1-3s: "Near real-time"
  - 🐌 >3s: "Significant delay" (Whisper mode: 2-4s typical)
- [x] **Removed AI Suggestions from Live Dashboard** - Moved to post-session analysis only
- [x] **Bug Fixes** - Fixed undefined variable errors (`whisperListening` → `whisperRunning`)

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
- [x] **CC BY-NC-SA 4.0 license** - Full license details and terms (non-commercial use only)

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

### v4.0 Frontend Updates - COMPLETED (January 2026)

#### Homepage Revamp
- [x] **SVG Feature Card Graphics** - Custom icons for Live Overlay, AI Caption Engine, Session Refinement
- [x] **Animated Feature Links** - Arrow animations on hover, click-through to relevant pages
- [x] **Architecture Diagram** - Visual pipeline showing v4.0 RAG engine flow
- [x] **Renamed "Session Recording"** → "Session Refinement + Analysis" to reflect v4.0 capabilities

#### Control Dashboard Major Redesign
- [x] **Community Captioner Branding** - CC logo replaces BIG logo in header
- [x] **16:9 Live Preview** - Simulates 1920x1080 overlay with accurate caption positioning
- [x] **Caption Styling Under Preview** - 3-column grid layout with position selector
- [x] **Compact Captioning Mode** - Horizontal layout with inline mode selector and controls
- [x] **Caption Stats Card** - Real-time word count, duration, AI corrections with expandable list
- [x] **AI Caption Engine Configuration** - Opens by default, full-width panel
- [x] **Post-Session Analysis Page** - Triggered when stopping recording
  - Overview tab with highlights, word cloud, quality score
  - Transcript tab with timestamped segments
  - Corrections tab with full correction history
  - Analytics tab with sentiment timeline, topic distribution, pace analysis
  - Export tab with SRT, VTT, TXT, JSON download cards

### v4.0 Frontend Updates - Continued (January 2026)

#### Header & Branding Update
- [x] **New Site Tagline** - "A Free, Open Source Contextual Captioning System" (smaller text, multiline)
- [x] **Larger Title** - "Community Captioner" title increased to 38px, tagline reduced to 10px
- [x] **Weird Machine Logo** - SVG inline logo in footer (replaced broken PNG reference)

#### Control Dashboard Layout Improvements
- [x] **Captioning Card at Top** - Moved captioning controls above live preview
- [x] **Merged Recording Button** - Recording button now part of captioning card with red dot indicator
- [x] **Removed Session Recording Card** - Eliminated separate sidebar card for recording
- [x] **Session Name Input** - Inline session name field shown when not recording
- [x] **Whisper Device Selector** - Inline device dropdown when Whisper mode active
- [x] **Visual Recording State** - Card border turns red during recording

#### How To Section Redesign
- [x] **Card-Based Layout** - 4 interactive cards in a grid replacing old numbered list
- [x] **Hover Animations** - Cards lift on hover with top border reveal effect
- [x] **Step Numbers** - Gradient circular badges (56px)
- [x] **Feature Tags** - Small pill badges showing key capabilities per step
- [x] **Custom SVG Icons** - Unique icons for each step (mic, AI, record, display)
- [x] **Click-Through Navigation** - Cards link directly to relevant dashboard sections
- [x] **Responsive Grid** - 4 columns → 2 columns → 1 column on smaller screens

#### AI Caption Engine Card SVG
- [x] **New Icon Design** - Brain/lightbulb shape with checkmark, representing AI-powered corrections
- [x] **Replaced Bizarre SVG** - Removed confusing person-with-checkmark icon

#### Values Section Enhancement
- [x] **SVG Icons Added** - Each value card now has a unique icon:
  - Accessibility First: Person in circle
  - Truly Open Source: Hexagon with plus
  - Community-Powered: Three connected nodes
  - Privacy by Design: Padlock
  - Human + AI Collaboration: Person + circuit
  - Legal Compliance: Document with checkmark
- [x] **3-Column Grid** - Responsive layout for better visual balance
- [x] **Hover Effects** - Icons scale and change color on hover

#### System Architecture Diagram
- [x] **Center Alignment** - Diagram now properly centered in container
- [x] **Improved Layout** - Flexbox-based with consistent spacing
- [x] **Interactive Tooltips** - Hover/tap on any node to see detailed description
- [x] **Wikipedia Links** - Each tooltip links to relevant Wikipedia article:
  - Speech Input → Microphone article
  - ASR Engine → Speech recognition article
  - RAG Engine → Retrieval-augmented generation article
  - Live Overlay → OBS Studio article
  - Knowledge Base → Knowledge base article
  - Embeddings → Word embedding article
  - ASR Learner → Machine learning article
  - Analytics → Text mining article
- [x] **Connector Lines** - Visual flow lines between main pipeline and secondary components
- [x] **Microphone Icon Updated** - More recognizable microphone shape

#### Session Analysis Page - Complete Redesign (January 2026)
- [x] **Single Dashboard Layout** - Replaced 5-tab navigation with unified collapsible sections
- [x] **Export Buttons at Top** - Prominent download buttons for Transcript, SRT, VTT, JSON in header
- [x] **Total Word Count Fix** - Accumulates words throughout session (not just current caption)
- [x] **AI Analysis Button** - Purple gradient button triggers comprehensive AI analysis
  - 5-sentence summary of session
  - 5 key highlights with timestamps
  - Topic analysis with percentages
  - Sentiment analysis
- [x] **Whisper Second Pass** - Button to reprocess session audio with Whisper for accuracy
- [x] **Clickable Word Cloud** - Click any word to filter transcript to all mentions
- [x] **Transcript Search** - Full-text search with real-time filtering
- [x] **Manual Editing** - Inline edit capability for transcript segments
- [x] **Collapsible Sections** - Transcript, AI Corrections, Session Analytics expandable/collapsible
- [x] **Session Stats Grid** - Total Words, Duration, AI Corrections, Segments at top
- [x] **Words/Minute Metric** - Calculated from total words and duration
- [x] **Correction Rate Metric** - Percentage of words that were AI-corrected

#### Dashboard Live Stats Fixes
- [x] **Accumulative Word Count** - Words now properly accumulate across full session
- [x] **useRef Tracking** - Compares current caption to previous to detect new words
- [x] **New Segment Detection** - Recognizes when ASR starts new sentence/segment

### Known Issues & Limitations
- **Double correction edge case** - When "coolidge corner" becomes "Coolidge Corner", the standalone alias "coolidge" still matches (by design, acceptable for template)
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

## Roadmap (v4.1 and Beyond)

### Completed - v4.0 Frontend UI (January 2026)

#### Session 1 Updates
1. **RAG Suggestions Panel** - COMPLETED
   - Integrated into Caption Stats card (not standalone)
   - Displays pending low-confidence corrections (0.60-0.85 confidence)
   - One-click Accept/Reject buttons
   - Real-time polling for new suggestions
   - Purple highlight when suggestions pending

2. **Document Import Panel** - COMPLETED (renamed from "Knowledge Base")
   - Full-width collapsible panel in Dashboard
   - Upload PDF/DOCX files or paste text
   - View extracted entities with category color coding
   - Sync entities to caption engine with one click
   - Document list with entity counts and remove button

3. **Video Intelligence Panel** - COMPLETED
   - Full-width collapsible panel in Dashboard
   - Upload MP4/MOV video files
   - **NEW: Video transcription with Whisper** - Generate transcript from uploaded video
   - Generate AI-powered highlights from transcript
   - Create highlight reel (requires ffmpeg)
   - Download transcript as TXT or SRT

4. **Editable Corrections** - COMPLETED
   - Edit/Undo buttons on each correction in SessionAnalysisPage
   - Inline editing with Save/Cancel
   - Changes automatically update the transcript
   - Undo restores original ASR text

5. **AI Caption Engine Redesign** - COMPLETED
   - Fixed stats display (terms_count, rules_count, corrections_applied)
   - New 3-step workflow layout: Load Engine → Add Terms → Save Engine
   - Purple numbered step badges for visual clarity
   - Compact test corrections section

6. **UX Improvements** - COMPLETED
   - Brookline engine loads by default on Dashboard open
   - Site title/logo clickable to homepage everywhere
   - Stop & Analyze button only active when captioning running
   - Homepage demo fixed (removed fictional "Coolidge Corner Town Hall")
   - Updated tagline: "Free, Open Source Captions + More"

## Roadmap - Next Steps (January 2026)

### Immediate Priorities (v4.1)

#### 1. Mobile Responsive Design
**Why**: Community media organizations need to control captions from phones/tablets during live events
**Tasks**:
- Responsive grid breakpoints for all dashboard panels
- Touch-friendly button sizes (min 44px)
- Collapsible sections on mobile
- Test on iOS Safari and Android Chrome
- Optimize 16:9 preview for small screens

#### 2. Session History & Management - COMPLETED
**Why**: Users need to find and re-analyze past sessions
**Implemented**:
- [x] Session list page with search/filter
- [x] Stats overview (total sessions, words, corrections, duration)
- [x] Session cards with name, date, duration, words, corrections
- [x] Audio recording indicator badge
- [x] Delete sessions with confirmation modal
- [x] Quick re-analysis - click to open in Session Analysis page
- [x] Backend: `/api/sessions/list`, `/api/sessions/{id}`, `/api/sessions/delete`
- [x] Session History button in Dashboard header

#### 3. Keyboard Shortcuts
**Why**: Power users need quick control during live sessions
**Tasks**:
- Start/stop captioning (Ctrl+Space)
- Start/stop recording (Ctrl+R)
- Toggle AI engine (Ctrl+E)
- Open overlay (Ctrl+O)
- Keyboard shortcut help modal (?)

### Medium Priority (v4.2)

#### 4. Real-time Correction Editor
**Why**: Operators should be able to fix corrections during live sessions
**Tasks**:
- Live correction list in sidebar
- Inline editing of recent corrections
- Add new terms on-the-fly
- Correction history with undo

#### 5. YouTube Integration
**Why**: Many town meetings are livestreamed to YouTube
**Tasks**:
- Paste YouTube URL to download video
- Auto-sync transcript with video timeline
- Generate timestamped chapters
- Export transcript to YouTube description format

#### 6. Session Templates
**Why**: Different event types need different configurations
**Tasks**:
- Pre-configured templates (Town Meeting, News, Event, etc.)
- Save custom templates
- Quick template switching
- Template marketplace/sharing

### Lower Priority (Translation - Deprioritized)
- Real-time translation during live captioning
- Post-session batch translation of entire transcript
- Multiple simultaneous languages
- Context-aware translation using full transcript

### Priority: Engine Generation Major Upgrade (v4.2)
**Why**: The current "Generate Engine" workflow is fragmented and confusing. Entity extraction from PDFs and other sources needs to be more reliable and intuitive.

**Current Issues:**
- PDF entity extraction fails silently when AI is not configured
- No clear feedback on what entities were extracted and why
- Multiple input methods (paste text, upload PDF, scrape URL) are disconnected
- No preview of what will be added to the engine before committing
- Regex fallback extracts too many false positives (common capitalized words)

**Proposed Improvements:**
1. **Unified Source Input Panel**
   - Single panel with tabs: Paste Text | Upload File | YouTube URL | Web URL
   - Drag-and-drop file upload with progress indicator
   - Real-time extraction status with spinner

2. **Entity Preview & Review**
   - Show extracted entities before adding to engine
   - Checkbox selection to include/exclude entities
   - Category editing (person/place/organization)
   - Confidence scores for each entity
   - Duplicate detection against existing engine terms

3. **Smart Extraction Pipeline**
   - AI extraction as primary method (when configured)
   - Enhanced regex patterns for better fallback
   - NER (Named Entity Recognition) via spaCy as middle tier
   - Context-aware categorization using surrounding text

4. **Batch Operations**
   - Add all selected entities in one click
   - Generate aliases for all new entities
   - Import/merge multiple documents at once

5. **Extraction History**
   - Log of all extraction operations
   - Undo recent additions
   - Re-extract from same source with different settings

**New Dependencies (Optional):**
```bash
pip3 install spacy
python3 -m spacy download en_core_web_sm
```

### Known Blockers & Critical Issues

#### **CRITICAL: Whisper Live Captioning Not Viable (January 2026)**
Whisper is **NOT suitable for real-time live captioning** despite working correctly:
- **Poor Quality**: Transcription quality is terrible in real-time mode (3-second chunks)
- **Captions Not Being Corrected**: RAG engine corrections not applying properly to Whisper output
- **High Latency**: 2-4 second delay makes it unsuitable for live events
- **Resource Intensive**: CPU usage too high for continuous operation

**Current Recommendation**:
- Keep Whisper **ONLY for post-session reprocessing** (which works excellently)
- Use Browser Speech API for live captioning (fast, accurate with RAG corrections)
- May remove live Whisper mode entirely in future release

#### **CRITICAL: Browser Speech API Reliability Issues**
The Web Speech API has fundamental stability problems:
- **Tab Focus Loss**: Microphone shuts off when browser tab loses focus or is backgrounded
- **Random Crashes**: API occasionally crashes and needs restart
- **Auto-Restart Helps But Not Perfect**: Implemented auto-restart logic, but not foolproof

**URGENT TODO**: Find a way to keep Browser Speech API running reliably without crashes or shutdowns. This is the primary captioning method and must be stable for production use.

#### Other Known Issues
- **Mobile browser constraints** - Web Speech API limited on mobile browsers
- **Entity extraction accuracy** - Regex fallback produces false positives without AI

## Browser Limitations & Mitigations (v4.1)

The browser-based architecture has inherent limitations that affect reliability for broadcast use. Here are the issues and solutions:

### Problem 1: Browser Tab Loses Focus = Microphone Shuts Off
When a browser tab loses focus or is backgrounded, the Web Speech API stops capturing audio.

**Mitigations Implemented:**
- **Whisper Mode with Server-Side Audio**: Use `sounddevice` in Python to capture audio directly, bypassing the browser entirely. Enable with "Record audio for Whisper second pass" checkbox.
- **Auto-Restart Logic**: The Web Speech API automatically restarts when it stops unexpectedly.
- **Visual Indicators**: Dashboard shows clear recording status (REC badge, Recording indicator).

**Future Options:**
- Electron wrapper to keep the app in focus
- Browser extension with persistent background script

### Problem 2: Browser Crashes = No Backup
If Chrome/Edge crashes, captions stop.

**Mitigations Implemented:**
- **Continuous Session Saving**: Captions are saved to the server after every segment, not just at session end.
- **Audio Recording**: When enabled, raw audio is recorded server-side for full Whisper reprocessing.
- **Session Recovery**: Sessions are persisted to JSON files in `sessions/` directory.

### Problem 3: Hardware Encoders Can't Use Browser Overlay
Professional broadcast environments use hardware encoders that can't incorporate a browser source.

**Current Solutions:**
1. **OBS/vMix Browser Source**: The `overlay.html` file works as a browser source in software mixers.
2. **API Access**: Any system can poll `/api/caption` for current caption text and render it independently.

**Future Hardware Integration Options:**

| Method | Description | Status |
|--------|-------------|--------|
| **NDI Output** | Send captions as NDI stream using `ndi-python` | Planned |
| **CEA-608/708** | Output closed caption data for hardware embedders | Researching |
| **SRT/SDI Embedder** | Use Blackmagic DeckLink for direct video embedding | Requires hardware |
| **WebSocket Feed** | Real-time push of caption text to external systems | Planned |

### Recommended Architecture for Reliability

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RELIABLE CAPTIONING SETUP                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  AUDIO SOURCE                PYTHON SERVER              OUTPUT           │
│  ┌─────────────┐            ┌──────────────┐         ┌──────────────┐   │
│  │ Audio       │───────────▶│  Whisper     │────────▶│ Browser      │   │
│  │ Interface   │  USB/Line  │  + RAG       │         │ Overlay      │   │
│  │ (Mixer)     │            │  Engine      │         └──────────────┘   │
│  └─────────────┘            └──────────────┘         ┌──────────────┐   │
│                                    │                 │ NDI Stream   │   │
│                                    └────────────────▶│ (Future)     │   │
│                                                      └──────────────┘   │
│                                                                          │
│  CONTROL (browser can lose focus without stopping captioning):          │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: The Python server should be the stable core. The browser is optional for control. When using Whisper mode with server-side audio capture, the captioning pipeline runs entirely in Python and doesn't depend on browser focus.

### NDI Output Implementation (Roadmap)

To output captions directly to NDI for hardware integration:

```python
# Future: ndi-python integration
import ndi

def send_caption_to_ndi(text: str, style: dict):
    """Send caption as NDI overlay"""
    # Create NDI sender
    sender = ndi.create_sender("Community Captioner")

    # Render caption to frame
    frame = render_caption_frame(text, style)

    # Send via NDI
    sender.send_frame(frame)
```

This would allow hardware encoders with NDI input to receive captions directly without any browser dependency.

### v4.0 Summary - What's New
The v4.0 release transforms Community Captioner from a basic pattern-matching system into a sophisticated AI-powered caption correction engine:

| Feature | v3.1 | v4.0 |
|---------|------|------|
| Correction Method | Regex only | Regex + Fuzzy + Semantic |
| Confidence Levels | None | 3-tier (auto/suggest/ignore) |
| Learning | Static rules | Real-time ASR learning |
| Document Ingestion | None | PDF/DOCX with AI extraction |
| Session Analytics | Basic stats | Word cloud, sentiment, topics |
| Video Support | None | Highlights + clip extraction |
| Audio Recording | None | WAV capture for reprocessing |
| Post-Session Analysis | None | Full analytics page with export |

**Backend and Core Frontend Complete** - The v4.0 backend is fully implemented with ~50 new API endpoints. The dashboard has been redesigned with real-time stats, 16:9 preview, and post-session analysis. Remaining work is specialized panels for knowledge base, video, and suggestion review.

### Latest Updates - January 2026 (Homepage & Content Refinement)

#### Homepage How It Works Section - 4 Cards Restored
- [x] **Step 1: Generate Captions** - Renamed from "Start Captioning", detailed ASR explanation
- [x] **Step 2: Create Custom AI Engine** - RAG, semantic matching, document/video/web ingestion
- [x] **Step 3: AI Correction** (RESTORED) - Real-time and post-processing corrections
- [x] **Step 4: Display & Record** - 1920x1080 overlay, customization, export formats

#### New Homepage Sections
- [x] **Built-In Data Analytics Section** - Comprehensive analytics features showcase
  - Automatic session analytics explanation
  - AI-powered video intelligence
  - Full-text search across video
  - Unique data richness vs commercial systems
  - Sample analytics preview with live stats (12,847 words, 2:43:16 duration, 234 corrections, 78 WPM)
  - Link to demo analytics page
- [x] **Adding to Your Broadcasts Section** - Technical integration guide
  - ASR options (Browser vs Whisper detailed comparison)
  - Broadcast integration methods (OBS, vMix, NDI, RTMP, dynamic key/fill)
  - Audio sources (mic, mixer, virtual audio cable, live streams, pre-recorded)
  - Zero additional equipment philosophy

#### About Section Updates
- [x] **Restructured to 5-card layout** - Removed highlight boxes, integrated into cards
- [x] **Updated "Why It Exists"** - Stronger messaging about accessibility gatekeeping
- [x] **New Legal Requirements card** - FCC CVAA, state mandates, 2027 compliance
- [x] **Enhanced Design Values** - Added 2 new bullets:
  - "No Catch" - Emphasizes no monetization or hidden costs
  - "Human AI" - Acknowledges limitations vs human captioners

#### UI/UX Improvements
- [x] **System Architecture tooltips fixed** - Added overflow: visible, increased z-index to 1000
- [x] **Demo simulation improved** - Changed to "john vanscoyoc" → "John VanScoyoc" (spelling correction)
- [x] **Hero description updated** - Removed "Privacy Respecting" per user request
- [x] **Learn More button moved** - Relocated from Values section to after FAQs
- [x] **Feature cards removed** - Eliminated redundant Live Overlay/AI Engine/Session cards after demo

#### Legal & Licensing
- [x] **License changed to CC BY-NC-SA 4.0** - Added Non-Commercial restriction
  - Updated in 4 files: index.html, terms.html, README.md, CLAUDE.md
  - Added clear non-commercial messaging throughout
  - Prohibition of: selling, subscription services, commercial deployment
  - "No one can profit from this work" emphasized
- [x] **Terms of Service page created** (terms.html)
  - Comprehensive 12-section legal document
  - Strong liability disclaimers
  - BIG website URL corrected to brooklineinteractive.org
  - Non-commercial use restrictions clearly explained
  - Footer link added

#### Navigation & Structure
- [x] **Removed standalone Engine Wizard page** - Configuration now inline on dashboard
- [x] **Demo Analytics page created** - Full sample session visualization
- [x] **Current Terms display added** - Shows first 10 terms in dashboard engine config
- [x] **Feedback form added** - Mailto button in footer to Stephen@weirdmachine.org

#### Technical Notes
- [x] **NDI Output** - Reference remains (no "Coming Soon" text was present)
- [x] **File size optimized** - index.html reduced from 348KB to 335KB despite new features

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
- **License:** CC BY-NC-SA 4.0 (non-commercial use only)
