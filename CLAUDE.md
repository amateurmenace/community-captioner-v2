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

### Latest Updates - January 1, 2026

#### Logo Redesign & Demo Enhancement
- [x] **Inline CC Icon** - CC now appears inline with "CAPTIONER" text using thin box styling
  - 17px font size (30% smaller than original 24px)
  - 1.5px border with 3px border-radius
  - Inter font at weight 900 for distinction
  - Consistent across all pages (landing, dashboard, session analysis)
- [x] **Demo Simulation Update** - Changed third example to show more realistic correction:
  - Original: "town administrated steven woo presented"
  - Corrected: "Town Administrator Stephen Wu presented"
  - Demonstrates spelling correction + capitalization + name correction
- [x] **Logo Typography** - Two-line "COMMUNITY CAPTIONER" in uppercase, 28px font size

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

#### 2. Session History & Management
**Why**: Users need to find and re-analyze past sessions
**Tasks**:
- Session list page with search/filter
- Thumbnail previews with key stats
- Date range filtering
- Delete/archive sessions
- Quick re-analysis from history

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

### Known Blockers
- **Whisper real-time latency** - 2-4s delay may be too slow for some live events
- **Browser Speech API reliability** - Occasional crashes/restarts needed
- **Mobile browser constraints** - Web Speech API limited on mobile browsers

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
