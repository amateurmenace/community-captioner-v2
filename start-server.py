#!/usr/bin/env python3
"""
Community Captioner v2.0 - Full-Featured Captioning Server

FEATURES:
  - Dual captioning modes (Web Speech / Whisper)
  - Local Context Engine with real-time corrections
  - Session recording with timestamps
  - Post-session Whisper accuracy pass
  - Export to SRT, VTT, TXT, JSON
  - Summary & highlights generation
  - Portable Caption Engines (download/upload)

INSTALLATION:
  Basic: python3 start-server.py
  
  For Whisper mode:
    pip3 install faster-whisper sounddevice numpy
    
  For summaries (optional):
    pip3 install openai  # or use local LLM
"""

import http.server
import socketserver
import webbrowser
import os
import json
import threading
import time
import socket
import re
import uuid
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
from collections import deque

# OpenAI for AI features
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# CONFIGURATION
# =============================================================================

PORT = 8080
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)

DATA_DIR = SCRIPT_DIR / "context-data"
SESSIONS_DIR = SCRIPT_DIR / "sessions"
ENGINES_DIR = SCRIPT_DIR / "engines"

for d in [DATA_DIR, SESSIONS_DIR, ENGINES_DIR]:
    d.mkdir(exist_ok=True)

# AI Configuration
AI_CONFIG_FILE = DATA_DIR / "ai_config.json"

# Default AI configuration
ai_config = {
    "provider": "none",  # none|openai|ollama|lmstudio|custom
    "openai_api_key": os.getenv('OPENAI_API_KEY', ''),
    "openai_model": "gpt-4o-mini",
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "llama3.2",
    "lmstudio_base_url": "http://localhost:1234/v1",
    "lmstudio_model": "local-model",
    "custom_base_url": "",
    "custom_api_key": "",
    "custom_model": ""
}

# Load saved AI config
if AI_CONFIG_FILE.exists():
    try:
        with open(AI_CONFIG_FILE) as f:
            saved_config = json.load(f)
            ai_config.update(saved_config)
    except:
        pass

# Initialize AI client based on config
openai_client = None

def init_ai_client():
    """Initialize AI client based on current configuration"""
    global openai_client
    openai_client = None

    if ai_config["provider"] == "openai" and OPENAI_AVAILABLE:
        api_key = ai_config.get("openai_api_key", "")
        if api_key:
            try:
                openai_client = OpenAI(api_key=api_key)
                print(f"✓ OpenAI client initialized (cloud)")
            except Exception as e:
                print(f"✗ OpenAI client failed: {e}")

    elif ai_config["provider"] == "ollama" and OPENAI_AVAILABLE:
        try:
            openai_client = OpenAI(
                base_url=ai_config.get("ollama_base_url", "http://localhost:11434/v1"),
                api_key="ollama"  # Ollama doesn't require real key
            )
            print(f"✓ Ollama client initialized (local)")
        except Exception as e:
            print(f"✗ Ollama client failed: {e}")

    elif ai_config["provider"] == "lmstudio" and OPENAI_AVAILABLE:
        try:
            openai_client = OpenAI(
                base_url=ai_config.get("lmstudio_base_url", "http://localhost:1234/v1"),
                api_key="lmstudio"  # LM Studio doesn't require real key
            )
            print(f"✓ LM Studio client initialized (local)")
        except Exception as e:
            print(f"✗ LM Studio client failed: {e}")

    elif ai_config["provider"] == "custom" and OPENAI_AVAILABLE:
        base_url = ai_config.get("custom_base_url", "")
        api_key = ai_config.get("custom_api_key", "")
        if base_url:
            try:
                openai_client = OpenAI(base_url=base_url, api_key=api_key or "none")
                is_local = "localhost" in base_url or "127.0.0.1" in base_url
                print(f"✓ Custom AI client initialized ({'local' if is_local else 'cloud'})")
            except Exception as e:
                print(f"✗ Custom AI client failed: {e}")

init_ai_client()

def get_ai_model():
    """Get the model name based on provider"""
    if ai_config["provider"] == "openai":
        return ai_config.get("openai_model", "gpt-4o-mini")
    elif ai_config["provider"] == "ollama":
        return ai_config.get("ollama_model", "llama3.2")
    elif ai_config["provider"] == "lmstudio":
        return ai_config.get("lmstudio_model", "local-model")
    elif ai_config["provider"] == "custom":
        return ai_config.get("custom_model", "unknown")
    return "none"

def is_ai_local():
    """Check if current AI provider is local (not cloud)"""
    provider = ai_config["provider"]
    if provider in ["ollama", "lmstudio"]:
        return True
    if provider == "custom":
        base_url = ai_config.get("custom_base_url", "")
        return "localhost" in base_url or "127.0.0.1" in base_url
    return False

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================

WHISPER_AVAILABLE = False
AUDIO_AVAILABLE = False
NUMPY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    pass

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# CAPTION ENGINE - Portable correction rules
# =============================================================================

class CaptionEngine:
    """
    Portable correction engine that can be exported/imported.
    Contains terms, patterns, and correction rules for a specific context.
    """
    
    VERSION = "1.0"
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.enabled = False
        self.name = "Default Engine"
        self.description = ""
        self.terms = {}  # term_lower -> {term, category, aliases, source}
        self.custom_rules = []  # [(pattern, replacement, description)]
        self.correction_rules = []  # compiled rules
        self.stats = {"corrections_applied": 0, "captions_processed": 0}
        self.corrections_log = deque(maxlen=100)  # Recent corrections
        self._load_state()
    
    def _load_state(self):
        state_file = self.data_dir / "engine_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    self.enabled = data.get("enabled", False)
                    self.name = data.get("name", "Default Engine")
                    self.description = data.get("description", "")
                    self.terms = data.get("terms", {})
                    self.custom_rules = data.get("custom_rules", [])
                    self._rebuild_rules()
            except Exception as e:
                print(f"Warning: Could not load engine state: {e}")
    
    def _save_state(self):
        try:
            with open(self.data_dir / "engine_state.json", "w") as f:
                json.dump({
                    "enabled": self.enabled,
                    "name": self.name,
                    "description": self.description,
                    "terms": self.terms,
                    "custom_rules": self.custom_rules
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save engine state: {e}")
    
    def _rebuild_rules(self):
        """Build correction rules from terms and custom rules"""
        self.correction_rules = []

        # Add Brooklyn → Brookline rule (common ASR error)
        self.correction_rules.append({
            "pattern": r'\b[Bb]rooklyn\b(?!\s+(NY|New York|Bridge|Nets|Dodgers))',
            "replacement": "Brookline",
            "category": "place",
            "description": "ASR commonly mishears Brookline as Brooklyn",
            "pattern_length": len("brooklyn")
        })

        # Add rules from terms
        for key, info in self.terms.items():
            term = info["term"]
            category = info.get("category", "other")

            # Main term pattern
            self.correction_rules.append({
                "pattern": rf'\b{re.escape(key)}\b',
                "replacement": term,
                "category": category,
                "description": f"Correct capitalization for {term}",
                "pattern_length": len(key)
            })

            # Aliases
            for alias in info.get("aliases", []):
                self.correction_rules.append({
                    "pattern": rf'\b{re.escape(alias.lower())}\b',
                    "replacement": term,
                    "category": category,
                    "description": f"Alias '{alias}' → {term}",
                    "pattern_length": len(alias)
                })

        # Add custom rules
        for rule in self.custom_rules:
            self.correction_rules.append({
                "pattern": rule[0],
                "replacement": rule[1],
                "category": "custom",
                "description": rule[2] if len(rule) > 2 else "Custom rule",
                "pattern_length": len(rule[0])
            })

        # Sort by pattern length (longer patterns first to match longest phrases first)
        self.correction_rules.sort(key=lambda x: x.get("pattern_length", 0), reverse=True)
    
    def correct(self, text: str, log=True) -> dict:
        """Apply corrections to text"""
        if not self.enabled or not text:
            return {"raw": text, "corrected": text, "corrections": []}

        self.stats["captions_processed"] += 1
        corrected = text
        corrections = []
        protected_ranges = []  # Track ranges in the corrected text that shouldn't be modified

        for rule in self.correction_rules:
            try:
                pattern = rule["pattern"]
                replacement = rule["replacement"]

                # Find matches in CURRENT corrected text
                new_corrected = corrected
                offset = 0

                for match in re.finditer(pattern, corrected, re.IGNORECASE):
                    original = match.group()
                    start, end = match.span()

                    # Adjust for offset from previous replacements in this rule
                    actual_start = start + offset
                    actual_end = end + offset

                    # Check if this position is in a protected range
                    is_protected = False
                    for prot_start, prot_end in protected_ranges:
                        if not (actual_end <= prot_start or actual_start >= prot_end):
                            is_protected = True
                            break

                    if is_protected or original.lower() == replacement.lower():
                        continue

                    # Record the correction
                    if not any(c["from"] == original and c["to"] == replacement for c in corrections):
                        correction = {
                            "from": original,
                            "to": replacement,
                            "category": rule["category"],
                            "timestamp": datetime.now().isoformat()
                        }
                        corrections.append(correction)
                        if log:
                            self.corrections_log.append(correction)

                    # Apply the correction with offset
                    new_corrected = new_corrected[:actual_start] + replacement + new_corrected[actual_end:]

                    # Add this replacement to protected ranges
                    new_end = actual_start + len(replacement)
                    protected_ranges.append((actual_start, new_end))

                    # Update offset for next match in this rule
                    offset += len(replacement) - len(original)

                corrected = new_corrected

            except (re.error, IndexError):
                continue

        if corrections:
            self.stats["corrections_applied"] += len(corrections)

        return {"raw": text, "corrected": corrected, "corrections": corrections}
    
    def get_status(self):
        return {
            "enabled": self.enabled,
            "name": self.name,
            "description": self.description,
            "terms_count": len(self.terms),
            "rules_count": len(self.correction_rules),
            "custom_rules_count": len(self.custom_rules),
            "stats": self.stats,
            "recent_corrections": list(self.corrections_log)[-20:]
        }
    
    def get_terms_list(self):
        return [
            {"term": info["term"], "category": info.get("category", "other"), "aliases": info.get("aliases", [])}
            for info in self.terms.values()
        ]
    
    def set_enabled(self, enabled):
        self.enabled = enabled
        self._save_state()
    
    def add_term(self, term, category="other", aliases=None):
        key = term.lower()
        self.terms[key] = {
            "term": term,
            "category": category,
            "aliases": aliases or [],
            "added": datetime.now().isoformat()
        }
        self._rebuild_rules()
        self._save_state()
    
    def remove_term(self, term):
        key = term.lower()
        if key in self.terms:
            del self.terms[key]
            self._rebuild_rules()
            self._save_state()
    
    def add_custom_rule(self, pattern, replacement, description=""):
        self.custom_rules.append([pattern, replacement, description])
        self._rebuild_rules()
        self._save_state()
    
    def add_defaults(self):
        """Add default Brookline terms"""
        defaults = {
            "person": [
                ("Bernard Greene", ["bernard green", "bernie greene"]),
                ("Heather Hamilton", []),
                ("John VanScoyoc", ["van scoyoc"]),
                ("Paul Warren", []),
                ("David Pearlman", ["dave pearlman"]),
                ("Todd Kirrane", ["tod kirrane"]),
                ("Mark Zarrillo", ["zarrillo"]),
                ("Ben Franco", []),
                ("Melissa Goff", []),
                ("Linus Guillory", []),
                ("Stephen Walter", ["steve walter"]),
            ],
            "place": [
                ("Brookline", []),
                ("Coolidge Corner", []),
                ("Brookline Village", []),
                ("Washington Square", []),
                ("Chestnut Hill", []),
                ("Town Hall", []),
                ("Brookline High School", ["bhs"]),
                ("Harvard Street", []),
                ("Beacon Street", []),
                ("Washington Street", []),
                ("Larz Anderson Park", ["larz anderson"]),
            ],
            "organization": [
                ("Select Board", ["selectboard", "board of selectmen"]),
                ("Town Meeting", []),
                ("Advisory Committee", []),
                ("School Committee", []),
                ("Planning Board", []),
                ("Zoning Board of Appeals", ["zba"]),
                ("BIG", []),
                ("Brookline Interactive Group", []),
                ("DPW", ["department of public works"]),
                ("MBTA", []),
            ]
        }
        
        for category, items in defaults.items():
            for term, aliases in items:
                key = term.lower()
                if key not in self.terms:
                    self.terms[key] = {
                        "term": term,
                        "category": category,
                        "aliases": aliases,
                        "source": "default"
                    }
        
        self._rebuild_rules()
        self._save_state()
        return len(self.terms)
    
    def export_engine(self) -> dict:
        """Export engine as portable JSON"""
        return {
            "version": self.VERSION,
            "name": self.name,
            "description": self.description,
            "created": datetime.now().isoformat(),
            "terms": self.terms,
            "custom_rules": self.custom_rules,
            "stats": {
                "terms_count": len(self.terms),
                "rules_count": len(self.correction_rules)
            }
        }
    
    def import_engine(self, data: dict):
        """Import engine from JSON"""
        if data.get("version") != self.VERSION:
            print(f"Warning: Engine version mismatch ({data.get('version')} vs {self.VERSION})")
        
        self.name = data.get("name", "Imported Engine")
        self.description = data.get("description", "")
        self.terms = data.get("terms", {})
        self.custom_rules = data.get("custom_rules", [])
        self._rebuild_rules()
        self._save_state()
        return {"status": "ok", "terms_count": len(self.terms)}


# =============================================================================
# SESSION MANAGER - Recording and export
# =============================================================================

class SessionManager:
    """Manages captioning sessions with timestamps and export"""
    
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.current_session = None
        self.captions = []  # [{timestamp, raw, corrected, corrections}]
        self.start_time = None
        self.is_recording = False
    
    def start_session(self, name=None):
        """Start a new captioning session"""
        self.current_session = {
            "id": str(uuid.uuid4())[:8],
            "name": name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "started": datetime.now().isoformat(),
            "mode": "browser"
        }
        self.captions = []
        self.start_time = datetime.now()
        self.is_recording = True
        return self.current_session
    
    def add_caption(self, raw: str, corrected: str, corrections: list):
        """Add a caption to the current session"""
        if not self.is_recording:
            return
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.captions.append({
            "timestamp": elapsed,
            "time_str": self._format_time(elapsed),
            "raw": raw,
            "corrected": corrected,
            "corrections": corrections,
            "added": datetime.now().isoformat()
        })
    
    def stop_session(self):
        """Stop the current session"""
        if not self.current_session:
            return None
        
        self.current_session["ended"] = datetime.now().isoformat()
        self.current_session["duration"] = (datetime.now() - self.start_time).total_seconds()
        self.current_session["caption_count"] = len(self.captions)
        self.is_recording = False
        
        # Save session
        session_file = self.sessions_dir / f"{self.current_session['id']}.json"
        with open(session_file, "w") as f:
            json.dump({
                "session": self.current_session,
                "captions": self.captions
            }, f, indent=2)
        
        return self.current_session
    
    def get_status(self):
        """Get current session status"""
        if not self.is_recording:
            return {"recording": False}
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        total_corrections = sum(len(c.get("corrections", [])) for c in self.captions)
        
        return {
            "recording": True,
            "session": self.current_session,
            "duration": elapsed,
            "duration_str": self._format_time(elapsed),
            "caption_count": len(self.captions),
            "correction_count": total_corrections
        }
    
    def get_captions(self):
        """Get all captions from current session"""
        return self.captions
    
    def _format_time(self, seconds):
        """Format seconds as HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
    
    def _format_vtt_time(self, seconds):
        """Format seconds as HH:MM:SS.mmm for VTT"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def export_srt(self, use_corrected=True):
        """Export captions as SRT"""
        lines = []
        for i, cap in enumerate(self.captions, 1):
            start = cap["timestamp"]
            end = start + 3.0  # 3 second duration per caption
            if i < len(self.captions):
                end = min(end, self.captions[i]["timestamp"])
            
            text = cap["corrected"] if use_corrected else cap["raw"]
            lines.append(f"{i}")
            lines.append(f"{self._format_time(start)} --> {self._format_time(end)}")
            lines.append(text)
            lines.append("")
        
        return "\n".join(lines)
    
    def export_vtt(self, use_corrected=True):
        """Export captions as WebVTT"""
        lines = ["WEBVTT", ""]
        for i, cap in enumerate(self.captions):
            start = cap["timestamp"]
            end = start + 3.0
            if i + 1 < len(self.captions):
                end = min(end, self.captions[i + 1]["timestamp"])
            
            text = cap["corrected"] if use_corrected else cap["raw"]
            lines.append(f"{self._format_vtt_time(start)} --> {self._format_vtt_time(end)}")
            lines.append(text)
            lines.append("")
        
        return "\n".join(lines)
    
    def export_txt(self, use_corrected=True):
        """Export as plain text transcript"""
        lines = []
        for cap in self.captions:
            text = cap["corrected"] if use_corrected else cap["raw"]
            lines.append(f"[{cap['time_str'][:8]}] {text}")
        return "\n".join(lines)
    
    def export_json(self):
        """Export full session data as JSON"""
        return json.dumps({
            "session": self.current_session,
            "captions": self.captions
        }, indent=2)
    
    def generate_summary(self):
        """Generate a summary of the session using AI if available"""
        if not self.captions:
            return {"summary": "No captions recorded.", "highlights": []}

        # Combine all text
        full_text = " ".join(cap["corrected"] for cap in self.captions)
        word_count = len(full_text.split())

        # Calculate stats
        total_corrections = sum(len(c.get("corrections", [])) for c in self.captions)
        duration = self.current_session.get("duration", 0) if self.current_session else 0

        # Base stats
        stats = {
            "duration_minutes": round(duration / 60, 1),
            "caption_count": len(self.captions),
            "word_count": word_count,
            "corrections_made": total_corrections
        }

        # Try AI-powered summary first
        if openai_client:
            try:
                # Generate AI summary
                response = openai_client.chat.completions.create(
                    model=get_ai_model(),
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at summarizing meeting transcripts and civic proceedings. "
                                      "Extract key topics, decisions, speakers mentioned, and notable quotes. "
                                      "Provide a concise executive summary, list of topics discussed, and key highlights."
                        },
                        {
                            "role": "user",
                            "content": f"Please summarize this meeting transcript:\n\n{full_text[:8000]}"  # Limit to avoid token limits
                        }
                    ],
                    temperature=0.3,
                    max_tokens=800
                )

                ai_summary = response.choices[0].message.content

                # Extract entities with AI
                entity_response = openai_client.chat.completions.create(
                    model=get_ai_model(),
                    messages=[
                        {
                            "role": "system",
                            "content": "Extract people names, places, and organizations mentioned in the text. "
                                      "Return ONLY a JSON object with three arrays: people, places, organizations. "
                                      "Each array should contain unique strings."
                        },
                        {
                            "role": "user",
                            "content": full_text[:4000]
                        }
                    ],
                    temperature=0.1,
                    max_tokens=500
                )

                try:
                    entities = json.loads(entity_response.choices[0].message.content)
                except:
                    entities = {"people": [], "places": [], "organizations": []}

                # Find highlights (captions with corrections)
                highlights = []
                for cap in self.captions:
                    if len(cap.get("corrections", [])) > 0:
                        highlights.append({
                            "time": cap["time_str"][:8],
                            "text": cap["corrected"],
                            "corrections": cap["corrections"]
                        })

                return {
                    "summary": ai_summary,
                    "stats": stats,
                    "highlights": highlights[:10],
                    "entities": entities,
                    "ai_generated": True
                }

            except Exception as e:
                print(f"AI summary failed: {e}")
                # Fall through to basic summary

        # Fallback: Basic summary without AI
        entities = {"people": set(), "places": set(), "organizations": set()}

        # Simple extraction based on capitalization patterns
        words = full_text.split()
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 2:
                # Check if it's part of a name (next word also capitalized)
                if i + 1 < len(words) and words[i + 1][0].isupper():
                    entities["people"].add(f"{word} {words[i + 1]}")

        basic_summary = (
            f"Session recorded {len(self.captions)} caption segments over {int(duration // 60)} minutes. "
            f"Approximately {word_count} words were transcribed with {total_corrections} automatic corrections applied."
        )

        # Extract potential highlights (longer captions, ones with corrections)
        highlights = []
        for cap in self.captions:
            if len(cap.get("corrections", [])) > 0:
                highlights.append({
                    "time": cap["time_str"][:8],
                    "text": cap["corrected"],
                    "corrections": cap["corrections"]
                })

        return {
            "summary": basic_summary,
            "stats": stats,
            "highlights": highlights[:10],
            "entities": {k: list(v)[:10] for k, v in entities.items()},
            "ai_generated": False
        }


# =============================================================================
# WHISPER ENGINE
# =============================================================================

class WhisperEngine:
    """Local speech-to-text using faster-whisper"""
    
    MODEL_INFO = {
        "tiny": {"size": "75 MB", "speed": "Fastest", "quality": "Basic"},
        "base": {"size": "150 MB", "speed": "Fast", "quality": "Good"},
        "small": {"size": "500 MB", "speed": "Medium", "quality": "Better"},
        "medium": {"size": "1.5 GB", "speed": "Slow", "quality": "Great"},
    }
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.is_running = False
        self.text_callback = None
        self.sample_rate = 16000
        self.chunk_duration = 3.0
        self.stream = None
        self.thread = None
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
    def get_status(self):
        missing = []
        if not WHISPER_AVAILABLE: missing.append("faster-whisper")
        if not AUDIO_AVAILABLE: missing.append("sounddevice")
        if not NUMPY_AVAILABLE: missing.append("numpy")
        
        return {
            "available": WHISPER_AVAILABLE and AUDIO_AVAILABLE and NUMPY_AVAILABLE,
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "is_running": self.is_running,
            "missing_packages": missing,
            "models": self.MODEL_INFO
        }
    
    def load_model(self, model_name="base"):
        if not WHISPER_AVAILABLE:
            return {"error": "faster-whisper not installed"}
        
        try:
            print(f"🔄 Loading Whisper model '{model_name}'...")
            self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
            self.model_name = model_name
            print(f"✅ Model '{model_name}' ready!")
            return {"status": "ok", "model": model_name}
        except Exception as e:
            return {"error": str(e)}
    
    def get_audio_devices(self):
        if not AUDIO_AVAILABLE:
            return []
        devices = []
        try:
            for i, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0:
                    devices.append({
                        "id": i,
                        "name": device['name'],
                        "is_default": i == sd.default.device[0]
                    })
        except:
            pass
        return devices
    
    def _audio_callback(self, indata, frames, time_info, status):
        with self.buffer_lock:
            self.audio_buffer.append(indata.copy())
    
    def _transcription_loop(self):
        samples_needed = int(self.sample_rate * self.chunk_duration)
        
        while self.is_running:
            try:
                time.sleep(0.1)
                with self.buffer_lock:
                    if not self.audio_buffer:
                        continue
                    total = sum(len(c) for c in self.audio_buffer)
                    if total < samples_needed:
                        continue
                    audio = np.concatenate(self.audio_buffer).flatten().astype(np.float32)
                    self.audio_buffer = []
                
                if self.model and len(audio) > 0:
                    segments, _ = self.model.transcribe(audio, beam_size=5, language="en", vad_filter=True)
                    text = " ".join(seg.text for seg in segments).strip()
                    if text and self.text_callback:
                        self.text_callback(text)
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ Transcription error: {e}")
                    time.sleep(1)
    
    def start(self, device_id=None, callback=None):
        if not AUDIO_AVAILABLE or not NUMPY_AVAILABLE:
            return {"error": "Missing packages"}
        if not self.model:
            return {"error": "No model loaded"}
        if self.is_running:
            return {"error": "Already running"}
        
        self.text_callback = callback
        self.is_running = True
        self.audio_buffer = []
        
        try:
            self.stream = sd.InputStream(
                device=device_id, channels=1, samplerate=self.sample_rate,
                callback=self._audio_callback, blocksize=int(self.sample_rate * 0.1)
            )
            self.stream.start()
            self.thread = threading.Thread(target=self._transcription_loop, daemon=True)
            self.thread.start()
            print(f"🎤 Whisper listening...")
            return {"status": "started"}
        except Exception as e:
            self.is_running = False
            return {"error": str(e)}
    
    def stop(self):
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
        return {"status": "stopped"}
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio data for accuracy pass"""
        if not self.model:
            return {"error": "No model loaded"}
        
        try:
            segments, _ = self.model.transcribe(audio_data, beam_size=5, language="en")
            return {"text": " ".join(seg.text for seg in segments).strip()}
        except Exception as e:
            return {"error": str(e)}


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

caption_engine = CaptionEngine(DATA_DIR)
session_manager = SessionManager(SESSIONS_DIR)
whisper_engine = WhisperEngine()

# Caption state
caption_state = {
    "caption": "",
    "raw_caption": "",
    "corrections": [],
    "mode": "browser",
    "whisper_running": False,
    "settings": {
        "fontSize": 48, "fontFamily": "DM Sans", "textColor": "#FFFFFF",
        "backgroundColor": "#000000", "backgroundOpacity": 75, "textAlign": "center",
        "position": "bottom", "maxLines": 2, "maxWidth": 80,
        "showBackground": True, "textShadow": True, "paddingH": 32, "paddingV": 16
    }
}

whisper_buffer = ""

def on_whisper_text(text):
    """Callback when Whisper produces text"""
    global whisper_buffer
    whisper_buffer = (whisper_buffer + " " + text).strip()
    result = process_caption(whisper_buffer)
    caption_state["raw_caption"] = result["raw"]
    caption_state["caption"] = result["corrected"]
    caption_state["corrections"] = result["corrections"]
    
    # Add to session if recording
    if session_manager.is_recording:
        session_manager.add_caption(result["raw"], result["corrected"], result["corrections"])
    
    print(f"   📝 {text}")

def limit_lines(text, max_lines=2, max_width=80):
    chars = int(50 * max_width / 80) * max_lines
    if len(text) <= chars:
        return text
    t = text[-chars:]
    sp = t.find(' ')
    return sp > 0 and sp < 20 and t[sp+1:] or t

def process_caption(text):
    limited = limit_lines(text, caption_state["settings"].get("maxLines", 2), caption_state["settings"].get("maxWidth", 80))
    return caption_engine.correct(limited)

# =============================================================================
# HTTP HANDLER
# =============================================================================

class Handler(http.server.SimpleHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)
    
    def send_file_download(self, content, filename, content_type='text/plain'):
        body = content.encode() if isinstance(content, str) else content
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)
    
    def read_json(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            return json.loads(self.rfile.read(length).decode()) if length else {}
        except:
            return {}
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        path = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)
        
        # Caption endpoints
        if path == '/api/caption':
            self.send_json({**caption_state, "session": session_manager.get_status()})
        
        elif path == '/api/info':
            self.send_json({
                "local_ip": LOCAL_IP, "port": PORT,
                "whisper_available": WHISPER_AVAILABLE and AUDIO_AVAILABLE and NUMPY_AVAILABLE,
                "version": "2.0"
            })
        
        # Engine endpoints
        elif path == '/api/engine/status':
            self.send_json(caption_engine.get_status())
        
        elif path == '/api/engine/terms':
            self.send_json({"terms": caption_engine.get_terms_list()})
        
        elif path == '/api/engine/test':
            text = query.get('text', [''])[0]
            self.send_json(caption_engine.correct(text, log=False))
        
        elif path == '/api/engine/corrections':
            self.send_json({"corrections": list(caption_engine.corrections_log)})
        
        elif path == '/api/engine/export':
            engine_data = caption_engine.export_engine()
            filename = f"{caption_engine.name.replace(' ', '_').lower()}_engine.json"
            self.send_file_download(json.dumps(engine_data, indent=2), filename, 'application/json')
        
        # Whisper endpoints
        elif path == '/api/whisper/status':
            self.send_json(whisper_engine.get_status())
        
        elif path == '/api/whisper/devices':
            self.send_json({"devices": whisper_engine.get_audio_devices()})
        
        # Session endpoints
        elif path == '/api/session/status':
            self.send_json(session_manager.get_status())
        
        elif path == '/api/session/captions':
            self.send_json({"captions": session_manager.get_captions()})
        
        elif path == '/api/session/export/srt':
            corrected = query.get('corrected', ['true'])[0] == 'true'
            srt = session_manager.export_srt(corrected)
            self.send_file_download(srt, "captions.srt", "text/srt")
        
        elif path == '/api/session/export/vtt':
            corrected = query.get('corrected', ['true'])[0] == 'true'
            vtt = session_manager.export_vtt(corrected)
            self.send_file_download(vtt, "captions.vtt", "text/vtt")
        
        elif path == '/api/session/export/txt':
            corrected = query.get('corrected', ['true'])[0] == 'true'
            txt = session_manager.export_txt(corrected)
            self.send_file_download(txt, "transcript.txt", "text/plain")
        
        elif path == '/api/session/export/json':
            self.send_file_download(session_manager.export_json(), "session.json", "application/json")
        
        elif path == '/api/session/summary':
            self.send_json(session_manager.generate_summary())
        
        # List saved engines
        elif path == '/api/engines/list':
            engines = []
            for f in ENGINES_DIR.glob("*.json"):
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                        engines.append({
                            "filename": f.name,
                            "name": data.get("name", f.stem),
                            "terms_count": len(data.get("terms", {})),
                            "created": data.get("created", "")
                        })
                except:
                    pass
            self.send_json({"engines": engines})

        # AI Configuration endpoints (GET)
        elif path == '/api/ai/config':
            # Get current AI configuration
            config_to_send = ai_config.copy()
            # Don't send full API keys, just indicate if they're set
            if config_to_send.get("openai_api_key"):
                config_to_send["openai_api_key_set"] = True
                config_to_send["openai_api_key"] = config_to_send["openai_api_key"][:8] + "..." if len(config_to_send["openai_api_key"]) > 8 else "***"
            if config_to_send.get("custom_api_key"):
                config_to_send["custom_api_key_set"] = True
                config_to_send["custom_api_key"] = "***"

            config_to_send["available"] = OPENAI_AVAILABLE
            config_to_send["active"] = openai_client is not None
            config_to_send["is_local"] = is_ai_local()
            config_to_send["model"] = get_ai_model()

            self.send_json(config_to_send)

        else:
            super().do_GET()
    
    def do_POST(self):
        global whisper_buffer
        path = urlparse(self.path).path
        data = self.read_json()
        
        # Caption from browser
        if path == '/api/caption':
            if 'caption' in data:
                result = process_caption(data['caption'])
                caption_state['raw_caption'] = result["raw"]
                caption_state['caption'] = result["corrected"]
                caption_state['corrections'] = result["corrections"]
                
                # Add to session
                if session_manager.is_recording and result["corrected"]:
                    session_manager.add_caption(result["raw"], result["corrected"], result["corrections"])
            
            if 'settings' in data:
                caption_state['settings'].update(data['settings'])
            
            self.send_json({"status": "ok", "corrections": caption_state.get("corrections", [])})
        
        elif path == '/api/clear':
            whisper_buffer = ""
            caption_state["caption"] = ""
            caption_state["raw_caption"] = ""
            caption_state["corrections"] = []
            self.send_json({"status": "ok"})
        
        # Engine controls
        elif path == '/api/engine/enable':
            caption_engine.set_enabled(data.get('enabled', True))
            self.send_json({"status": "ok", "enabled": caption_engine.enabled})
        
        elif path == '/api/engine/term':
            term = data.get('term', '').strip()
            if term:
                caption_engine.add_term(term, data.get('category', 'other'), data.get('aliases', []))
                self.send_json({"status": "ok"})
            else:
                self.send_json({"error": "No term"}, 400)
        
        elif path == '/api/engine/term/remove':
            caption_engine.remove_term(data.get('term', ''))
            self.send_json({"status": "ok"})
        
        elif path == '/api/engine/defaults':
            count = caption_engine.add_defaults()
            self.send_json({"status": "ok", "count": count})
        
        elif path == '/api/engine/import':
            try:
                result = caption_engine.import_engine(data)
                self.send_json(result)
            except Exception as e:
                self.send_json({"error": str(e)}, 400)
        
        elif path == '/api/engine/save':
            # Save engine to engines folder
            engine_data = caption_engine.export_engine()
            filename = data.get('filename', f"{caption_engine.name.replace(' ', '_').lower()}.json")
            with open(ENGINES_DIR / filename, 'w') as f:
                json.dump(engine_data, f, indent=2)
            self.send_json({"status": "ok", "filename": filename})
        
        elif path == '/api/engine/load':
            # Load engine from engines folder
            filename = data.get('filename', '')
            filepath = ENGINES_DIR / filename
            if filepath.exists():
                with open(filepath) as f:
                    engine_data = json.load(f)
                result = caption_engine.import_engine(engine_data)
                self.send_json(result)
            else:
                self.send_json({"error": "Engine not found"}, 404)

        elif path == '/api/engine/document/extract':
            # Extract terms from document text using AI
            if not openai_client:
                self.send_json({"error": "OpenAI not available. Set OPENAI_API_KEY environment variable."}, 400)
                return

            text = data.get('text', '')
            if not text:
                self.send_json({"error": "No text provided"}, 400)
                return

            try:
                response = openai_client.chat.completions.create(
                    model=get_ai_model(),
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at extracting proper nouns from documents. "
                                      "Extract all person names, place names, and organization names from the text. "
                                      "Return ONLY a JSON object with this structure:\n"
                                      '{"terms": [{"term": "Name", "category": "person|place|organization", "aliases": ["alternate spelling"]}, ...]}\n'
                                      "Include common misspellings or alternate forms in aliases. "
                                      "Only include proper nouns that appear in the text."
                        },
                        {
                            "role": "user",
                            "content": f"Extract proper nouns from this text:\n\n{text[:6000]}"
                        }
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )

                result_text = response.choices[0].message.content

                # Parse JSON from response
                try:
                    extracted = json.loads(result_text)
                    terms = extracted.get('terms', [])

                    # Add terms to engine
                    added = 0
                    for term_data in terms:
                        term = term_data.get('term', '')
                        category = term_data.get('category', 'other')
                        aliases = term_data.get('aliases', [])

                        if term:
                            caption_engine.add_term(term, category, aliases, source="document")
                            added += 1

                    self.send_json({
                        "status": "ok",
                        "terms_extracted": len(terms),
                        "terms_added": added,
                        "terms": terms
                    })

                except json.JSONDecodeError:
                    self.send_json({"error": "Failed to parse AI response", "raw": result_text}, 500)

            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        elif path == '/api/engine/web/extract':
            # Extract terms from web URL using AI
            if not openai_client:
                self.send_json({"error": "OpenAI not available. Set OPENAI_API_KEY environment variable."}, 400)
                return

            url = data.get('url', '')
            if not url:
                self.send_json({"error": "No URL provided"}, 400)
                return

            try:
                # Fetch web page content
                import urllib.request

                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )

                with urllib.request.urlopen(req, timeout=10) as response:
                    html = response.read().decode('utf-8', errors='ignore')

                # Extract text from HTML (simple approach)
                import re
                # Remove script and style elements
                text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', text)
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()

                # Use AI to extract terms
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at extracting proper nouns from web content. "
                                      "Extract all person names, place names, and organization names from the text. "
                                      "Return ONLY a JSON object with this structure:\n"
                                      '{"terms": [{"term": "Name", "category": "person|place|organization", "aliases": ["alternate spelling"]}, ...]}\n'
                                      "Include common misspellings or alternate forms in aliases. "
                                      "Only include proper nouns that appear in the text."
                        },
                        {
                            "role": "user",
                            "content": f"Extract proper nouns from this webpage text:\n\n{text[:6000]}"
                        }
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )

                result_text = response.choices[0].message.content

                # Parse JSON from response
                try:
                    extracted = json.loads(result_text)
                    terms = extracted.get('terms', [])

                    # Add terms to engine
                    added = 0
                    for term_data in terms:
                        term = term_data.get('term', '')
                        category = term_data.get('category', 'other')
                        aliases = term_data.get('aliases', [])

                        if term:
                            caption_engine.add_term(term, category, aliases, source="web")
                            added += 1

                    self.send_json({
                        "status": "ok",
                        "url": url,
                        "terms_extracted": len(terms),
                        "terms_added": added,
                        "terms": terms
                    })

                except json.JSONDecodeError:
                    self.send_json({"error": "Failed to parse AI response", "raw": result_text}, 500)

            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        # Whisper controls
        elif path == '/api/whisper/load':
            self.send_json(whisper_engine.load_model(data.get('model', 'base')))
        
        elif path == '/api/whisper/start':
            whisper_buffer = ""
            result = whisper_engine.start(data.get('device_id'), on_whisper_text)
            if result.get("status") == "started":
                caption_state["mode"] = "whisper"
                caption_state["whisper_running"] = True
            self.send_json(result)
        
        elif path == '/api/whisper/stop':
            result = whisper_engine.stop()
            caption_state["whisper_running"] = False
            self.send_json(result)
        
        # Session controls
        elif path == '/api/session/start':
            session = session_manager.start_session(data.get('name'))
            self.send_json({"status": "ok", "session": session})
        
        elif path == '/api/session/stop':
            session = session_manager.stop_session()
            self.send_json({"status": "ok", "session": session})
        
        elif path == '/api/session/reprocess':
            # Reprocess session with Whisper for accuracy
            # This would need audio recording - for now just return the session
            self.send_json({"status": "ok", "message": "Reprocessing requires audio recording"})

        # AI Configuration endpoints (POST)
        elif path == '/api/ai/config/update':
            # Update AI configuration
            new_config = data

            # Update config
            for key in ["provider", "openai_api_key", "openai_model", "ollama_base_url",
                       "ollama_model", "lmstudio_base_url", "lmstudio_model",
                       "custom_base_url", "custom_api_key", "custom_model"]:
                if key in new_config:
                    ai_config[key] = new_config[key]

            # Save config to file
            try:
                with open(AI_CONFIG_FILE, 'w') as f:
                    json.dump(ai_config, f, indent=2)
            except Exception as e:
                self.send_json({"error": f"Failed to save config: {e}"}, 500)
                return

            # Reinitialize client
            init_ai_client()

            self.send_json({
                "status": "ok",
                "provider": ai_config["provider"],
                "active": openai_client is not None,
                "is_local": is_ai_local(),
                "model": get_ai_model()
            })

        elif path == '/api/ai/test':
            # Test AI connection
            if not openai_client:
                self.send_json({"error": "No AI provider configured"}, 400)
                return

            try:
                response = openai_client.chat.completions.create(
                    model=get_ai_model(),
                    messages=[{"role": "user", "content": "Say 'test successful' and nothing else."}],
                    max_tokens=10
                )
                result_text = response.choices[0].message.content
                self.send_json({
                    "status": "ok",
                    "response": result_text,
                    "provider": ai_config["provider"],
                    "model": get_ai_model(),
                    "is_local": is_ai_local()
                })
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        if args and '/api/' not in str(args[0]):
            super().log_message(format, *args)


# =============================================================================
# MAIN
# =============================================================================

LOCAL_IP = get_local_ip()

def print_banner():
    whisper_ok = WHISPER_AVAILABLE and AUDIO_AVAILABLE and NUMPY_AVAILABLE
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🎤 COMMUNITY CAPTIONER v2.0 🎤                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FEATURES:                                                                   ║
║    ⚡ Web Speech Mode     - Real-time browser captions (~200ms)              ║
║    🎯 Whisper Mode        - Accurate local AI ({"✅ Ready" if whisper_ok else "❌ pip3 install faster-whisper":<24})  ║
║    ✨ Caption Engine      - Auto-correct names & places                      ║
║    📝 Session Recording   - Timestamps, export SRT/VTT/TXT                   ║
║    📊 Summaries           - Generate highlights                              ║
║    💾 Portable Engines    - Download/upload correction rules                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Control Panel:  http://localhost:{PORT}                                      ║
║  OBS Overlay:    http://localhost:{PORT}?overlay=true                         ║
║  Network:        http://{LOCAL_IP}:{PORT:<38}  ║
║                                                                              ║
║  Press Ctrl+C to stop                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print_banner()
webbrowser.open(f'http://localhost:{PORT}')

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

with ReusableTCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        whisper_engine.stop()
        if session_manager.is_recording:
            session_manager.stop_session()
        print("\n\nServer stopped. Goodbye!")
