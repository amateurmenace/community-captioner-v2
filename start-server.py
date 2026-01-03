#!/usr/bin/env python3
"""
Community Captioner v4.1 - Advanced RAG Caption Engine

FEATURES:
  - Triple captioning modes (Web Speech / Whisper / Speechmatics)
  - Advanced RAG Caption Engine with semantic similarity matching
  - Vector embeddings for context-aware corrections
  - Fuzzy matching with confidence thresholds
  - Real-time learning from ASR patterns
  - Adaptive latency management with queue drop/catch-up
  - Post-session refinement with consistency enforcement
  - Session recording with timestamps and audio
  - Post-session analytics dashboard
  - Export to SRT, VTT, TXT, JSON
  - AI-powered summary & highlights generation
  - Portable Caption Engines (download/upload)
  - Video intelligence integration

INSTALLATION:
  Basic: python3 start-server.py

  For Whisper mode:
    pip3 install faster-whisper sounddevice numpy

  For Speechmatics mode (cloud ASR):
    pip3 install speechmatics-python

  For AI features (optional):
    pip3 install openai  # or use local LLM via Ollama

  For advanced RAG (optional):
    pip3 install sentence-transformers  # for local embeddings
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
import math
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
import urllib.request
import urllib.error
from collections import deque, Counter
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional, Any

# OpenAI for AI features
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass

# Sentence transformers for local embeddings
EMBEDDINGS_AVAILABLE = False
sentence_model = None
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    pass

# Speechmatics for cloud ASR
SPEECHMATICS_AVAILABLE = False
try:
    import speechmatics
    from speechmatics.models import ConnectionSettings
    from speechmatics.batch_client import BatchClient
    SPEECHMATICS_AVAILABLE = True
except ImportError:
    pass

# WebSocket for Speechmatics real-time
WEBSOCKETS_AVAILABLE = False
try:
    import asyncio
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    pass

# YouTube Transcript API
YOUTUBE_TRANSCRIPT_AVAILABLE = False
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    pass

# PDF and DOCX support
PDF_AVAILABLE = False
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
    except ImportError:
        pass

DOCX_AVAILABLE = False
try:
    import docx
    DOCX_AVAILABLE = True
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
AUDIO_DIR = SCRIPT_DIR / "audio"
KNOWLEDGE_DIR = SCRIPT_DIR / "knowledge"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
ANALYTICS_DIR = SCRIPT_DIR / "analytics"

for d in [DATA_DIR, SESSIONS_DIR, ENGINES_DIR, AUDIO_DIR, KNOWLEDGE_DIR, EMBEDDINGS_DIR, ANALYTICS_DIR]:
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
    "custom_model": "",
    # Speechmatics configuration
    "speechmatics_api_key": os.getenv('SPEECHMATICS_API_KEY', ''),
    # Local Whisper API configuration (faster-whisper-server, LocalAI, etc.)
    "local_whisper_api_url": os.getenv('LOCAL_WHISPER_API_URL', 'http://localhost:8000/v1/audio/transcriptions'),
    "local_whisper_api_key": os.getenv('LOCAL_WHISPER_API_KEY', ''),
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

# Silero VAD for Voice Activity Detection (optional but recommended)
SILERO_VAD_AVAILABLE = False
silero_vad_model = None
silero_get_speech_timestamps = None
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

def load_silero_vad():
    """Load Silero VAD model for voice activity detection"""
    global silero_vad_model, silero_get_speech_timestamps, SILERO_VAD_AVAILABLE
    if not TORCH_AVAILABLE:
        print("⚠️ Silero VAD requires torch. Install with: pip3 install torch")
        return False
    try:
        import torch
        # Load model and utility functions
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True  # Use ONNX for faster inference without torchaudio dependency
        )
        silero_vad_model = model
        (silero_get_speech_timestamps, _, _, _, _) = utils
        SILERO_VAD_AVAILABLE = True
        print("✅ Silero VAD loaded (ONNX mode)")
        return True
    except Exception as e:
        # Try alternative loading method
        try:
            import torch
            silero_vad_model = torch.jit.load(torch.hub.load_state_dict_from_url(
                'https://models.silero.ai/vad_models/silero_vad.jit',
                map_location='cpu'
            ))
            SILERO_VAD_AVAILABLE = True
            print("✅ Silero VAD loaded (JIT fallback)")
            return True
        except Exception as e2:
            print(f"⚠️ Silero VAD not available: {e}. Using energy-based VAD.")
            return False

# =============================================================================
# ADVANCED RAG CAPTION ENGINE v4.0 - Semantic matching with learning
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def fuzzy_ratio(s1: str, s2: str) -> float:
    """Calculate fuzzy match ratio between two strings (0-1)"""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


class EmbeddingsManager:
    """Manages vector embeddings for semantic similarity matching"""

    def __init__(self, embeddings_dir: Path):
        self.embeddings_dir = embeddings_dir
        self.embeddings_cache = {}  # term -> embedding vector
        self.model = None
        self.model_type = None  # 'local' or 'openai'
        self._load_cache()

    def _load_cache(self):
        """Load cached embeddings from disk"""
        cache_file = self.embeddings_dir / "embeddings_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    self.embeddings_cache = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load embeddings cache: {e}")

    def _save_cache(self):
        """Save embeddings cache to disk"""
        try:
            cache_file = self.embeddings_dir / "embeddings_cache.json"
            with open(cache_file, "w") as f:
                json.dump(self.embeddings_cache, f)
        except Exception as e:
            print(f"Warning: Could not save embeddings cache: {e}")

    def init_model(self, use_local=True):
        """Initialize embedding model (local sentence-transformers or OpenAI)"""
        global sentence_model

        if use_local and EMBEDDINGS_AVAILABLE:
            try:
                if sentence_model is None:
                    print("Loading sentence-transformers model...")
                    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model = sentence_model
                self.model_type = 'local'
                print("✓ Local embeddings model loaded (sentence-transformers)")
                return True
            except Exception as e:
                print(f"✗ Local embeddings failed: {e}")

        if openai_client:
            self.model_type = 'openai'
            print("✓ Using OpenAI embeddings")
            return True

        self.model_type = None
        return False

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector for text"""
        if not text:
            return None

        # Check cache first
        cache_key = text.lower().strip()
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]

        embedding = None

        if self.model_type == 'local' and self.model:
            try:
                embedding = self.model.encode(text).tolist()
            except Exception as e:
                print(f"Local embedding error: {e}")

        elif self.model_type == 'openai' and openai_client:
            try:
                response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                embedding = response.data[0].embedding
            except Exception as e:
                print(f"OpenAI embedding error: {e}")

        if embedding:
            self.embeddings_cache[cache_key] = embedding
            self._save_cache()

        return embedding

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def find_similar(self, query: str, candidates: Dict[str, Any], threshold: float = 0.7) -> List[Tuple[str, float, Any]]:
        """Find similar terms from candidates using embeddings"""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        results = []
        for term_key, term_info in candidates.items():
            term = term_info.get("term", term_key)
            term_embedding = self.get_embedding(term)

            if term_embedding:
                similarity = self.cosine_similarity(query_embedding, term_embedding)
                if similarity >= threshold:
                    results.append((term, similarity, term_info))

            # Also check aliases
            for alias in term_info.get("aliases", []):
                alias_embedding = self.get_embedding(alias)
                if alias_embedding:
                    similarity = self.cosine_similarity(query_embedding, alias_embedding)
                    if similarity >= threshold:
                        results.append((term, similarity, term_info))

        # Sort by similarity, highest first
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def is_available(self) -> bool:
        return self.model_type is not None


class ASRLearner:
    """Learns from ASR mistakes to improve future corrections"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.patterns_file = data_dir / "asr_patterns.json"
        self.learned_patterns = {}  # mistake -> {correction, count, confidence}
        self._load_patterns()

    def _load_patterns(self):
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file) as f:
                    self.learned_patterns = json.load(f)
            except Exception:
                pass

    def _save_patterns(self):
        try:
            with open(self.patterns_file, "w") as f:
                json.dump(self.learned_patterns, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save ASR patterns: {e}")

    def learn(self, mistake: str, correction: str):
        """Learn from a correction"""
        key = mistake.lower().strip()
        if key == correction.lower().strip():
            return

        if key in self.learned_patterns:
            entry = self.learned_patterns[key]
            if entry["correction"] == correction:
                entry["count"] += 1
                entry["confidence"] = min(1.0, entry["count"] / 10.0)  # Max confidence at 10 occurrences
            else:
                # Different correction for same mistake - might be context-dependent
                if entry["count"] < 3:  # Switch if not well established
                    entry["correction"] = correction
                    entry["count"] = 1
                    entry["confidence"] = 0.1
        else:
            self.learned_patterns[key] = {
                "correction": correction,
                "count": 1,
                "confidence": 0.1,
                "first_seen": datetime.now().isoformat()
            }

        self._save_patterns()

    def get_correction(self, text: str) -> Optional[Tuple[str, float]]:
        """Get learned correction for a mistake if exists"""
        key = text.lower().strip()
        if key in self.learned_patterns:
            entry = self.learned_patterns[key]
            return (entry["correction"], entry["confidence"])
        return None

    def get_all_patterns(self) -> Dict:
        return self.learned_patterns.copy()

    def clear_patterns(self):
        self.learned_patterns = {}
        self._save_patterns()


class CaptionEngine:
    """
    Advanced RAG Caption Engine v4.0

    Features:
    - Regex-based pattern matching (v1.0 compatibility)
    - Semantic similarity matching with vector embeddings
    - Fuzzy matching with Levenshtein distance
    - Confidence thresholds for auto-correct vs suggestions
    - Real-time learning from corrections
    - Context-aware disambiguation
    - Post-session refinement
    """

    VERSION = "4.0"

    # Confidence thresholds
    AUTO_CORRECT_THRESHOLD = 0.85  # Auto-correct if confidence >= this
    SUGGEST_THRESHOLD = 0.60      # Suggest correction if confidence >= this
    FUZZY_THRESHOLD = 0.80        # Fuzzy match ratio threshold

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.enabled = False
        self.rag_enabled = True  # Enable advanced RAG features
        self.name = "Default Engine"
        self.description = ""
        self.terms = {}  # term_lower -> {term, category, aliases, source, context, metadata}
        self.custom_rules = []  # [(pattern, replacement, description)]
        self.correction_rules = []  # compiled regex rules
        self.stats = {
            "corrections_applied": 0,
            "captions_processed": 0,
            "semantic_matches": 0,
            "fuzzy_matches": 0,
            "learned_corrections": 0
        }
        self.corrections_log = deque(maxlen=200)  # Recent corrections
        self.pending_suggestions = []  # Low-confidence suggestions for review

        # Advanced RAG components
        self.embeddings = EmbeddingsManager(EMBEDDINGS_DIR)
        self.learner = ASRLearner(data_dir)

        # Context tracking for multi-word entities
        self.recent_context = deque(maxlen=50)  # Recent words for context

        # Common ASR confusions for Brookline context
        self.asr_confusions = {
            "brooklyn": "Brookline",
            "brook line": "Brookline",
            "brookland": "Brookline",
            "brooklynn": "Brookline",
            "selectmen": "Select Board",
            "select men": "Select Board",
            "town hall": "Town Hall",
        }

        self._load_state()

        # Initialize embeddings if available
        self.embeddings.init_model(use_local=True)

    def _load_state(self):
        state_file = self.data_dir / "engine_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    self.enabled = data.get("enabled", False)
                    self.rag_enabled = data.get("rag_enabled", True)
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
                    "rag_enabled": self.rag_enabled,
                    "name": self.name,
                    "description": self.description,
                    "terms": self.terms,
                    "custom_rules": self.custom_rules,
                    "version": self.VERSION
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save engine state: {e}")

    def _rebuild_rules(self):
        """Build correction rules from terms and custom rules"""
        self.correction_rules = []

        # Add Brooklyn → Brookline rule (common ASR error) with context exclusions
        self.correction_rules.append({
            "pattern": r'\b[Bb]rooklyn\b(?!\s+(NY|New York|Bridge|Nets|Dodgers|Heights|Center))',
            "replacement": "Brookline",
            "category": "place",
            "description": "ASR commonly mishears Brookline as Brooklyn",
            "pattern_length": len("brooklyn"),
            "confidence": 0.95
        })

        # Add rules from terms
        for key, info in self.terms.items():
            term = info["term"]
            category = info.get("category", "other")

            # Main term pattern - exact match with capitalization fix
            self.correction_rules.append({
                "pattern": rf'\b{re.escape(key)}\b',
                "replacement": term,
                "category": category,
                "description": f"Correct capitalization for {term}",
                "pattern_length": len(key),
                "confidence": 1.0
            })

            # Aliases - known alternate forms
            for alias in info.get("aliases", []):
                alias_lower = alias.lower()
                self.correction_rules.append({
                    "pattern": rf'\b{re.escape(alias_lower)}\b',
                    "replacement": term,
                    "category": category,
                    "description": f"Alias '{alias}' → {term}",
                    "pattern_length": len(alias),
                    "confidence": 0.95
                })

        # Add custom rules
        for rule in self.custom_rules:
            self.correction_rules.append({
                "pattern": rule[0],
                "replacement": rule[1],
                "category": "custom",
                "description": rule[2] if len(rule) > 2 else "Custom rule",
                "pattern_length": len(rule[0]),
                "confidence": 0.90
            })

        # Sort by pattern length (longer patterns first to match longest phrases first)
        self.correction_rules.sort(key=lambda x: x.get("pattern_length", 0), reverse=True)

    def _fuzzy_match_terms(self, word: str) -> Optional[Tuple[str, float, str]]:
        """Try to fuzzy match a word against known terms"""
        if len(word) < 3:
            return None

        word_lower = word.lower()
        best_match = None
        best_score = 0

        for key, info in self.terms.items():
            term = info["term"]

            # Check main term
            ratio = fuzzy_ratio(word_lower, key)
            if ratio > best_score and ratio >= self.FUZZY_THRESHOLD:
                best_match = (term, ratio, info.get("category", "other"))
                best_score = ratio

            # Check aliases
            for alias in info.get("aliases", []):
                ratio = fuzzy_ratio(word_lower, alias.lower())
                if ratio > best_score and ratio >= self.FUZZY_THRESHOLD:
                    best_match = (term, ratio, info.get("category", "other"))
                    best_score = ratio

        return best_match

    def _semantic_match_terms(self, phrase: str) -> Optional[Tuple[str, float, str]]:
        """Try to semantically match a phrase against known terms"""
        if not self.embeddings.is_available():
            return None

        similar = self.embeddings.find_similar(phrase, self.terms, threshold=self.SUGGEST_THRESHOLD)

        if similar:
            term, similarity, info = similar[0]
            return (term, similarity, info.get("category", "other"))

        return None

    def _check_context_disambiguation(self, word: str, context: List[str]) -> Optional[str]:
        """Check context to disambiguate corrections"""
        word_lower = word.lower()
        context_lower = " ".join(context).lower()

        # Brooklyn vs Brookline disambiguation
        if word_lower in ["brooklyn", "brook", "line"]:
            # If NY, New York, or other NYC references nearby, don't correct
            nyc_indicators = ["ny", "new york", "manhattan", "queens", "bronx", "nets", "dodgers"]
            for indicator in nyc_indicators:
                if indicator in context_lower:
                    return None  # Don't correct - probably actually means Brooklyn, NY

            # Brookline indicators
            brookline_indicators = ["massachusetts", "ma", "select board", "town meeting", "brookline"]
            for indicator in brookline_indicators:
                if indicator in context_lower:
                    return "Brookline"  # Definitely Brookline

        return None

    def correct(self, text: str, log=True, use_rag=None) -> dict:
        """
        Apply corrections to text using hybrid approach:
        1. Regex pattern matching (fast, exact)
        2. Learned ASR patterns
        3. Fuzzy matching (for typos)
        4. Semantic similarity (for context-aware corrections)
        """
        if not self.enabled or not text:
            return {"raw": text, "corrected": text, "corrections": [], "suggestions": []}

        use_rag = use_rag if use_rag is not None else self.rag_enabled

        self.stats["captions_processed"] += 1
        corrected = text
        corrections = []
        suggestions = []
        protected_ranges = []

        # Update context
        words = text.split()
        self.recent_context.extend(words)
        context = list(self.recent_context)

        # Phase 1: Apply regex pattern matching
        for rule in self.correction_rules:
            try:
                pattern = rule["pattern"]
                replacement = rule["replacement"]
                confidence = rule.get("confidence", 1.0)

                new_corrected = corrected
                offset = 0

                for match in re.finditer(pattern, corrected, re.IGNORECASE):
                    original = match.group()
                    start, end = match.span()

                    actual_start = start + offset
                    actual_end = end + offset

                    is_protected = False
                    for prot_start, prot_end in protected_ranges:
                        if not (actual_end <= prot_start or actual_start >= prot_end):
                            is_protected = True
                            break

                    if is_protected or original.lower() == replacement.lower():
                        continue

                    # Check context disambiguation
                    context_override = self._check_context_disambiguation(original, context)
                    if context_override is None and rule.get("description", "").startswith("ASR"):
                        # Context suggests not to correct
                        continue
                    elif context_override:
                        replacement = context_override

                    correction = {
                        "from": original,
                        "to": replacement,
                        "category": rule["category"],
                        "confidence": confidence,
                        "method": "regex",
                        "timestamp": datetime.now().isoformat()
                    }

                    if confidence >= self.AUTO_CORRECT_THRESHOLD:
                        if not any(c["from"] == original and c["to"] == replacement for c in corrections):
                            corrections.append(correction)
                            if log:
                                self.corrections_log.append(correction)
                                self.learner.learn(original, replacement)

                        new_corrected = new_corrected[:actual_start] + replacement + new_corrected[actual_end:]
                        new_end = actual_start + len(replacement)
                        protected_ranges.append((actual_start, new_end))
                        offset += len(replacement) - len(original)
                    else:
                        suggestions.append(correction)

                corrected = new_corrected

            except (re.error, IndexError):
                continue

        # Phase 2: Check learned ASR patterns (if RAG enabled)
        if use_rag:
            words = corrected.split()
            new_words = []

            for word in words:
                learned = self.learner.get_correction(word)
                if learned:
                    learned_correction, learned_confidence = learned
                    if learned_confidence >= self.AUTO_CORRECT_THRESHOLD:
                        if word.lower() != learned_correction.lower():
                            correction = {
                                "from": word,
                                "to": learned_correction,
                                "category": "learned",
                                "confidence": learned_confidence,
                                "method": "learned",
                                "timestamp": datetime.now().isoformat()
                            }
                            corrections.append(correction)
                            self.stats["learned_corrections"] += 1
                            if log:
                                self.corrections_log.append(correction)
                            new_words.append(learned_correction)
                            continue

                new_words.append(word)

            corrected = " ".join(new_words)

        # Phase 3: Fuzzy matching for remaining unmatched words (if RAG enabled)
        if use_rag:
            words = corrected.split()
            new_words = []

            for word in words:
                # Skip short words and already corrected
                if len(word) < 4 or any(word in c["to"] for c in corrections):
                    new_words.append(word)
                    continue

                fuzzy_result = self._fuzzy_match_terms(word)
                if fuzzy_result:
                    fuzzy_term, fuzzy_score, fuzzy_category = fuzzy_result
                    if word.lower() != fuzzy_term.lower():
                        correction = {
                            "from": word,
                            "to": fuzzy_term,
                            "category": fuzzy_category,
                            "confidence": fuzzy_score,
                            "method": "fuzzy",
                            "timestamp": datetime.now().isoformat()
                        }

                        if fuzzy_score >= self.AUTO_CORRECT_THRESHOLD:
                            corrections.append(correction)
                            self.stats["fuzzy_matches"] += 1
                            if log:
                                self.corrections_log.append(correction)
                                self.learner.learn(word, fuzzy_term)
                            new_words.append(fuzzy_term)
                            continue
                        else:
                            suggestions.append(correction)

                new_words.append(word)

            corrected = " ".join(new_words)

        # Phase 4: Semantic matching for multi-word phrases (if RAG enabled and embeddings available)
        if use_rag and self.embeddings.is_available():
            # Look for 2-3 word phrases that might be entity names
            words = corrected.split()
            for n in [3, 2]:  # Try 3-word phrases first, then 2-word
                i = 0
                while i <= len(words) - n:
                    phrase = " ".join(words[i:i+n])

                    # Skip if already looks correct (starts with capital)
                    if phrase[0].isupper() and all(w[0].isupper() for w in words[i:i+n] if w):
                        i += 1
                        continue

                    semantic_result = self._semantic_match_terms(phrase)
                    if semantic_result:
                        semantic_term, semantic_score, semantic_category = semantic_result
                        if phrase.lower() != semantic_term.lower():
                            correction = {
                                "from": phrase,
                                "to": semantic_term,
                                "category": semantic_category,
                                "confidence": semantic_score,
                                "method": "semantic",
                                "timestamp": datetime.now().isoformat()
                            }

                            if semantic_score >= self.AUTO_CORRECT_THRESHOLD:
                                self.stats["semantic_matches"] += 1
                                corrections.append(correction)
                                if log:
                                    self.corrections_log.append(correction)
                                # Replace phrase in corrected text
                                corrected = corrected.replace(phrase, semantic_term, 1)
                            else:
                                suggestions.append(correction)

                    i += 1

        if corrections:
            self.stats["corrections_applied"] += len(corrections)

        # Store suggestions for review
        self.pending_suggestions.extend(suggestions)
        if len(self.pending_suggestions) > 100:
            self.pending_suggestions = self.pending_suggestions[-100:]

        return {
            "raw": text,
            "corrected": corrected,
            "corrections": corrections,
            "suggestions": suggestions
        }

    def refine_session(self, captions: List[Dict]) -> Dict:
        """
        Post-session refinement:
        1. Consistency enforcement across session
        2. Cross-reference validation
        3. Bulk corrections with preview
        """
        if not captions:
            return {"status": "no_captions", "refinements": []}

        refinements = []
        full_text = " ".join(c.get("corrected", c.get("raw", "")) for c in captions)

        # Find inconsistencies (same entity spelled differently)
        word_counts = Counter()
        for cap in captions:
            text = cap.get("corrected", cap.get("raw", ""))
            words = text.split()
            for i, word in enumerate(words):
                if word[0].isupper() if word else False:
                    # Potential proper noun
                    word_counts[word] += 1
                    # Also count 2-word phrases
                    if i + 1 < len(words) and words[i+1][0].isupper() if words[i+1] else False:
                        phrase = f"{word} {words[i+1]}"
                        word_counts[phrase] += 1

        # Find similar terms that might be inconsistent
        terms_list = list(word_counts.keys())
        for i, term1 in enumerate(terms_list):
            for term2 in terms_list[i+1:]:
                ratio = fuzzy_ratio(term1, term2)
                if 0.80 <= ratio < 1.0:  # Similar but not identical
                    count1 = word_counts[term1]
                    count2 = word_counts[term2]

                    # The more frequent one is likely correct
                    if count1 > count2:
                        preferred = term1
                        variant = term2
                        variant_count = count2
                    else:
                        preferred = term2
                        variant = term1
                        variant_count = count1

                    # Only suggest if there are actual occurrences to fix
                    if variant_count > 0:
                        refinements.append({
                            "type": "consistency",
                            "from": variant,
                            "to": preferred,
                            "occurrences": variant_count,
                            "confidence": ratio,
                            "reason": f"'{preferred}' appears {max(count1, count2)}x, '{variant}' appears {min(count1, count2)}x"
                        })

        # Check for known terms that should appear but are misspelled
        for term_key, term_info in self.terms.items():
            term = term_info["term"]
            if term.lower() not in full_text.lower() and len(term) > 5:
                # Term not found, check for close matches
                for word, count in word_counts.items():
                    if len(word) > 3 and fuzzy_ratio(word, term) >= 0.75:
                        refinements.append({
                            "type": "correction",
                            "from": word,
                            "to": term,
                            "occurrences": count,
                            "confidence": fuzzy_ratio(word, term),
                            "reason": f"Possible misspelling of known term '{term}'"
                        })

        return {
            "status": "ok",
            "refinements": refinements,
            "total_captions": len(captions),
            "unique_terms": len(word_counts)
        }

    def apply_refinement(self, captions: List[Dict], from_text: str, to_text: str) -> List[Dict]:
        """Apply a refinement to all captions"""
        refined = []
        for cap in captions:
            new_cap = cap.copy()
            if "corrected" in new_cap:
                new_cap["corrected"] = new_cap["corrected"].replace(from_text, to_text)
            if "raw" in new_cap:
                new_cap["raw_original"] = new_cap.get("raw_original", new_cap["raw"])
            refined.append(new_cap)
        return refined

    def generate_aliases(self, term: str, category: str = "other") -> List[str]:
        """Generate potential aliases/misspellings for a term using AI"""
        aliases = []

        # Rule-based alias generation
        term_lower = term.lower()

        # Common variations
        if " " in term:
            # No space version: "Select Board" -> "selectboard"
            aliases.append(term_lower.replace(" ", ""))
            # Hyphenated: "Town Hall" -> "town-hall"
            aliases.append(term_lower.replace(" ", "-"))

        # Common letter substitutions
        substitutions = [
            ("ie", "y"), ("y", "ie"), ("ee", "ea"), ("ea", "ee"),
            ("ph", "f"), ("f", "ph"), ("ck", "k"), ("k", "ck"),
            ("tion", "shun"), ("c", "k"), ("s", "z")
        ]

        for old, new in substitutions:
            if old in term_lower:
                aliases.append(term_lower.replace(old, new, 1))

        # AI-powered alias generation if available
        if openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model=get_ai_model(),
                    messages=[
                        {
                            "role": "system",
                            "content": "Generate 3-5 common misspellings or alternate forms of the given proper noun. "
                                      "Consider: ASR errors, phonetic spellings, typos, abbreviations. "
                                      "Return ONLY a JSON array of strings, nothing else."
                        },
                        {
                            "role": "user",
                            "content": f"Term: {term}\nCategory: {category}"
                        }
                    ],
                    temperature=0.3,
                    max_tokens=100
                )

                try:
                    ai_aliases = json.loads(response.choices[0].message.content)
                    if isinstance(ai_aliases, list):
                        aliases.extend([a.lower() for a in ai_aliases if isinstance(a, str)])
                except:
                    pass
            except Exception as e:
                print(f"AI alias generation failed: {e}")

        # Remove duplicates and the term itself
        aliases = list(set(a for a in aliases if a != term_lower and len(a) > 2))

        return aliases[:10]  # Limit to 10 aliases

    def reset_session_stats(self):
        """Reset per-session statistics for a fresh session"""
        self.stats = {
            "corrections_applied": 0,
            "captions_processed": 0,
            "semantic_matches": 0,
            "fuzzy_matches": 0,
            "learned_corrections": 0
        }
        self.corrections_log.clear()
        self.pending_suggestions.clear()
        self.recent_context.clear()

    def get_status(self):
        return {
            "enabled": self.enabled,
            "rag_enabled": self.rag_enabled,
            "name": self.name,
            "description": self.description,
            "version": self.VERSION,
            "terms_count": len(self.terms),
            "rules_count": len(self.correction_rules),
            "custom_rules_count": len(self.custom_rules),
            "stats": self.stats,
            "recent_corrections": list(self.corrections_log)[-20:],
            "pending_suggestions": self.pending_suggestions[-10:],
            "embeddings_available": self.embeddings.is_available(),
            "embeddings_type": self.embeddings.model_type,
            "learned_patterns_count": len(self.learner.learned_patterns)
        }

    def get_terms_list(self):
        return [
            {
                "term": info["term"],
                "category": info.get("category", "other"),
                "aliases": info.get("aliases", []),
                "source": info.get("source", "manual"),
                "context": info.get("context", ""),
                "metadata": info.get("metadata", {})
            }
            for info in self.terms.values()
        ]

    def set_enabled(self, enabled):
        self.enabled = enabled
        self._save_state()

    def set_rag_enabled(self, enabled):
        self.rag_enabled = enabled
        self._save_state()

    def add_term(self, term, category="other", aliases=None, source="manual", context="", metadata=None):
        """Add a term with optional metadata for RAG"""
        key = term.lower()

        # Auto-generate aliases if none provided
        if not aliases:
            aliases = self.generate_aliases(term, category)

        self.terms[key] = {
            "term": term,
            "category": category,
            "aliases": aliases or [],
            "source": source,
            "context": context,
            "metadata": metadata or {},
            "added": datetime.now().isoformat()
        }

        # Pre-compute embeddings for new term
        if self.embeddings.is_available():
            self.embeddings.get_embedding(term)
            for alias in (aliases or []):
                self.embeddings.get_embedding(alias)

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

    def accept_suggestion(self, from_text: str, to_text: str):
        """Accept a pending suggestion and learn from it"""
        self.learner.learn(from_text, to_text)
        self.pending_suggestions = [
            s for s in self.pending_suggestions
            if not (s["from"] == from_text and s["to"] == to_text)
        ]

    def reject_suggestion(self, from_text: str, to_text: str):
        """Reject a suggestion (could add to negative list)"""
        self.pending_suggestions = [
            s for s in self.pending_suggestions
            if not (s["from"] == from_text and s["to"] == to_text)
        ]

    def add_defaults(self):
        """Add default Brookline terms with auto-generated aliases"""
        defaults = {
            "person": [
                ("Bernard Greene", ["bernard green", "bernie greene", "bernard green"]),
                ("Heather Hamilton", []),
                ("John VanScoyoc", ["van scoyoc", "vanscoyoc", "john van scoyoc"]),
                ("Paul Warren", []),
                ("David Pearlman", ["dave pearlman"]),
                ("Todd Kirrane", ["tod kirrane", "tod kiran"]),
                ("Mark Zarrillo", ["zarrillo", "zarillo"]),
                ("Ben Franco", []),
                ("Melissa Goff", []),
                ("Linus Guillory", ["linus guillary"]),
                ("Stephen Walter", ["steve walter", "steven walter"]),
            ],
            "place": [
                ("Brookline", ["brooklyn", "brook line", "brooklynn"]),
                ("Coolidge Corner", ["coolidge", "coolidge corners"]),
                ("Brookline Village", ["brookline villiage"]),
                ("Washington Square", []),
                ("Chestnut Hill", []),
                ("Town Hall", ["townhall", "town haul"]),
                ("Brookline High School", ["bhs", "brookline high"]),
                ("Harvard Street", []),
                ("Beacon Street", []),
                ("Washington Street", []),
                ("Larz Anderson Park", ["larz anderson", "lars anderson", "larz andersen"]),
            ],
            "organization": [
                ("Select Board", ["selectboard", "board of selectmen", "select bored", "selectmen"]),
                ("Town Meeting", ["town meating"]),
                ("Advisory Committee", ["advisory commitee"]),
                ("School Committee", ["school commitee"]),
                ("Planning Board", []),
                ("Zoning Board of Appeals", ["zba", "zoning board"]),
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
                        "source": "default",
                        "context": "Brookline, MA municipal context",
                        "metadata": {}
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
            "learned_patterns": self.learner.get_all_patterns(),
            "stats": {
                "terms_count": len(self.terms),
                "rules_count": len(self.correction_rules),
                "learned_patterns_count": len(self.learner.learned_patterns)
            },
            "settings": {
                "auto_correct_threshold": self.AUTO_CORRECT_THRESHOLD,
                "suggest_threshold": self.SUGGEST_THRESHOLD,
                "fuzzy_threshold": self.FUZZY_THRESHOLD
            }
        }

    def import_engine(self, data: dict):
        """Import engine from JSON"""
        version = data.get("version", "1.0")
        if version != self.VERSION:
            print(f"Info: Importing engine version {version} into v{self.VERSION}")

        self.name = data.get("name", "Imported Engine")
        self.description = data.get("description", "")
        self.terms = data.get("terms", {})
        self.custom_rules = data.get("custom_rules", [])

        # Import learned patterns if available
        if "learned_patterns" in data:
            self.learner.learned_patterns = data["learned_patterns"]
            self.learner._save_patterns()

        self._rebuild_rules()
        self._save_state()

        # Pre-compute embeddings for imported terms
        if self.embeddings.is_available():
            for key, info in self.terms.items():
                self.embeddings.get_embedding(info["term"])

        return {"status": "ok", "terms_count": len(self.terms), "version_imported": version}


# =============================================================================
# KNOWLEDGE BASE - Document ingestion and search
# =============================================================================

class KnowledgeBase:
    """
    Manages uploaded documents for entity extraction and context enhancement.
    Supports PDF, DOCX, and plain text files.
    """

    def __init__(self, knowledge_dir: Path, embeddings_manager: EmbeddingsManager):
        self.knowledge_dir = knowledge_dir
        self.embeddings = embeddings_manager
        self.documents = {}  # doc_id -> {filename, text, entities, chunks, added}
        self.index_file = knowledge_dir / "knowledge_index.json"
        self._load_index()

    def _load_index(self):
        if self.index_file.exists():
            try:
                with open(self.index_file) as f:
                    self.documents = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load knowledge index: {e}")

    def _save_index(self):
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.documents, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save knowledge index: {e}")

    def extract_text_from_pdf(self, file_path: Path) -> tuple:
        """Extract text from PDF file. Returns (text, error_message)"""
        # Try PyPDF2 first
        try:
            import PyPDF2
            text_parts = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
            result = "\n".join(text_parts).strip()
            if result:
                return result, None
        except ImportError:
            pass
        except Exception as e:
            print(f"PyPDF2 extraction error: {e}")

        # Try pdfplumber as fallback
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            result = "\n".join(text_parts).strip()
            if result:
                return result, None
        except ImportError:
            pass
        except Exception as e:
            print(f"pdfplumber extraction error: {e}")

        # Check if any PDF library is available
        try:
            import PyPDF2
            pypdf_available = True
        except ImportError:
            pypdf_available = False

        try:
            import pdfplumber
            pdfplumber_available = True
        except ImportError:
            pdfplumber_available = False

        if not pypdf_available and not pdfplumber_available:
            return "", "PDF libraries not installed. Run: pip3 install PyPDF2 pdfplumber"

        return "", "Could not extract text from PDF. The file may be scanned/image-based."

    def extract_text_from_url(self, url: str) -> tuple:
        """Extract text from a URL. Returns (text, error_message)"""
        try:
            import urllib.request
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.skip_tags = {'script', 'style', 'nav', 'footer', 'header'}
                    self.current_tag = None

                def handle_starttag(self, tag, attrs):
                    self.current_tag = tag

                def handle_endtag(self, tag):
                    self.current_tag = None

                def handle_data(self, data):
                    if self.current_tag not in self.skip_tags:
                        text = data.strip()
                        if text:
                            self.text.append(text)

            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                html = response.read().decode('utf-8', errors='ignore')

            parser = TextExtractor()
            parser.feed(html)
            text = ' '.join(parser.text)

            if len(text) < 100:
                return "", "Could not extract meaningful text from URL"

            return text, None
        except Exception as e:
            return "", f"Error fetching URL: {str(e)}"

    def extract_youtube_transcript(self, url: str) -> tuple:
        """Extract transcript from YouTube video. Returns (text, error_message)"""
        import re

        # Extract video ID from URL
        video_id = None
        patterns = [
            r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'(?:embed/)([a-zA-Z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                break

        if not video_id:
            return "", "Could not extract YouTube video ID from URL"

        # Try youtube_transcript_api (v1.x uses .fetch() instead of .get_transcript())
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            fetched = YouTubeTranscriptApi().fetch(video_id)
            # New API returns FetchedTranscript with snippets attribute
            text = ' '.join([snippet.text for snippet in fetched.snippets])
            return text, None
        except ImportError:
            pass
        except Exception as e:
            print(f"YouTube transcript API error: {e}")

        # Fallback: Try to fetch captions via direct YouTube API
        try:
            import urllib.request
            import xml.etree.ElementTree as ET

            # Fetch video page to get caption tracks
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            req = urllib.request.Request(video_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept-Language': 'en-US,en;q=0.9'
            })
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8', errors='ignore')

            # Look for captionTracks in ytInitialPlayerResponse
            caption_match = re.search(r'"captionTracks":\s*\[(.*?)\]', html)
            if caption_match:
                caption_tracks = caption_match.group(1)
                base_url_match = re.search(r'"baseUrl":\s*"([^"]+)"', caption_tracks)
                if base_url_match:
                    caption_url = base_url_match.group(1).replace('\\u0026', '&')

                    # Fetch the actual caption content
                    caption_req = urllib.request.Request(caption_url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    with urllib.request.urlopen(caption_req, timeout=15) as caption_response:
                        caption_content = caption_response.read().decode('utf-8', errors='ignore')

                    # Parse XML format captions
                    if caption_content.strip().startswith('<?xml') or caption_content.strip().startswith('<transcript'):
                        root = ET.fromstring(caption_content)
                        text_parts = []
                        for text_elem in root.findall('.//text'):
                            text = text_elem.text
                            if text:
                                text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                                text = text.replace('&#39;', "'").replace('&quot;', '"')
                                text_parts.append(text)
                        if text_parts:
                            print(f"✅ Fetched YouTube transcript via direct API for knowledge base")
                            return ' '.join(text_parts), None
                    else:
                        # Try JSON format
                        try:
                            caption_json = json.loads(caption_content)
                            events = caption_json.get('events', [])
                            text_parts = []
                            for event in events:
                                segs = event.get('segs', [])
                                for seg in segs:
                                    text = seg.get('utf8', '')
                                    if text and text.strip():
                                        text_parts.append(text)
                            if text_parts:
                                print(f"✅ Fetched YouTube transcript via direct API (JSON) for knowledge base")
                                return ' '.join(text_parts), None
                        except:
                            pass
        except Exception as e:
            print(f"YouTube direct API fallback error: {e}")

        # Check if youtube_transcript_api is available for a better error message
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            return "", f"Could not fetch transcript for this video. The video may not have captions available, or captions may be disabled."
        except ImportError:
            return "", "Could not fetch YouTube transcript. Install: pip3 install youtube-transcript-api"

    def extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            try:
                from docx import Document
                doc = Document(file_path)
                return "\n".join(para.text for para in doc.paragraphs)
            except ImportError:
                pass

            return ""
        except Exception as e:
            print(f"DOCX extraction error: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for embedding"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using AI or regex fallback"""
        import re

        # First try AI extraction if available
        if openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model=get_ai_model(),
                    messages=[
                        {
                            "role": "system",
                            "content": "Extract all proper nouns from the text. Categorize them as people, places, or organizations. "
                                      "Return ONLY a JSON object: {\"people\": [...], \"places\": [...], \"organizations\": [...]}"
                        },
                        {
                            "role": "user",
                            "content": text[:6000]
                        }
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )

                try:
                    result = json.loads(response.choices[0].message.content)
                    # Validate structure
                    if isinstance(result, dict) and any(result.get(k) for k in ['people', 'places', 'organizations']):
                        return result
                except:
                    pass

            except Exception as e:
                print(f"AI entity extraction error: {e}")

        # Fallback: Use regex to extract capitalized multi-word phrases
        entities = {"people": [], "places": [], "organizations": []}

        # Find capitalized phrases (2+ words starting with capitals)
        capitalized_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)

        # Common person name patterns (first last or title first last)
        person_titles = ['Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Representative', 'Senator', 'Mayor', 'Chair', 'Director', 'Commissioner']
        # Organization keywords
        org_keywords = ['Committee', 'Board', 'Department', 'Office', 'Council', 'Commission', 'Authority', 'Association', 'Foundation', 'Institute', 'Company', 'Corp', 'Inc', 'LLC', 'Group', 'Agency']
        # Place keywords
        place_keywords = ['Street', 'Avenue', 'Road', 'Park', 'Center', 'Building', 'Hall', 'Square', 'Village', 'Town', 'City', 'County', 'State']

        seen = set()
        for phrase in capitalized_phrases:
            phrase_clean = phrase.strip()
            if phrase_clean in seen or len(phrase_clean) < 4:
                continue
            seen.add(phrase_clean)

            # Categorize
            if any(kw in phrase_clean for kw in org_keywords):
                entities["organizations"].append(phrase_clean)
            elif any(kw in phrase_clean for kw in place_keywords):
                entities["places"].append(phrase_clean)
            elif any(phrase_clean.startswith(title) for title in person_titles):
                entities["people"].append(phrase_clean)
            elif len(phrase_clean.split()) == 2:  # Two word phrases are likely names
                entities["people"].append(phrase_clean)

        # Also extract single capitalized words that look like proper nouns
        single_caps = re.findall(r'\b([A-Z][a-z]{2,})\b', text)
        name_like = [w for w in single_caps if w not in ['The', 'This', 'That', 'What', 'When', 'Where', 'Which', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

        # Add unique single words as potential entities
        for name in set(name_like):
            if name not in seen and len(name) > 3:
                # Could be a person's name or place
                entities["people"].append(name)

        # Limit to reasonable numbers
        entities["people"] = list(set(entities["people"]))[:50]
        entities["places"] = list(set(entities["places"]))[:30]
        entities["organizations"] = list(set(entities["organizations"]))[:30]

        return entities

    def add_document(self, filename: str, content: bytes = None, text: str = None) -> Dict:
        """Add a document to the knowledge base"""
        doc_id = str(uuid.uuid4())[:8]
        file_path = self.knowledge_dir / f"{doc_id}_{filename}"

        # Save file if content provided
        if content:
            with open(file_path, 'wb') as f:
                f.write(content)

        # Extract text based on file type
        extracted_text = text or ""
        extraction_error = None
        if not extracted_text and content:
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if ext == 'pdf':
                extracted_text, extraction_error = self.extract_text_from_pdf(file_path)
            elif ext in ['docx', 'doc']:
                extracted_text = self.extract_text_from_docx(file_path)
                if not extracted_text:
                    extraction_error = "Could not extract text from DOCX. Install: pip3 install python-docx"
            elif ext in ['txt', 'md', 'csv']:
                extracted_text = content.decode('utf-8', errors='ignore')

        if not extracted_text:
            return {"error": extraction_error or "Could not extract text from document"}

        # Chunk text for embeddings
        chunks = self.chunk_text(extracted_text)

        # Extract entities
        entities = self.extract_entities_from_text(extracted_text)

        # Create embeddings for chunks
        if self.embeddings.is_available():
            for i, chunk in enumerate(chunks[:50]):  # Limit chunks for performance
                self.embeddings.get_embedding(chunk)

        # Store document info
        self.documents[doc_id] = {
            "filename": filename,
            "text": extracted_text[:50000],  # Limit stored text
            "chunks": chunks[:50],
            "entities": entities,
            "word_count": len(extracted_text.split()),
            "added": datetime.now().isoformat()
        }

        self._save_index()

        return {
            "status": "ok",
            "doc_id": doc_id,
            "filename": filename,
            "word_count": len(extracted_text.split()),
            "chunks": len(chunks),
            "entities": {k: len(v) for k, v in entities.items()}
        }

    def remove_document(self, doc_id: str):
        """Remove a document from the knowledge base"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._save_index()
            return {"status": "ok"}
        return {"error": "Document not found"}

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search knowledge base using semantic similarity"""
        if not self.embeddings.is_available():
            return []

        query_embedding = self.embeddings.get_embedding(query)
        if not query_embedding:
            return []

        results = []
        for doc_id, doc in self.documents.items():
            for i, chunk in enumerate(doc.get("chunks", [])):
                chunk_embedding = self.embeddings.get_embedding(chunk)
                if chunk_embedding:
                    similarity = self.embeddings.cosine_similarity(query_embedding, chunk_embedding)
                    if similarity > 0.5:
                        results.append({
                            "doc_id": doc_id,
                            "filename": doc["filename"],
                            "chunk_index": i,
                            "text": chunk[:300],
                            "similarity": round(similarity, 3)
                        })

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def get_all_entities(self) -> Dict[str, List[str]]:
        """Get all entities from all documents"""
        all_entities = {"people": set(), "places": set(), "organizations": set()}

        for doc in self.documents.values():
            entities = doc.get("entities", {})
            for category in all_entities:
                all_entities[category].update(entities.get(category, []))

        return {k: list(v) for k, v in all_entities.items()}

    def sync_entities_to_engine(self, engine: CaptionEngine) -> Dict:
        """Add all extracted entities to the caption engine"""
        all_entities = self.get_all_entities()
        added = 0

        for person in all_entities["people"]:
            if person and len(person) > 2:
                engine.add_term(person, "person", source="knowledge_base")
                added += 1

        for place in all_entities["places"]:
            if place and len(place) > 2:
                engine.add_term(place, "place", source="knowledge_base")
                added += 1

        for org in all_entities["organizations"]:
            if org and len(org) > 2:
                engine.add_term(org, "organization", source="knowledge_base")
                added += 1

        return {"status": "ok", "terms_added": added}

    def get_status(self) -> Dict:
        return {
            "document_count": len(self.documents),
            "documents": [
                {
                    "id": doc_id,
                    "filename": doc["filename"],
                    "word_count": doc.get("word_count", 0),
                    "entities": {k: len(v) for k, v in doc.get("entities", {}).items()},
                    "added": doc.get("added", "")
                }
                for doc_id, doc in self.documents.items()
            ],
            "total_entities": {k: len(v) for k, v in self.get_all_entities().items()}
        }


# =============================================================================
# SESSION MANAGER - Recording and export
# =============================================================================

class SessionManager:
    """Manages captioning sessions with timestamps, audio recording, and export"""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.current_session = None
        self.captions = []  # [{timestamp, raw, corrected, corrections}]
        self.start_time = None
        self.is_recording = False

        # Audio recording
        self.audio_recording = False
        self.audio_buffer = []
        self.audio_stream = None
        self.audio_thread = None
        self.sample_rate = 16000

    def start_session(self, name=None, record_audio=True, audio_device=None, **kwargs):
        """Start a new captioning session with audio recording by default"""
        session_id = str(uuid.uuid4())[:8]
        self.current_session = {
            "id": session_id,
            "name": name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "started": datetime.now().isoformat(),
            "mode": "browser",
            "has_audio": False
        }
        self.captions = []
        self.start_time = datetime.now()
        self.is_recording = True

        # Start audio recording by default (for Whisper second pass)
        if record_audio and AUDIO_AVAILABLE and NUMPY_AVAILABLE:
            try:
                # Convert device_id to int if provided as string
                dev_id = int(audio_device) if audio_device is not None else None
                self._start_audio_recording(session_id, device_id=dev_id)
                self.current_session["has_audio"] = True
            except Exception as e:
                print(f"⚠️ Could not start audio recording: {e}")

        return self.current_session

    def _start_audio_recording(self, session_id: str, device_id=None):
        """Start recording audio to file"""
        try:
            # Clean up any existing stream first to prevent bus error
            if hasattr(self, 'audio_stream') and self.audio_stream is not None:
                try:
                    self.audio_stream.stop()
                    self.audio_stream.close()
                except:
                    pass
                self.audio_stream = None

            self.audio_buffer = []
            self.audio_recording = True

            def audio_callback(indata, frames, time_info, status):
                if self.audio_recording:
                    self.audio_buffer.append(indata.copy())

            self.audio_stream = sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=self.sample_rate,
                callback=audio_callback,
                blocksize=int(self.sample_rate * 0.1)
            )
            self.audio_stream.start()
            print(f"🎙️ Audio recording started")
        except Exception as e:
            print(f"⚠️ Audio recording failed: {e}")
            self.audio_recording = False
            if hasattr(self, 'audio_stream'):
                self.audio_stream = None

    def _stop_audio_recording(self, session_id: str):
        """Stop audio recording and save to file"""
        if not self.audio_recording:
            return None

        try:
            self.audio_recording = False
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None

            if self.audio_buffer and NUMPY_AVAILABLE:
                audio_data = np.concatenate(self.audio_buffer).flatten()
                audio_file = AUDIO_DIR / f"{session_id}.wav"

                # Save as WAV file
                try:
                    import wave
                    with wave.open(str(audio_file), 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(self.sample_rate)
                        audio_int16 = (audio_data * 32767).astype(np.int16)
                        wf.writeframes(audio_int16.tobytes())
                    print(f"🎙️ Audio saved: {audio_file}")
                    return str(audio_file)
                except Exception as e:
                    print(f"⚠️ Failed to save audio: {e}")

            self.audio_buffer = []
        except Exception as e:
            print(f"⚠️ Error stopping audio recording: {e}")

        return None
    
    def add_caption(self, raw: str, corrected: str, corrections: list):
        """Add a caption to the current session with light deduplication"""
        if not self.is_recording:
            return

        corrected_clean = corrected.strip()
        if not corrected_clean:
            return

        # Light deduplication: only skip exact duplicates of the LAST caption
        if self.captions:
            last_text = (self.captions[-1].get('corrected') or self.captions[-1].get('raw', '')).strip()
            if corrected_clean.lower() == last_text.lower():
                return  # Skip exact duplicate

        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.captions.append({
            "timestamp": elapsed,
            "time_str": self._format_time(elapsed),
            "raw": raw,
            "corrected": corrected,
            "corrections": corrections,
            "added": datetime.now().isoformat()
        })

        print(f"   💾 Caption saved: [{self._format_time(elapsed)}] {corrected[:50]}...")
    
    def stop_session(self):
        """Stop the current session and save audio if recorded"""
        if not self.current_session:
            return None

        session_id = self.current_session["id"]

        # Stop audio recording if active
        if self.audio_recording:
            audio_file = self._stop_audio_recording(session_id)
            if audio_file:
                self.current_session["audio_file"] = audio_file

        self.current_session["ended"] = datetime.now().isoformat()
        self.current_session["duration"] = (datetime.now() - self.start_time).total_seconds()
        self.current_session["caption_count"] = len(self.captions)
        self.is_recording = False

        # Save session
        session_file = self.sessions_dir / f"{session_id}.json"
        with open(session_file, "w") as f:
            json.dump({
                "session": self.current_session,
                "captions": self.captions
            }, f, indent=2)

        return self.current_session

    def get_audio_file(self) -> Optional[str]:
        """Get path to audio file for current/last session"""
        if self.current_session and self.current_session.get("audio_file"):
            return str(AUDIO_DIR / self.current_session["audio_file"])
        return None

    def has_audio(self) -> bool:
        """Check if current session has audio recording"""
        return bool(self.current_session and self.current_session.get("audio_recorded"))
    
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

    def list_sessions(self):
        """List all saved sessions"""
        sessions = []
        for session_file in sorted(self.sessions_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(session_file) as f:
                    data = json.load(f)
                    session = data.get("session", {})
                    session["id"] = session.get("id", session_file.stem)
                    session["caption_count"] = len(data.get("captions", []))
                    session["file"] = session_file.name
                    # Calculate total words
                    total_words = sum(
                        len((c.get("corrected") or c.get("raw", "")).split())
                        for c in data.get("captions", [])
                    )
                    session["word_count"] = total_words
                    # Count corrections
                    total_corrections = sum(len(c.get("corrections", [])) for c in data.get("captions", []))
                    session["correction_count"] = total_corrections
                    sessions.append(session)
            except Exception as e:
                print(f"Error reading session {session_file}: {e}")
        return sessions

    def load_session(self, session_id: str):
        """Load a specific saved session"""
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            return None
        try:
            with open(session_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None

    def delete_session(self, session_id: str):
        """Delete a saved session"""
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            # Also delete audio file if exists
            audio_file = AUDIO_DIR / f"{session_id}.wav"
            if audio_file.exists():
                audio_file.unlink()
            return True
        return False

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
# WHISPER ENGINE - Ultra-Low Latency Real-Time Captioning
# =============================================================================

class WhisperEngine:
    """
    Ultra-optimized local speech-to-text using faster-whisper.

    Key optimizations for INSTANT captions:
    1. Continuous streaming with overlapping windows
    2. Energy-based VAD that's fast and reliable
    3. Immediate transcription on speech detection (no waiting)
    4. Rolling buffer with incremental output
    5. Aggressive settings for minimum latency
    """

    MODEL_INFO = {
        "tiny": {"size": "75 MB", "speed": "Instant", "quality": "Basic", "latency": "~0.3s"},
        "tiny.en": {"size": "75 MB", "speed": "Instant", "quality": "Good (English)", "latency": "~0.3s"},
        "base": {"size": "150 MB", "speed": "Very Fast", "quality": "Good", "latency": "~0.5s"},
        "base.en": {"size": "150 MB", "speed": "Very Fast", "quality": "Better (English)", "latency": "~0.5s"},
        "small": {"size": "500 MB", "speed": "Fast", "quality": "Better", "latency": "~0.8s"},
        "small.en": {"size": "500 MB", "speed": "Fast", "quality": "Great (English)", "latency": "~0.8s"},
        "medium": {"size": "1.5 GB", "speed": "Medium", "quality": "Great", "latency": "~1.5s"},
        "medium.en": {"size": "1.5 GB", "speed": "Medium", "quality": "Excellent (English)", "latency": "~1.5s"},
        "large-v3": {"size": "3 GB", "speed": "Slow", "quality": "Best", "latency": "~2.5s"},
    }

    def __init__(self):
        self.model = None
        self.model_name = None
        self.is_running = False
        self.text_callback = None
        self.sample_rate = 16000
        self.stream = None
        self.thread = None
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()

        # AGGRESSIVE settings for ultra-low latency
        self.min_chunk_duration = 0.4   # 400ms minimum - very fast!
        self.max_chunk_duration = 1.5   # Force output after 1.5s max
        self.silence_threshold = 0.008  # Lower threshold = more sensitive
        self.speech_pad_ms = 100        # Less padding = faster

        # GPU acceleration settings
        self.device = "cpu"
        self.compute_type = "int8"
        self.gpu_available = False

        # Performance tracking
        self.avg_latency_ms = 0
        self.transcription_count = 0
        self.last_output_time = 0

        # Streaming state
        self.previous_text = ""
        self.speech_energy_history = deque(maxlen=10)  # Track energy levels

        # LATENCY TRACKING - detect when we're falling behind
        self.audio_start_time = 0           # When audio capture started
        self.total_audio_received_ms = 0    # Total audio duration received
        self.total_audio_processed_ms = 0   # Total audio duration processed
        self.current_lag_ms = 0             # How far behind we are
        self.dropped_audio_ms = 0           # Total audio dropped to catch up
        self.max_allowed_lag_ms = 2000      # Max lag before aggressive catch-up
        self.catch_up_events = 0            # Number of times we had to catch up

        self._detect_gpu()

    def _detect_gpu(self):
        """Detect if GPU acceleration is available (CUDA or Apple MPS)"""
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.device = "cuda"
                self.compute_type = "float16"
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 NVIDIA GPU detected: {gpu_name}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Apple Silicon (M1/M2/M3) - faster-whisper doesn't support MPS directly
                # but we can still use optimized CPU with more threads
                self.gpu_available = False  # MPS not supported by faster-whisper
                self.device = "cpu"
                self.compute_type = "int8"  # int8 is fastest on Apple Silicon
                print("🍎 Apple Silicon detected - using optimized CPU (int8)")
                print("   Note: faster-whisper doesn't support MPS, but int8 is fast on M-series")
            else:
                print("💻 No GPU detected - using CPU (int8 quantized)")
        except ImportError:
            print("💻 PyTorch not available - using CPU (int8)")
        except Exception as e:
            print(f"💻 GPU detection failed: {e}")

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
            "models": self.MODEL_INFO,
            "gpu_available": self.gpu_available,
            "device": self.device,
            "compute_type": self.compute_type,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "min_chunk_ms": int(self.min_chunk_duration * 1000),
            "max_chunk_ms": int(self.max_chunk_duration * 1000),
            # Latency tracking
            "current_lag_ms": round(self.current_lag_ms, 0),
            "dropped_audio_ms": round(self.dropped_audio_ms, 0),
            "catch_up_events": self.catch_up_events,
            "is_behind": self.current_lag_ms > 1000,  # True if > 1 second behind
            "latency_stats": {
                "current_lag_ms": round(self.current_lag_ms, 0),
                "dropped_audio_ms": round(self.dropped_audio_ms, 0),
                "catch_up_events": self.catch_up_events,
                "max_allowed_lag_ms": self.max_allowed_lag_ms
            }
        }

    def load_model(self, model_name="tiny.en"):
        """Load Whisper model - defaults to tiny.en for fastest response"""
        if not WHISPER_AVAILABLE:
            return {"error": "faster-whisper not installed. Run: pip3 install faster-whisper"}

        try:
            # Detect CPU cores for optimal threading
            import os
            cpu_count = os.cpu_count() or 4
            # Use more threads on Apple Silicon (it's efficient with many threads)
            cpu_threads = min(cpu_count, 8)  # Cap at 8 for stability

            print(f"🔄 Loading Whisper model '{model_name}' on {self.device} ({cpu_threads} threads)...")

            self.model = WhisperModel(
                model_name,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=cpu_threads,
                num_workers=1  # Single worker for lowest latency
            )
            self.model_name = model_name

            device_info = "GPU" if self.gpu_available else f"CPU ({cpu_threads} threads)"
            print(f"✅ Model '{model_name}' ready on {device_info}!")

            return {
                "status": "ok",
                "model": model_name,
                "device": self.device,
                "gpu_accelerated": self.gpu_available,
                "cpu_threads": cpu_threads
            }
        except Exception as e:
            if self.device == "cuda":
                print(f"⚠️ GPU failed, trying CPU: {e}")
                self.device = "cpu"
                self.compute_type = "int8"
                return self.load_model(model_name)
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
        """Callback for audio input with latency tracking"""
        # Track how much audio we've received
        audio_duration_ms = (frames / self.sample_rate) * 1000
        self.total_audio_received_ms += audio_duration_ms

        with self.buffer_lock:
            self.audio_buffer.append({
                "audio": indata.copy(),
                "timestamp": time.time(),
                "duration_ms": audio_duration_ms
            })

    def _get_audio_energy(self, audio: np.ndarray) -> float:
        """Fast energy calculation for VAD"""
        return np.sqrt(np.mean(audio ** 2))

    def _has_speech(self, audio: np.ndarray) -> bool:
        """Fast energy-based speech detection with adaptive threshold"""
        energy = self._get_audio_energy(audio)
        self.speech_energy_history.append(energy)

        # Adaptive threshold based on recent history
        if len(self.speech_energy_history) >= 5:
            avg_energy = sum(self.speech_energy_history) / len(self.speech_energy_history)
            threshold = max(self.silence_threshold, avg_energy * 0.3)
        else:
            threshold = self.silence_threshold

        return energy > threshold

    def _transcription_loop(self):
        """
        Ultra-fast transcription loop optimized for minimum latency.

        Strategy:
        - Transcribe immediately when we have enough audio with speech
        - Don't wait for silence - output as soon as possible
        - Use fastest possible Whisper settings
        - CATCH UP: Drop old audio if falling behind real-time
        """
        min_samples = int(self.sample_rate * self.min_chunk_duration)
        max_samples = int(self.sample_rate * self.max_chunk_duration)

        # Use a numpy array for accumulated audio (more efficient)
        audio_accumulator = np.array([], dtype=np.float32)
        oldest_audio_timestamp = None  # Track age of oldest audio in accumulator
        has_active_speech = False
        speech_start_time = 0
        last_speech_time = 0

        while self.is_running:
            try:
                time.sleep(0.02)  # 20ms polling for responsiveness

                # Get buffered audio
                with self.buffer_lock:
                    if not self.audio_buffer:
                        continue
                    chunks = self.audio_buffer.copy()
                    self.audio_buffer = []

                now = time.time()

                # === LATENCY CHECK: Are we falling behind? ===
                if chunks:
                    oldest_chunk_time = chunks[0]["timestamp"]
                    buffer_lag_ms = (now - oldest_chunk_time) * 1000

                    # If buffer is too old, we're behind - need to catch up
                    if buffer_lag_ms > self.max_allowed_lag_ms:
                        # Calculate how much to drop
                        dropped_ms = 0
                        chunks_to_keep = []

                        for chunk in reversed(chunks):
                            chunk_age_ms = (now - chunk["timestamp"]) * 1000
                            if chunk_age_ms <= self.max_allowed_lag_ms / 2:
                                chunks_to_keep.insert(0, chunk)
                            else:
                                dropped_ms += chunk["duration_ms"]

                        if dropped_ms > 0:
                            self.dropped_audio_ms += dropped_ms
                            self.catch_up_events += 1
                            print(f"⚡ CATCH-UP: Dropped {dropped_ms:.0f}ms of old audio (was {buffer_lag_ms:.0f}ms behind)")

                            # Also clear the accumulator - it has old audio too
                            audio_accumulator = np.array([], dtype=np.float32)
                            oldest_audio_timestamp = None
                            has_active_speech = False

                            chunks = chunks_to_keep

                # Track oldest audio timestamp
                if chunks and oldest_audio_timestamp is None:
                    oldest_audio_timestamp = chunks[0]["timestamp"]

                # Flatten and accumulate audio chunks
                for chunk in chunks:
                    audio_data = chunk["audio"]
                    flat_chunk = audio_data.flatten().astype(np.float32)
                    audio_accumulator = np.concatenate([audio_accumulator, flat_chunk])
                    self.total_audio_processed_ms += chunk["duration_ms"]

                total_samples = len(audio_accumulator)
                if total_samples == 0:
                    continue

                # Update current lag estimate
                if oldest_audio_timestamp:
                    self.current_lag_ms = (now - oldest_audio_timestamp) * 1000

                # Check for speech in recent 200ms
                check_samples = min(int(self.sample_rate * 0.2), total_samples)
                recent_audio = audio_accumulator[-check_samples:]
                is_speaking = self._has_speech(recent_audio)

                if is_speaking:
                    if not has_active_speech:
                        has_active_speech = True
                        speech_start_time = now
                    last_speech_time = now

                # Decide when to transcribe
                should_transcribe = False
                reason = ""

                if has_active_speech:
                    speech_duration = now - speech_start_time
                    silence_duration = now - last_speech_time

                    # Transcribe if:
                    # 1. We have minimum audio AND silence detected (speech ended)
                    if total_samples >= min_samples and silence_duration > 0.25:
                        should_transcribe = True
                        reason = "speech_ended"
                    # 2. Max duration reached (force output for long speech)
                    elif total_samples >= max_samples:
                        should_transcribe = True
                        reason = "max_duration"
                    # 3. Been speaking for a while, output partial
                    elif total_samples >= min_samples and speech_duration > 0.8:
                        should_transcribe = True
                        reason = "partial_output"

                if should_transcribe and self.model:
                    transcription_start = time.time()

                    # FASTEST possible settings
                    segments, _ = self.model.transcribe(
                        audio_accumulator,
                        beam_size=1,              # Fastest - greedy decoding
                        best_of=1,
                        language="en",
                        vad_filter=False,         # We already did VAD
                        condition_on_previous_text=False,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                        no_speech_threshold=0.5,
                        without_timestamps=True,
                        temperature=0.0,          # Deterministic = faster
                    )

                    text = " ".join(seg.text for seg in segments).strip()

                    # Track performance
                    transcription_time = (time.time() - transcription_start) * 1000
                    self.transcription_count += 1
                    self.avg_latency_ms = (
                        (self.avg_latency_ms * (self.transcription_count - 1) + transcription_time)
                        / self.transcription_count
                    )

                    # Output text
                    if text and self.text_callback:
                        self.text_callback(text)
                        self.last_output_time = now

                    # Reset state
                    if reason == "speech_ended":
                        audio_accumulator = np.array([], dtype=np.float32)
                        oldest_audio_timestamp = None
                        has_active_speech = False
                        self.current_lag_ms = 0  # We're caught up
                    elif reason == "max_duration" or reason == "partial_output":
                        # Keep some audio for context continuity
                        keep_samples = int(self.sample_rate * 0.3)
                        if total_samples > keep_samples:
                            audio_accumulator = audio_accumulator[-keep_samples:].copy()
                            # Update timestamp estimate for kept audio
                            oldest_audio_timestamp = now - (keep_samples / self.sample_rate)
                        # Stay in speech mode

            except Exception as e:
                if self.is_running:
                    print(f"⚠️ Transcription error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.3)
                    audio_accumulator = np.array([], dtype=np.float32)
                    has_active_speech = False

    def start(self, device_id=None, callback=None):
        if not AUDIO_AVAILABLE or not NUMPY_AVAILABLE:
            return {"error": "Missing packages: sounddevice and numpy required"}
        if not self.model:
            return {"error": "No model loaded. Load a model first."}
        if self.is_running:
            return {"error": "Already running"}

        # Clean up existing stream
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None

        self.text_callback = callback
        self.is_running = True
        self.audio_buffer = []
        self.avg_latency_ms = 0
        self.transcription_count = 0
        self.speech_energy_history.clear()

        # Reset latency tracking
        self.audio_start_time = time.time()
        self.total_audio_received_ms = 0
        self.total_audio_processed_ms = 0
        self.current_lag_ms = 0
        self.dropped_audio_ms = 0
        self.catch_up_events = 0

        try:
            # Small blocksize for responsiveness
            self.stream = sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.02)  # 20ms blocks!
            )
            self.stream.start()
            self.thread = threading.Thread(target=self._transcription_loop, daemon=True)
            self.thread.start()

            print(f"🎤 Whisper listening (ultra-low latency mode)...")
            print(f"   Min chunk: {int(self.min_chunk_duration*1000)}ms, Max: {int(self.max_chunk_duration*1000)}ms")

            return {
                "status": "started",
                "device": self.device,
                "gpu_accelerated": self.gpu_available,
                "min_latency_ms": int(self.min_chunk_duration * 1000)
            }
        except Exception as e:
            self.is_running = False
            self.stream = None
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

        return {
            "status": "stopped",
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "transcription_count": self.transcription_count
        }

    def transcribe_audio(self, audio_data, high_accuracy=False):
        """Transcribe audio data for post-processing"""
        if not self.model:
            return {"error": "No model loaded"}

        try:
            beam_size = 10 if high_accuracy else 5
            segments, _ = self.model.transcribe(
                audio_data,
                beam_size=beam_size,
                best_of=5 if high_accuracy else 2,
                language="en",
                vad_filter=True,
                word_timestamps=high_accuracy,
                condition_on_previous_text=True
            )
            return {"text": " ".join(seg.text for seg in segments).strip()}
        except Exception as e:
            return {"error": str(e)}

    def set_realtime_mode(self, enabled: bool = True):
        """Configure for real-time vs accuracy"""
        if enabled:
            self.min_chunk_duration = 0.4
            self.max_chunk_duration = 1.5
        else:
            self.min_chunk_duration = 1.5
            self.max_chunk_duration = 4.0
        return {"mode": "realtime" if enabled else "accuracy"}

    def set_latency(self, min_ms: int = 400, max_ms: int = 1500):
        """Adjust latency settings"""
        self.min_chunk_duration = min_ms / 1000.0
        self.max_chunk_duration = max_ms / 1000.0
        return {
            "min_chunk_ms": min_ms,
            "max_chunk_ms": max_ms
        }


# =============================================================================
# ADAPTIVE LATENCY MANAGER - Queue drop/catch-up for real-time captions
# =============================================================================

class AdaptiveLatencyManager:
    """
    Manages caption latency by dropping old audio chunks when falling behind.
    Ensures captions stay as close to real-time as possible.
    """

    def __init__(self, max_latency_ms: int = 2000):
        self.max_latency_ms = max_latency_ms  # Maximum acceptable latency
        self.queue = deque()
        self.queue_lock = threading.Lock()
        self.last_process_time = time.time()
        self.dropped_count = 0
        self.total_processed = 0
        self.current_latency_ms = 0

    def add_chunk(self, audio_chunk, timestamp: float = None):
        """Add audio chunk to queue with timestamp, proactively dropping old chunks"""
        if timestamp is None:
            timestamp = time.time()

        with self.queue_lock:
            # Proactively clean old chunks when adding new ones
            current_time = time.time()
            dropped = 0
            while self.queue:
                oldest = self.queue[0]
                age_ms = (current_time - oldest["timestamp"]) * 1000
                # Be aggressive: drop anything older than max_latency
                if age_ms > self.max_latency_ms:
                    self.queue.popleft()
                    dropped += 1
                    self.dropped_count += 1
                else:
                    break

            if dropped > 0:
                print(f"⚡ Proactive catch-up: dropped {dropped} old chunks (queue was behind)")

            self.queue.append({
                "audio": audio_chunk,
                "timestamp": timestamp
            })

    def get_next_chunk(self):
        """Get next chunk to process, dropping old ones if needed"""
        with self.queue_lock:
            if not self.queue:
                return None

            current_time = time.time()

            # Drop old chunks if we're behind - keep only the most recent
            dropped_this_round = 0
            while len(self.queue) > 1:
                oldest = self.queue[0]
                age_ms = (current_time - oldest["timestamp"]) * 1000

                if age_ms > self.max_latency_ms:
                    self.queue.popleft()
                    dropped_this_round += 1
                    self.dropped_count += 1
                else:
                    break

            if dropped_this_round > 0:
                print(f"⚡ Latency catch-up: dropped {dropped_this_round} old chunks")

            if self.queue:
                chunk = self.queue.popleft()
                self.current_latency_ms = (current_time - chunk["timestamp"]) * 1000
                self.total_processed += 1
                return chunk

            return None

    def drop_all_but_latest(self):
        """Aggressively drop everything except the latest chunk"""
        with self.queue_lock:
            if len(self.queue) <= 1:
                return 0
            dropped = len(self.queue) - 1
            latest = self.queue[-1] if self.queue else None
            self.queue.clear()
            if latest:
                self.queue.append(latest)
            self.dropped_count += dropped
            if dropped > 0:
                print(f"⚡ Aggressive catch-up: dropped {dropped} chunks, keeping only latest")
            return dropped

    def get_stats(self) -> Dict:
        """Get latency management statistics"""
        return {
            "queue_size": len(self.queue),
            "current_latency_ms": round(self.current_latency_ms, 1),
            "dropped_count": self.dropped_count,
            "total_processed": self.total_processed,
            "max_latency_ms": self.max_latency_ms,
            "drop_rate": round(self.dropped_count / max(self.total_processed, 1) * 100, 1)
        }

    def set_max_latency(self, max_latency_ms: int):
        """Update maximum acceptable latency"""
        self.max_latency_ms = max_latency_ms

    def clear(self):
        """Clear the queue"""
        with self.queue_lock:
            self.queue.clear()
            self.dropped_count = 0
            self.total_processed = 0
            self.current_latency_ms = 0


# =============================================================================
# SPEECHMATICS ENGINE - Cloud-based real-time ASR
# =============================================================================

class SpeechmaticsEngine:
    """
    Real-time speech-to-text using Speechmatics cloud API.
    Provides low-latency (~300-500ms) high-accuracy transcription.

    Authentication: Uses API key in Authorization header for server-side connections.
    See: https://docs.speechmatics.com/get-started/authentication
    """

    # US region endpoint (us.rt not us1.rt)
    SPEECHMATICS_RT_URL = "wss://us.rt.speechmatics.com/v2"
    SPEECHMATICS_TOKEN_URL = "https://mp.speechmatics.com/v1/api_keys"

    def __init__(self):
        self.api_key = None
        self.is_running = False
        self.text_callback = None
        self.sample_rate = 16000
        self.stream = None
        self.ws_thread = None
        self.audio_thread = None
        self.ws = None
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.latency_manager = AdaptiveLatencyManager(max_latency_ms=2000)
        self.loop = None

    def get_status(self) -> Dict:
        """Get Speechmatics engine status"""
        has_key = bool(ai_config.get("speechmatics_api_key"))
        return {
            "available": WEBSOCKETS_AVAILABLE and AUDIO_AVAILABLE,
            "has_api_key": has_key,
            "is_running": self.is_running,
            "missing_packages": [] if WEBSOCKETS_AVAILABLE else ["websockets"],
            "latency_stats": self.latency_manager.get_stats()
        }

    def set_api_key(self, api_key: str):
        """Set the Speechmatics API key"""
        self.api_key = api_key
        ai_config["speechmatics_api_key"] = api_key
        # Save to config
        try:
            with open(AI_CONFIG_FILE, 'w') as f:
                json.dump(ai_config, f, indent=2)
        except:
            pass
        return {"status": "ok"}

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input - adds to latency-managed queue"""
        self.latency_manager.add_chunk(indata.copy())

    def _get_temp_token(self, api_key: str) -> Optional[str]:
        """
        Generate a temporary JWT token from the long-lived API key.
        This is the recommended approach per Speechmatics docs.
        """
        try:
            url = f"{self.SPEECHMATICS_TOKEN_URL}?type=rt"
            data = json.dumps({"ttl": 3600}).encode('utf-8')  # 1 hour TTL

            req = urllib.request.Request(url, data=data, method='POST')
            req.add_header('Content-Type', 'application/json')
            req.add_header('Authorization', f'Bearer {api_key}')

            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                temp_key = result.get('key_value')
                if temp_key:
                    print("✓ Speechmatics temporary token generated")
                    return temp_key
                else:
                    print(f"⚠️ No key_value in response: {result}")
                    return None
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else str(e)
            print(f"❌ Failed to generate temp token: {e.code} - {error_body}")
            return None
        except Exception as e:
            print(f"❌ Failed to generate temp token: {e}")
            return None

    async def _websocket_handler(self):
        """Handle WebSocket connection to Speechmatics"""
        api_key = self.api_key or ai_config.get("speechmatics_api_key")
        if not api_key:
            print("❌ No Speechmatics API key configured")
            return

        # Try API key directly in Authorization header (works reliably)
        url = self.SPEECHMATICS_RT_URL
        headers = [("Authorization", f"Bearer {api_key}")]

        try:
            connect_kwargs = {'additional_headers': headers}
            await self._try_websocket_connection(url, connect_kwargs)
        except Exception as e:
            print(f"❌ Speechmatics connection failed: {e}")

    async def _try_websocket_connection(self, url: str, connect_kwargs: dict):
        """Attempt WebSocket connection with given URL and kwargs"""
        async with websockets.connect(url, **connect_kwargs) as ws:
            self.ws = ws

            # Send start recognition message
            start_msg = {
                "message": "StartRecognition",
                "audio_format": {
                    "type": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.sample_rate
                },
                "transcription_config": {
                    "language": "en",
                    "enable_partials": True,
                    "max_delay": 2.0
                }
            }
            await ws.send(json.dumps(start_msg))
            print("🎤 Speechmatics connected, starting transcription...")

            # Start audio sending task
            send_task = asyncio.create_task(self._send_audio(ws))

            # Receive transcription results
            async for message in ws:
                if not self.is_running:
                    break

                try:
                    data = json.loads(message)
                    msg_type = data.get("message")

                    # Debug: log all message types
                    if msg_type not in ["AudioAdded"]:  # Ignore audio acks
                        print(f"   [WS] Received: {msg_type}")

                    if msg_type == "RecognitionStarted":
                        print("✓ Speechmatics recognition started")

                    elif msg_type == "AddTranscript":
                        # Final transcript - check both possible locations
                        # Debug: print full data structure
                        print(f"   [WS] AddTranscript data keys: {data.keys()}")
                        text = data.get("metadata", {}).get("transcript", "")
                        if not text:
                            # Try alternative: results array
                            results = data.get("results", [])
                            if results:
                                text = " ".join([r.get("alternatives", [{}])[0].get("content", "") for r in results])
                        print(f"   [WS] AddTranscript text: '{text}'")
                        if text and self.text_callback:
                            self.text_callback(text, is_final=True)
                        else:
                            print(f"   [WS] WARNING: No text or no callback! metadata={data.get('metadata')}, results={data.get('results')}")

                    elif msg_type == "AddPartialTranscript":
                        # Partial/interim transcript - debug first one
                        text = data.get("metadata", {}).get("transcript", "")
                        if not text:
                            results = data.get("results", [])
                            if results:
                                text = " ".join([r.get("alternatives", [{}])[0].get("content", "") for r in results])
                        # Debug: log partial transcripts too
                        if text:
                            print(f"   [WS] Partial: '{text}'")
                        if text and self.text_callback:
                            self.text_callback(text, is_final=False)

                    elif msg_type == "Error":
                        error_info = data.get("reason", data)
                        print(f"❌ Speechmatics error: {error_info}")

                    elif msg_type == "Warning":
                        print(f"⚠️ Speechmatics warning: {data.get('reason', data)}")

                except json.JSONDecodeError:
                    pass

            send_task.cancel()

    async def _send_audio(self, ws):
        """Send audio chunks to Speechmatics"""
        while self.is_running:
            chunk_data = self.latency_manager.get_next_chunk()

            if chunk_data:
                audio = chunk_data["audio"]
                # Convert to 16-bit PCM
                if hasattr(audio, 'tobytes'):
                    audio_bytes = (audio * 32768).astype('int16').tobytes()
                else:
                    audio_bytes = audio

                try:
                    await ws.send(audio_bytes)
                except:
                    break
            else:
                await asyncio.sleep(0.01)

    def _run_async_loop(self):
        """Run the async event loop in a separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._websocket_handler())
        finally:
            self.loop.close()
            self.loop = None

    def start(self, device_id=None, callback=None):
        """Start Speechmatics real-time transcription"""
        if not AUDIO_AVAILABLE:
            return {"error": "sounddevice not installed"}
        if not WEBSOCKETS_AVAILABLE:
            return {"error": "websockets not installed. Run: pip3 install websockets"}

        api_key = self.api_key or ai_config.get("speechmatics_api_key")
        if not api_key:
            return {"error": "No Speechmatics API key configured"}

        # If already running, stop first (handles stale state after page refresh)
        if self.is_running or self.stream or self.ws_thread:
            print("⚠️ Speechmatics was in stale state, cleaning up...")
            self.stop()

        self.text_callback = callback
        self.is_running = True
        self.latency_manager.clear()

        try:
            # Start audio input
            self.stream = sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.1)
            )
            self.stream.start()

            # Start WebSocket handler in separate thread
            self.ws_thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.ws_thread.start()

            print("🎤 Speechmatics listening...")
            return {"status": "started"}

        except Exception as e:
            self.is_running = False
            self.stream = None
            return {"error": str(e)}

    def stop(self):
        """Stop Speechmatics transcription"""
        print("🛑 Stopping Speechmatics...")
        self.is_running = False

        # Stop audio stream first
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"   Error stopping audio stream: {e}")
            self.stream = None

        # Close WebSocket connection
        if self.ws:
            try:
                # Send end of stream message if possible
                pass
            except:
                pass
            self.ws = None

        # Stop the async loop
        if self.loop and self.loop.is_running():
            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception as e:
                print(f"   Error stopping loop: {e}")

        # Wait for thread to finish (with timeout)
        if self.ws_thread and self.ws_thread.is_alive():
            try:
                self.ws_thread.join(timeout=2.0)
            except:
                pass
        self.ws_thread = None
        self.loop = None

        self.latency_manager.clear()
        print("✓ Speechmatics stopped")
        return {"status": "stopped", "stats": self.latency_manager.get_stats()}

    def set_max_latency(self, max_latency_ms: int):
        """Set maximum acceptable latency"""
        self.latency_manager.set_max_latency(max_latency_ms)
        return {"status": "ok", "max_latency_ms": max_latency_ms}


# =============================================================================
# LOCAL WHISPER API ENGINE - OpenAI-compatible local transcription
# =============================================================================

class LocalWhisperAPIEngine:
    """
    Real-time speech-to-text using a local OpenAI-compatible Whisper API.
    Works with: faster-whisper-server, LocalAI, whisper.cpp server, etc.

    Setup options:
    1. faster-whisper-server: pip install faster-whisper-server && faster-whisper-server
    2. LocalAI: Follow https://localai.io docs
    3. whisper.cpp: ./server -m models/ggml-base.bin
    """

    def __init__(self):
        self.api_url = None
        self.api_key = None
        self.is_running = False
        self.text_callback = None
        self.sample_rate = 16000
        self.stream = None
        self.process_thread = None
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.latency_manager = AdaptiveLatencyManager(max_latency_ms=1500)
        self.chunk_duration = 0.8  # seconds of audio per API call (reduced from 2.0 for ultra-fast response)

    def get_status(self) -> Dict:
        """Get Local Whisper API engine status"""
        api_url = ai_config.get("local_whisper_api_url", "")
        configured = bool(api_url)

        # Try to check if server is running
        server_available = False
        if configured:
            try:
                import urllib.request
                # Try a simple GET to check if server is up
                base_url = api_url.rsplit('/v1', 1)[0] if '/v1' in api_url else api_url.rsplit('/audio', 1)[0]
                req = urllib.request.Request(base_url, method='GET')
                req.add_header('User-Agent', 'Community-Captioner/4.1')
                with urllib.request.urlopen(req, timeout=2) as resp:
                    server_available = resp.status == 200
            except:
                # Server might still work, just not respond to GET on base
                server_available = None  # Unknown

        return {
            "available": AUDIO_AVAILABLE,
            "configured": configured,
            "api_url": api_url,
            "server_available": server_available,
            "is_running": self.is_running,
            "latency_stats": self.latency_manager.get_stats()
        }

    def set_config(self, api_url: str, api_key: str = ""):
        """Set the Local Whisper API configuration"""
        self.api_url = api_url
        self.api_key = api_key
        ai_config["local_whisper_api_url"] = api_url
        ai_config["local_whisper_api_key"] = api_key
        # Save to config
        try:
            with open(AI_CONFIG_FILE, 'w') as f:
                json.dump(ai_config, f, indent=2)
        except:
            pass
        return {"status": "ok"}

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input - adds to latency-managed queue"""
        self.latency_manager.add_chunk(indata.copy())

    def _transcription_loop(self):
        """Process audio chunks and send to API"""
        import urllib.request
        import urllib.error
        import io
        import wave

        accumulated_audio = []
        samples_needed = int(self.sample_rate * self.chunk_duration)

        while self.is_running:
            chunk_data = self.latency_manager.get_next_chunk()

            if chunk_data:
                accumulated_audio.append(chunk_data["audio"])
                total_samples = sum(len(a) for a in accumulated_audio)

                # When we have enough audio, send to API
                if total_samples >= samples_needed:
                    # Combine audio chunks
                    audio_data = np.concatenate(accumulated_audio)
                    accumulated_audio = []

                    # Convert to WAV bytes
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(self.sample_rate)
                        wf.writeframes((audio_data * 32768).astype(np.int16).tobytes())
                    wav_bytes = wav_buffer.getvalue()

                    # Send to API
                    try:
                        api_url = self.api_url or ai_config.get("local_whisper_api_url")
                        api_key = self.api_key or ai_config.get("local_whisper_api_key", "")

                        # Build multipart form data
                        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
                        body = (
                            f'--{boundary}\r\n'
                            f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                            f'Content-Type: audio/wav\r\n\r\n'
                        ).encode('utf-8') + wav_bytes + (
                            f'\r\n--{boundary}\r\n'
                            f'Content-Disposition: form-data; name="model"\r\n\r\n'
                            f'whisper-1\r\n'
                            f'--{boundary}--\r\n'
                        ).encode('utf-8')

                        req = urllib.request.Request(api_url, data=body, method='POST')
                        req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
                        if api_key:
                            req.add_header('Authorization', f'Bearer {api_key}')

                        with urllib.request.urlopen(req, timeout=10) as resp:
                            result = json.loads(resp.read().decode('utf-8'))
                            text = result.get('text', '').strip()
                            if text and self.text_callback:
                                self.text_callback(text)

                    except urllib.error.URLError as e:
                        print(f"❌ Local Whisper API error: {e}")
                    except Exception as e:
                        print(f"❌ Local Whisper API error: {e}")

            else:
                time.sleep(0.01)

    def start(self, device_id=None, callback=None):
        """Start Local Whisper API transcription"""
        if not AUDIO_AVAILABLE:
            return {"error": "sounddevice not installed"}

        api_url = self.api_url or ai_config.get("local_whisper_api_url")
        if not api_url:
            return {"error": "No Local Whisper API URL configured"}

        # If already running, stop first
        if self.is_running or self.stream or self.process_thread:
            print("⚠️ Local Whisper API was in stale state, cleaning up...")
            self.stop()

        self.text_callback = callback
        self.is_running = True
        self.latency_manager.clear()

        try:
            # Start audio input with small blocksize for low latency
            self.stream = sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.02)  # 20ms blocks for ultra-low latency
            )
            self.stream.start()

            # Start transcription thread
            self.process_thread = threading.Thread(target=self._transcription_loop, daemon=True)
            self.process_thread.start()

            print("🎤 Local Whisper API listening...")
            return {"status": "started"}

        except Exception as e:
            self.is_running = False
            self.stream = None
            return {"error": str(e)}

    def stop(self):
        """Stop Local Whisper API transcription"""
        print("🛑 Stopping Local Whisper API...")
        self.is_running = False

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"   Error stopping audio stream: {e}")
            self.stream = None

        if self.process_thread and self.process_thread.is_alive():
            try:
                self.process_thread.join(timeout=2.0)
            except:
                pass
        self.process_thread = None

        self.latency_manager.clear()
        print("✓ Local Whisper API stopped")
        return {"status": "stopped", "stats": self.latency_manager.get_stats()}

    def set_max_latency(self, max_latency_ms: int):
        """Set maximum acceptable latency"""
        self.latency_manager.set_max_latency(max_latency_ms)
        return {"status": "ok", "max_latency_ms": max_latency_ms}


# =============================================================================
# GPT POST-PROCESSOR - Real-time caption enhancement
# =============================================================================

class GPTPostProcessor:
    """
    Enhances captions using GPT for:
    - Grammar and punctuation fixes
    - Sentence boundary detection
    - Named entity recognition
    - Real-time quality improvement
    """

    def __init__(self):
        self.enabled = False
        self.buffer = []  # Buffer recent captions for context
        self.buffer_size = 5
        self.processed_count = 0
        self.entities_found = {"people": set(), "places": set(), "organizations": set()}

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def process(self, text: str, context: str = "") -> Dict:
        """Process caption text with GPT for enhancement"""
        if not self.enabled or not openai_client or not text:
            return {"text": text, "enhanced": False}

        # Add to buffer for context
        self.buffer.append(text)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        context_text = " ".join(self.buffer[-3:]) if len(self.buffer) > 1 else text

        try:
            response = openai_client.chat.completions.create(
                model=get_ai_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a real-time caption enhancer. Given raw speech-to-text output, "
                            "fix grammar, add proper punctuation, and correct obvious transcription errors. "
                            "Keep the meaning exactly the same. Be concise. "
                            "Return ONLY the corrected text, nothing else."
                        )
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.1,
                max_tokens=200
            )

            enhanced = response.choices[0].message.content.strip()
            self.processed_count += 1

            return {
                "text": enhanced,
                "original": text,
                "enhanced": True
            }

        except Exception as e:
            print(f"GPT post-processing error: {e}")
            return {"text": text, "enhanced": False, "error": str(e)}

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        if not openai_client or not text:
            return {"people": [], "places": [], "organizations": []}

        try:
            response = openai_client.chat.completions.create(
                model=get_ai_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract named entities from this text. "
                            "Return ONLY a JSON object: {\"people\": [], \"places\": [], \"organizations\": []}"
                        )
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.1,
                max_tokens=300
            )

            try:
                entities = json.loads(response.choices[0].message.content)
                # Track entities found
                for category in self.entities_found:
                    if category in entities:
                        self.entities_found[category].update(entities[category])
                return entities
            except:
                return {"people": [], "places": [], "organizations": []}

        except Exception as e:
            print(f"Entity extraction error: {e}")
            return {"people": [], "places": [], "organizations": []}

    def get_status(self) -> Dict:
        return {
            "enabled": self.enabled,
            "ai_available": openai_client is not None,
            "processed_count": self.processed_count,
            "entities_found": {k: list(v) for k, v in self.entities_found.items()},
            "buffer_size": len(self.buffer)
        }

    def clear(self):
        self.buffer = []
        self.processed_count = 0
        self.entities_found = {"people": set(), "places": set(), "organizations": set()}


# =============================================================================
# SESSION ANALYTICS - Post-session analysis and visualization data
# =============================================================================

class SessionAnalytics:
    """
    Generates analytics data for post-session dashboard:
    - Word frequency / word cloud data
    - Sentiment timeline
    - Topic segmentation
    - Pace analysis
    - Quality metrics
    """

    def __init__(self):
        pass

    def analyze_session(self, captions: List[Dict]) -> Dict:
        """Generate comprehensive analytics for a session"""
        if not captions:
            return {"error": "No captions to analyze"}

        # Basic stats
        total_words = 0
        word_freq = Counter()
        corrections_timeline = []
        pace_data = []

        for i, cap in enumerate(captions):
            text = cap.get("corrected", cap.get("raw", ""))
            words = text.split()
            total_words += len(words)
            word_freq.update(w.lower() for w in words if len(w) > 2)

            # Track corrections over time
            corrections = cap.get("corrections", [])
            if corrections:
                corrections_timeline.append({
                    "index": i,
                    "time": cap.get("time_str", ""),
                    "count": len(corrections)
                })

            # Calculate pace (words per caption as proxy)
            pace_data.append({
                "index": i,
                "words": len(words)
            })

        # Word cloud data (top 50 words)
        word_cloud = [
            {"word": word, "count": count}
            for word, count in word_freq.most_common(50)
        ]

        # Proper nouns (capitalized words)
        proper_nouns = Counter()
        for cap in captions:
            text = cap.get("corrected", cap.get("raw", ""))
            words = text.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 2:
                    proper_nouns[word] += 1

        # Quality metrics
        total_captions = len(captions)
        captions_with_corrections = sum(1 for c in captions if c.get("corrections"))
        total_corrections = sum(len(c.get("corrections", [])) for c in captions)

        analytics = {
            "summary": {
                "total_captions": total_captions,
                "total_words": total_words,
                "avg_words_per_caption": round(total_words / max(total_captions, 1), 1),
                "captions_with_corrections": captions_with_corrections,
                "total_corrections": total_corrections,
                "correction_rate": round(captions_with_corrections / max(total_captions, 1) * 100, 1)
            },
            "word_cloud": word_cloud,
            "proper_nouns": [
                {"word": word, "count": count}
                for word, count in proper_nouns.most_common(30)
            ],
            "corrections_timeline": corrections_timeline,
            "pace_data": pace_data
        }

        # Add AI-powered analysis if available
        if openai_client:
            full_text = " ".join(c.get("corrected", c.get("raw", "")) for c in captions)
            ai_analysis = self._ai_analyze(full_text[:8000])
            analytics["ai_analysis"] = ai_analysis

        return analytics

    def _ai_analyze(self, text: str) -> Dict:
        """Use AI for deeper analysis"""
        try:
            response = openai_client.chat.completions.create(
                model=get_ai_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Analyze this meeting/event transcript. Provide:\n"
                            "1. topics: List of 3-5 main topics discussed\n"
                            "2. sentiment: Overall tone (positive/neutral/negative)\n"
                            "3. key_moments: 2-3 notable quotes or moments\n"
                            "4. speakers_mentioned: List of people who appear to be speaking\n"
                            "Return as JSON: {\"topics\": [], \"sentiment\": \"\", \"key_moments\": [], \"speakers_mentioned\": []}"
                        )
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )

            try:
                return json.loads(response.choices[0].message.content)
            except:
                return {"topics": [], "sentiment": "unknown", "key_moments": [], "speakers_mentioned": []}

        except Exception as e:
            print(f"AI analysis error: {e}")
            return {"error": str(e)}

    def generate_sentiment_timeline(self, captions: List[Dict], window_size: int = 10) -> List[Dict]:
        """Generate sentiment scores over time"""
        if not openai_client or not captions:
            return []

        timeline = []
        for i in range(0, len(captions), window_size):
            window = captions[i:i + window_size]
            text = " ".join(c.get("corrected", c.get("raw", "")) for c in window)

            try:
                response = openai_client.chat.completions.create(
                    model=get_ai_model(),
                    messages=[
                        {
                            "role": "system",
                            "content": "Rate the sentiment of this text on a scale of -1 (very negative) to 1 (very positive). Return ONLY a number."
                        },
                        {
                            "role": "user",
                            "content": text[:1000]
                        }
                    ],
                    temperature=0.1,
                    max_tokens=10
                )

                try:
                    score = float(response.choices[0].message.content.strip())
                    score = max(-1, min(1, score))
                except:
                    score = 0

                timeline.append({
                    "index": i,
                    "time": window[0].get("time_str", "") if window else "",
                    "sentiment": score
                })

            except:
                timeline.append({"index": i, "sentiment": 0})

        return timeline


# =============================================================================
# VIDEO INTELLIGENCE - Video upload, sync, and highlight generation
# =============================================================================

# Check for ffmpeg availability
FFMPEG_AVAILABLE = False
try:
    import subprocess
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    FFMPEG_AVAILABLE = result.returncode == 0
except:
    pass


class VideoIntelligence:
    """
    Handles video processing:
    - Video upload and storage
    - YouTube video download with progress
    - Transcript-to-video sync
    - AI-powered highlight detection with quote matching
    - Clip generation with ffmpeg (with padding)
    - Highlight reel compilation
    - Caption file generation
    """

    def __init__(self, video_dir: Path):
        self.video_dir = video_dir
        self.video_dir.mkdir(exist_ok=True)
        self.clips_dir = video_dir / "clips"
        self.clips_dir.mkdir(exist_ok=True)
        self.current_video = None
        self.transcript = ""  # Store transcript for caption generation
        self.transcript_segments = []  # Store timestamped segments
        self.download_progress = {"status": "idle", "percent": 0, "message": ""}

    def upload_video(self, filename: str, content: bytes) -> Dict:
        """Save uploaded video file"""
        video_id = str(uuid.uuid4())[:8]
        ext = filename.split('.')[-1] if '.' in filename else 'mp4'
        video_path = self.video_dir / f"{video_id}.{ext}"

        try:
            with open(video_path, 'wb') as f:
                f.write(content)

            # Get video duration using ffmpeg
            duration = self._get_video_duration(video_path)

            self.current_video = {
                "id": video_id,
                "filename": filename,
                "path": str(video_path),
                "duration": duration,
                "uploaded": datetime.now().isoformat()
            }

            return {
                "status": "ok",
                "video_id": video_id,
                "filename": filename,
                "duration": duration,
                "path": str(video_path)
            }

        except Exception as e:
            return {"error": str(e)}

    def download_youtube_video(self, url: str) -> Dict:
        """Download YouTube video with detailed progress tracking using yt-dlp"""
        import re
        import time

        # Extract video ID
        video_id_match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
        if not video_id_match:
            return {"error": "Invalid YouTube URL"}

        yt_video_id = video_id_match.group(1)
        video_id = str(uuid.uuid4())[:8]
        output_path = self.video_dir / f"{video_id}.mp4"

        self.download_progress = {
            "status": "starting",
            "percent": 0,
            "message": "Initializing download...",
            "downloaded_size": "",
            "total_size": "",
            "speed": "",
            "eta": ""
        }

        start_time = time.time()

        try:
            # Use yt-dlp to download - prefer 1080p, fallback to lower
            cmd = [
                'yt-dlp',
                '-f', 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best',
                '--merge-output-format', 'mp4',
                '-o', str(output_path),
                '--progress',
                '--newline',
                url
            ]

            self.download_progress = {
                "status": "downloading",
                "percent": 2,
                "message": "Connecting to YouTube...",
                "downloaded_size": "",
                "total_size": "",
                "speed": "",
                "eta": ""
            }

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Parse progress from yt-dlp output with detailed info
            for line in process.stdout:
                line = line.strip()
                if '[download]' in line:
                    # Parse: [download]  45.2% of  150.00MiB at   5.50MiB/s ETA 00:15
                    percent_match = re.search(r'(\d+\.?\d*)%', line)
                    size_match = re.search(r'of\s+([\d.]+\s*\w+)', line)
                    speed_match = re.search(r'at\s+([\d.]+\s*\w+/s)', line)
                    eta_match = re.search(r'ETA\s+(\d+:\d+(?::\d+)?)', line)
                    downloaded_match = re.search(r'(\d+\.?\d*)\s*%\s+of\s+[\d.]+', line)

                    if percent_match:
                        percent = float(percent_match.group(1))
                        total_size = size_match.group(1) if size_match else ""
                        speed = speed_match.group(1) if speed_match else ""
                        eta = eta_match.group(1) if eta_match else ""

                        # Calculate downloaded size
                        if total_size and percent > 0:
                            try:
                                size_val = float(re.search(r'[\d.]+', total_size).group())
                                size_unit = re.search(r'[A-Za-z]+', total_size).group()
                                downloaded_val = size_val * (percent / 100)
                                downloaded_size = f"{downloaded_val:.1f} {size_unit}"
                            except:
                                downloaded_size = ""
                        else:
                            downloaded_size = ""

                        self.download_progress = {
                            "status": "downloading",
                            "percent": min(90, percent),
                            "message": f"Downloading video: {percent:.1f}%",
                            "downloaded_size": downloaded_size,
                            "total_size": total_size,
                            "speed": speed,
                            "eta": eta
                        }
                elif 'Merging' in line:
                    self.download_progress = {
                        "status": "processing",
                        "percent": 92,
                        "message": "Merging video and audio streams...",
                        "downloaded_size": self.download_progress.get("total_size", ""),
                        "total_size": self.download_progress.get("total_size", ""),
                        "speed": "",
                        "eta": "~30s"
                    }
                elif 'Deleting' in line or 'has already been downloaded' in line:
                    self.download_progress = {
                        "status": "processing",
                        "percent": 98,
                        "message": "Finalizing...",
                        "downloaded_size": self.download_progress.get("total_size", ""),
                        "total_size": self.download_progress.get("total_size", ""),
                        "speed": "",
                        "eta": "~5s"
                    }

            process.wait()

            if process.returncode != 0 or not output_path.exists():
                self.download_progress = {"status": "error", "percent": 0, "message": "Download failed"}
                return {"error": "Failed to download video. Try uploading directly."}

            # Get video duration and file size
            duration = self._get_video_duration(output_path)
            file_size = output_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            elapsed = time.time() - start_time

            self.current_video = {
                "id": video_id,
                "youtube_id": yt_video_id,
                "filename": f"youtube_{yt_video_id}.mp4",
                "path": str(output_path),
                "duration": duration,
                "file_size": file_size,
                "file_size_mb": file_size_mb,
                "is_youtube": True,
                "source_url": url,
                "uploaded": datetime.now().isoformat()
            }

            self.download_progress = {
                "status": "complete",
                "percent": 100,
                "message": f"Download complete! ({file_size_mb:.1f} MB in {elapsed:.0f}s)",
                "downloaded_size": f"{file_size_mb:.1f} MB",
                "total_size": f"{file_size_mb:.1f} MB",
                "speed": "",
                "eta": ""
            }

            return {
                "status": "ok",
                "video_id": video_id,
                "youtube_id": yt_video_id,
                "filename": f"youtube_{yt_video_id}.mp4",
                "duration": duration,
                "file_size_mb": file_size_mb,
                "path": str(output_path)
            }

        except FileNotFoundError:
            self.download_progress = {"status": "error", "percent": 0, "message": "yt-dlp not installed"}
            return {"error": "yt-dlp not installed. Run: pip install yt-dlp"}
        except Exception as e:
            self.download_progress = {"status": "error", "percent": 0, "message": str(e)}
            return {"error": str(e)}

    def get_download_progress(self) -> Dict:
        """Get current download progress"""
        return self.download_progress

    def set_transcript(self, transcript: str, segments: List[Dict] = None):
        """Store transcript for caption generation and highlight matching"""
        self.transcript = transcript
        self.transcript_segments = segments or []
        # Parse segments from transcript if not provided
        if not segments and transcript:
            self._parse_transcript_segments(transcript)

    def _parse_transcript_segments(self, transcript: str):
        """Parse transcript text into timestamped segments for quote matching"""
        import re
        self.transcript_segments = []

        # Try to find timestamped lines like [0:00] or [00:00:00] or (0:00)
        lines = transcript.split('\n')
        current_time = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for timestamps at start of line
            time_match = re.match(r'[\[\(]?(\d{1,2}):(\d{2})(?::(\d{2}))?[\]\)]?\s*(.+)', line)
            if time_match:
                hours = int(time_match.group(3)) if time_match.group(3) else 0
                mins = int(time_match.group(1)) if not time_match.group(3) else int(time_match.group(1)) * 60 + int(time_match.group(2))
                secs = int(time_match.group(2)) if not time_match.group(3) else int(time_match.group(3))
                current_time = hours * 3600 + mins * 60 + secs
                text = time_match.group(4)
            else:
                text = line

            if text:
                self.transcript_segments.append({
                    "start": current_time,
                    "text": text,
                    "duration": 5  # Default duration estimate
                })
                current_time += 5  # Increment for non-timestamped lines

    def find_quote_timestamp(self, quote: str, padding: float = 2.0) -> Dict:
        """Find the timestamp of a quote in the transcript with padding"""
        if not self.transcript_segments:
            return None

        quote_lower = quote.lower().strip()
        best_match = None
        best_score = 0

        for seg in self.transcript_segments:
            seg_text = seg.get('text', '').lower()

            # Exact substring match
            if quote_lower in seg_text:
                return {
                    "start": max(0, seg['start'] - padding),
                    "end": seg['start'] + seg.get('duration', 10) + padding,
                    "matched_text": seg['text']
                }

            # Fuzzy matching - check word overlap
            quote_words = set(quote_lower.split())
            seg_words = set(seg_text.split())
            overlap = len(quote_words & seg_words) / max(len(quote_words), 1)

            if overlap > best_score and overlap > 0.5:
                best_score = overlap
                best_match = {
                    "start": max(0, seg['start'] - padding),
                    "end": seg['start'] + seg.get('duration', 10) + padding,
                    "matched_text": seg['text'],
                    "confidence": overlap
                }

        return best_match

    def generate_caption_file(self, format: str = 'srt') -> Dict:
        """Generate caption file (SRT, VTT, or TXT) from stored transcript"""
        if not self.transcript_segments:
            if self.transcript:
                self._parse_transcript_segments(self.transcript)
            else:
                return {"error": "No transcript available"}

        video_id = self.current_video.get('id', 'captions') if self.current_video else 'captions'

        if format == 'srt':
            return self._generate_srt(video_id)
        elif format == 'vtt':
            return self._generate_vtt(video_id)
        elif format == 'txt':
            return self._generate_txt(video_id)
        else:
            return {"error": f"Unknown format: {format}"}

    def _generate_srt(self, video_id: str) -> Dict:
        """Generate SRT caption file"""
        output_path = self.clips_dir / f"{video_id}_captions.srt"

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(self.transcript_segments, 1):
                    start = seg.get('start', 0)
                    duration = seg.get('duration', 5)
                    end = start + duration

                    start_str = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
                    end_str = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"

                    f.write(f"{i}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{seg.get('text', '')}\n\n")

            return {
                "status": "ok",
                "path": str(output_path),
                "filename": f"{video_id}_captions.srt",
                "format": "srt"
            }
        except Exception as e:
            return {"error": str(e)}

    def _generate_vtt(self, video_id: str) -> Dict:
        """Generate WebVTT caption file"""
        output_path = self.clips_dir / f"{video_id}_captions.vtt"

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                for i, seg in enumerate(self.transcript_segments, 1):
                    start = seg.get('start', 0)
                    duration = seg.get('duration', 5)
                    end = start + duration

                    start_str = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d}.{int((start%1)*1000):03d}"
                    end_str = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d}.{int((end%1)*1000):03d}"

                    f.write(f"{i}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{seg.get('text', '')}\n\n")

            return {
                "status": "ok",
                "path": str(output_path),
                "filename": f"{video_id}_captions.vtt",
                "format": "vtt"
            }
        except Exception as e:
            return {"error": str(e)}

    def _generate_txt(self, video_id: str) -> Dict:
        """Generate plain text transcript file"""
        output_path = self.clips_dir / f"{video_id}_transcript.txt"

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for seg in self.transcript_segments:
                    start = seg.get('start', 0)
                    time_str = f"[{int(start//60):02d}:{int(start%60):02d}]"
                    f.write(f"{time_str} {seg.get('text', '')}\n")

            return {
                "status": "ok",
                "path": str(output_path),
                "filename": f"{video_id}_transcript.txt",
                "format": "txt"
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds using ffprobe"""
        if not FFMPEG_AVAILABLE:
            return 0

        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0

    def generate_highlights_from_transcript(self, transcript: str, clip_padding: float = 2.0) -> Dict:
        """
        Use AI to identify highlight moments from transcript text.
        Uses DIRECT QUOTES to find exact timestamps in the transcript.
        This allows highlight detection even without a captioning session.
        """
        if not transcript or not transcript.strip():
            return {"error": "No transcript provided"}

        if not openai_client:
            return {"error": "AI not available for highlight detection. Configure OpenAI API key."}

        # Store transcript for quote matching
        self.set_transcript(transcript)

        try:
            response = openai_client.chat.completions.create(
                model=get_ai_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Analyze this transcript and identify 5-8 highlight moments.\n"
                            "Look for: key points, interesting statements, memorable quotes, important announcements, "
                            "conclusions, or engaging content good for a highlight reel.\n\n"
                            "CRITICAL: For each highlight, you MUST include a DIRECT QUOTE from the transcript. "
                            "The quote should be 5-15 words that appear EXACTLY in the transcript text. "
                            "This quote will be used to find the exact timestamp.\n\n"
                            "Return ONLY a valid JSON array:\n"
                            '[\n'
                            '  {\n'
                            '    "quote": "exact words from the transcript",\n'
                            '    "title": "short 3-5 word title for this moment",\n'
                            '    "reason": "why this is a good highlight (10-20 words)",\n'
                            '    "importance": 8\n'
                            '  }\n'
                            ']\n\n'
                            "Order by importance (10=most important, 1=least). Include 5-8 highlights."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Transcript:\n\n{transcript[:12000]}"
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )

            response_text = response.choices[0].message.content.strip()

            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                response_text = json_match.group(0)

            try:
                highlights = json.loads(response_text)
            except json.JSONDecodeError:
                return {"error": "Failed to parse AI response", "raw": response_text}

            # Now match quotes to timestamps
            matched_highlights = []
            for h in highlights:
                quote = h.get('quote', '')
                if not quote:
                    continue

                # Find timestamp for this quote
                match = self.find_quote_timestamp(quote, padding=clip_padding)

                if match:
                    matched_highlights.append({
                        "title": h.get('title', 'Highlight'),
                        "quote": quote,
                        "reason": h.get('reason', ''),
                        "importance": h.get('importance', 5),
                        "start_time": match['start'],
                        "end_time": match['end'],
                        "matched_text": match.get('matched_text', ''),
                        "timestamp": self._seconds_to_timestamp(match['start']),
                        "end_timestamp": self._seconds_to_timestamp(match['end'])
                    })
                else:
                    # Couldn't find quote - estimate from position in transcript
                    position_ratio = transcript.lower().find(quote.lower()[:20]) / max(len(transcript), 1)
                    estimated_time = position_ratio * (self.current_video.get('duration', 300) if self.current_video else 300)
                    matched_highlights.append({
                        "title": h.get('title', 'Highlight'),
                        "quote": quote,
                        "reason": h.get('reason', ''),
                        "importance": h.get('importance', 5),
                        "start_time": max(0, estimated_time - clip_padding),
                        "end_time": estimated_time + 15 + clip_padding,  # 15 second default clip
                        "timestamp": self._seconds_to_timestamp(max(0, estimated_time - clip_padding)),
                        "end_timestamp": self._seconds_to_timestamp(estimated_time + 15 + clip_padding),
                        "estimated": True
                    })

            # Generate a brief summary of the content
            summary = None
            if matched_highlights:
                try:
                    summary_response = openai_client.chat.completions.create(
                        model=get_ai_model(),
                        messages=[
                            {
                                "role": "system",
                                "content": "Generate a 2-3 sentence summary of this content, focusing on the main themes and key takeaways. Be concise."
                            },
                            {
                                "role": "user",
                                "content": f"Transcript:\n\n{transcript[:4000]}"
                            }
                        ],
                        temperature=0.3,
                        max_tokens=150
                    )
                    summary = summary_response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"Error generating summary: {e}")
                    summary = None

            return {
                "status": "ok",
                "highlights": matched_highlights,
                "count": len(matched_highlights),
                "matched": len([h for h in matched_highlights if not h.get('estimated')]),
                "summary": summary
            }

        except Exception as e:
            return {"error": str(e)}

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    def generate_highlights(self, captions: List[Dict], video_path: str = None) -> Dict:
        """Use AI to identify highlight moments from session captions"""
        if not captions:
            return {"error": "No captions to analyze"}

        # Build transcript from captions and use the new method
        transcript = "\n".join([
            f"[{c.get('time_str', '')}] {c.get('corrected', c.get('raw', ''))}"
            for c in captions
        ])

        return self.generate_highlights_from_transcript(transcript)

    def extract_clip(self, video_path: str, start_time: str, end_time: str, output_name: str = None) -> Dict:
        """Extract a clip from video using ffmpeg"""
        if not FFMPEG_AVAILABLE:
            return {"error": "ffmpeg not available"}

        if not Path(video_path).exists():
            return {"error": "Video file not found"}

        clip_id = str(uuid.uuid4())[:8]
        output_name = output_name or f"clip_{clip_id}.mp4"
        output_path = self.clips_dir / output_name

        try:
            result = subprocess.run([
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', start_time,
                '-to', end_time,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'fast',
                str(output_path)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                return {
                    "status": "ok",
                    "clip_id": clip_id,
                    "path": str(output_path),
                    "filename": output_name
                }
            else:
                return {"error": result.stderr}

        except Exception as e:
            return {"error": str(e)}

    def generate_highlight_reel(self, video_path: str, highlights: List[Dict], output_name: str = None,
                                  aspect_ratio: str = '16:9', burn_captions: bool = False,
                                  target_duration: int = 60) -> Dict:
        """Create a highlight reel by concatenating clips with optional 9:16 crop and caption burning"""
        if not FFMPEG_AVAILABLE:
            return {"error": "ffmpeg not available"}

        if not highlights:
            return {"error": "No highlights provided"}

        # First extract all clips
        clips = []
        total_duration = 0
        for i, h in enumerate(highlights):
            # Stop if we've hit target duration
            if total_duration >= target_duration:
                break

            start = h.get("timestamp") or h.get("start_time", 0)
            if isinstance(start, (int, float)):
                start = f"{int(start // 3600):02d}:{int((start % 3600) // 60):02d}:{int(start % 60):02d}"

            end = h.get("end_timestamp") or h.get("end_time")
            if end is None:
                end = self._add_seconds(start, min(30, target_duration - total_duration))
            elif isinstance(end, (int, float)):
                end = f"{int(end // 3600):02d}:{int((end % 3600) // 60):02d}:{int(end % 60):02d}"

            clip_result = self.extract_clip(video_path, start, end, f"highlight_{i}.mp4")
            if clip_result.get("status") == "ok":
                clips.append(clip_result["path"])
                # Estimate clip duration
                clip_dur = h.get("duration", 30)
                total_duration += clip_dur if isinstance(clip_dur, (int, float)) else 30

        if not clips:
            return {"error": "No clips could be extracted"}

        # Create concat file
        concat_file = self.clips_dir / "concat_list.txt"
        with open(concat_file, 'w') as f:
            for clip in clips:
                f.write(f"file '{clip}'\n")

        # Generate highlight reel
        reel_id = str(uuid.uuid4())[:8]

        # Build ffmpeg command with optional filters
        vf_filters = []

        # Handle aspect ratio (9:16 for social media)
        if aspect_ratio == '9:16':
            # Crop center for 9:16 portrait mode
            vf_filters.append("crop=ih*9/16:ih")
            output_name = output_name or f"highlight_reel_{reel_id}_9x16.mp4"
        else:
            output_name = output_name or f"highlight_reel_{reel_id}.mp4"

        # Handle caption burning
        if burn_captions and self.transcript:
            # Create SRT file from transcript
            srt_file = self.clips_dir / f"captions_{reel_id}.srt"
            try:
                self._create_srt_for_reel(highlights, srt_file)
                # Add subtitles filter with styling for social media
                if aspect_ratio == '9:16':
                    vf_filters.append(f"subtitles='{srt_file}':force_style='FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Alignment=2'")
                else:
                    vf_filters.append(f"subtitles='{srt_file}':force_style='FontSize=20,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2'")
            except Exception as e:
                print(f"Warning: Could not create SRT for burning: {e}")

        output_path = self.clips_dir / output_name

        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0',
                '-i', str(concat_file)
            ]

            # Add video filter if needed
            if vf_filters:
                cmd.extend(['-vf', ','.join(vf_filters)])

            cmd.extend([
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'fast',
                str(output_path)
            ])

            result = subprocess.run(cmd, capture_output=True, text=True)

            # Clean up individual clips
            for clip in clips:
                try:
                    Path(clip).unlink()
                except:
                    pass
            concat_file.unlink()

            if result.returncode == 0:
                return {
                    "status": "ok",
                    "reel_id": reel_id,
                    "reel_path": str(output_path),
                    "path": str(output_path),
                    "filename": output_name,
                    "clips_count": len(clips),
                    "aspect_ratio": aspect_ratio,
                    "captions_burned": burn_captions,
                    "duration": total_duration
                }
            else:
                return {"error": result.stderr}

        except Exception as e:
            return {"error": str(e)}

    def _create_srt_for_reel(self, highlights: List[Dict], srt_path: Path):
        """Create SRT file from highlights for caption burning"""
        with open(srt_path, 'w') as f:
            for i, h in enumerate(highlights):
                start_sec = h.get("start_time", i * 10)
                end_sec = h.get("end_time", start_sec + 10)
                text = h.get("text", h.get("title", f"Highlight {i + 1}"))

                # Format times as HH:MM:SS,mmm
                start_str = f"{int(start_sec // 3600):02d}:{int((start_sec % 3600) // 60):02d}:{int(start_sec % 60):02d},000"
                end_str = f"{int(end_sec // 3600):02d}:{int((end_sec % 3600) // 60):02d}:{int(end_sec % 60):02d},000"

                f.write(f"{i + 1}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{text}\n\n")

    def _add_seconds(self, timestamp: str, seconds: int) -> str:
        """Add seconds to a timestamp string HH:MM:SS"""
        parts = timestamp.split(':')
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2].split('.')[0])
            total = h * 3600 + m * 60 + s + seconds
            new_h = total // 3600
            new_m = (total % 3600) // 60
            new_s = total % 60
            return f"{new_h:02d}:{new_m:02d}:{new_s:02d}"
        return timestamp

    def get_status(self) -> Dict:
        return {
            "ffmpeg_available": FFMPEG_AVAILABLE,
            "current_video": self.current_video,
            "video_dir": str(self.video_dir),
            "clips_dir": str(self.clips_dir)
        }


# Create video directory
VIDEO_DIR = SCRIPT_DIR / "videos"
VIDEO_DIR.mkdir(exist_ok=True)


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

caption_engine = CaptionEngine(DATA_DIR)
knowledge_base = KnowledgeBase(KNOWLEDGE_DIR, caption_engine.embeddings)
session_manager = SessionManager(SESSIONS_DIR)
whisper_engine = WhisperEngine()
speechmatics_engine = SpeechmaticsEngine()
local_whisper_api_engine = LocalWhisperAPIEngine()
gpt_processor = GPTPostProcessor()
session_analytics = SessionAnalytics()
video_intelligence = VideoIntelligence(VIDEO_DIR)
latency_manager = AdaptiveLatencyManager(max_latency_ms=2000)  # Global for browser captions

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
speechmatics_buffer = ""  # Rolling buffer for Speechmatics
local_whisper_api_buffer = ""
browser_caption_buffer = ""  # Rolling buffer for Browser Speech API

# =============================================================================
# ROLLING CAPTION BUFFER - Keeps text visible longer (up to 2 lines)
# =============================================================================

def update_rolling_caption(new_text, is_final=False):
    """
    Update caption with rolling buffer behavior.
    Text accumulates up to 2 lines worth, then older text scrolls off.
    """
    global speechmatics_buffer

    new_text = new_text.strip()
    if not new_text:
        return

    if is_final:
        # For final transcripts, append to buffer
        if speechmatics_buffer:
            speechmatics_buffer = speechmatics_buffer + " " + new_text
        else:
            speechmatics_buffer = new_text
    else:
        # For partials, replace the end of buffer after last final
        # This prevents partials from accumulating incorrectly
        speechmatics_buffer = new_text

    # Apply corrections and update caption state
    result = process_caption(speechmatics_buffer)
    caption_state["raw_caption"] = result["raw"]
    caption_state["caption"] = result["corrected"]
    caption_state["corrections"] = result["corrections"]

# =============================================================================
# CAPTION CALLBACKS - All use rolling buffer for 2-line display
# =============================================================================

def on_whisper_text(text):
    """Callback when Whisper produces text"""
    global whisper_buffer

    text = text.strip()
    if not text:
        return

    # Accumulate for display
    whisper_buffer = (whisper_buffer + " " + text).strip()

    # Apply corrections and limit to 2 lines
    result = process_caption(whisper_buffer)
    caption_state["raw_caption"] = result["raw"]
    caption_state["caption"] = result["corrected"]
    caption_state["corrections"] = result["corrections"]

    # Include latency stats for frontend display
    caption_state["latency_stats"] = whisper_engine.get_status().get("latency_stats", {})
    caption_state["is_behind"] = whisper_engine.current_lag_ms > 1000

    # Save to session
    if session_manager.is_recording:
        session_manager.add_caption(result["raw"], result["corrected"], result["corrections"])

    # Log with latency info
    lag_info = f" [lag: {whisper_engine.current_lag_ms:.0f}ms]" if whisper_engine.current_lag_ms > 500 else ""
    print(f"   📝 Whisper: {text}{lag_info}")

def on_speechmatics_text(text, is_final=False):
    """
    Callback when Speechmatics produces text.
    Only shows FINAL transcripts - partials are ignored to prevent repetition.
    """
    global speechmatics_buffer

    text = text.strip()
    if not text:
        return

    # IGNORE partials entirely - they just repeat what will become final
    # This prevents the "falling behind" issue from repeating text
    if not is_final:
        return

    print(f"   ✅ FINAL: {text}")

    # Save to session (just this segment)
    result = process_caption(text)
    if session_manager.is_recording:
        session_manager.add_caption(result["raw"], result["corrected"], result["corrections"])

    # Append to buffer and trim to ~100 chars (roughly 2 lines)
    if speechmatics_buffer:
        speechmatics_buffer = speechmatics_buffer + " " + text
    else:
        speechmatics_buffer = text

    # Trim buffer to last ~100 chars, breaking at word boundary
    if len(speechmatics_buffer) > 100:
        trimmed = speechmatics_buffer[-100:]
        space_idx = trimmed.find(' ')
        if space_idx > 0 and space_idx < 30:
            speechmatics_buffer = trimmed[space_idx + 1:]
        else:
            speechmatics_buffer = trimmed

    # Update display with trimmed buffer
    result = process_caption(speechmatics_buffer)
    caption_state["raw_caption"] = result["raw"]
    caption_state["caption"] = result["corrected"]
    caption_state["corrections"] = result["corrections"]

    # Update latency stats
    try:
        caption_state["latency_stats"] = speechmatics_engine.latency_manager.get_stats()
    except:
        pass

def on_local_whisper_api_text(text):
    """Callback when Local Whisper API produces text"""
    global local_whisper_api_buffer

    text = text.strip()
    if not text:
        return

    # Accumulate for display
    local_whisper_api_buffer = (local_whisper_api_buffer + " " + text).strip()

    # Apply corrections
    result = process_caption(local_whisper_api_buffer)
    caption_state["raw_caption"] = result["raw"]
    caption_state["caption"] = result["corrected"]
    caption_state["corrections"] = result["corrections"]

    # Save to session
    if session_manager.is_recording:
        session_manager.add_caption(result["raw"], result["corrected"], result["corrections"])

    # Update latency stats
    try:
        caption_state["latency_stats"] = local_whisper_api_engine.latency_manager.get_stats()
    except:
        pass

    print(f"   📝 Local Whisper API: {text}")

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
        try:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Client disconnected, silently ignore
            pass
        except Exception as e:
            print(f"Error sending JSON response: {e}")
    
    def send_file_download(self, content, filename, content_type='text/plain'):
        try:
            body = content.encode() if isinstance(content, str) else content
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.send_header('Content-Length', len(body))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            print(f"Error sending file download: {e}")
    
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
        try:
            self._handle_get()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # Client disconnected
        except Exception as e:
            print(f"Error in GET {self.path}: {e}")
            try:
                self.send_json({"error": str(e)}, 500)
            except:
                pass

    def _handle_get(self):
        path = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)

        # Caption endpoints
        if path == '/api/caption':
            # Debug: log every caption request (uncomment to debug)
            # print(f"   [API] Caption requested. Current: '{caption_state.get('caption', '')[:40]}...'")
            self.send_json({**caption_state, "session": session_manager.get_status()})
        
        elif path == '/api/info':
            # Return system capabilities using global constants
            self.send_json({
                "local_ip": LOCAL_IP, "port": PORT,
                "whisper_available": WHISPER_AVAILABLE and AUDIO_AVAILABLE and NUMPY_AVAILABLE,
                "embeddings_available": EMBEDDINGS_AVAILABLE,
                "rag_enabled": caption_engine.rag_enabled,
                "ffmpeg_available": FFMPEG_AVAILABLE,
                "pdf_available": PDF_AVAILABLE,
                "docx_available": DOCX_AVAILABLE,
                "youtube_transcript_available": YOUTUBE_TRANSCRIPT_AVAILABLE,
                "ai_available": openai_client is not None,
                "version": "4.0"
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

        # RAG Engine endpoints (v4.0)
        elif path == '/api/engine/suggestions':
            self.send_json({"suggestions": caption_engine.pending_suggestions[-50:]})

        elif path == '/api/engine/learned':
            self.send_json({"patterns": caption_engine.learner.get_all_patterns()})

        elif path == '/api/engine/embeddings/status':
            self.send_json({
                "available": caption_engine.embeddings.is_available(),
                "type": caption_engine.embeddings.model_type,
                "cached_count": len(caption_engine.embeddings.embeddings_cache)
            })

        # Whisper endpoints
        elif path == '/api/whisper/status':
            self.send_json(whisper_engine.get_status())

        elif path == '/api/whisper/devices':
            self.send_json({"devices": whisper_engine.get_audio_devices()})

        # Speechmatics endpoints
        elif path == '/api/speechmatics/status':
            self.send_json(speechmatics_engine.get_status())

        # Local Whisper API status endpoint
        elif path == '/api/localwhisper/status':
            self.send_json(local_whisper_api_engine.get_status())

        # Latency stats endpoint
        elif path == '/api/latency/stats':
            self.send_json({
                "global": latency_manager.get_stats(),
                "speechmatics": speechmatics_engine.latency_manager.get_stats(),
                "localwhisper": local_whisper_api_engine.latency_manager.get_stats()
            })

        # Server-Sent Events for real-time caption streaming (hardware encoder integration)
        elif path == '/api/caption/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            last_caption = ""
            try:
                while True:
                    current = caption_state.get('caption', '')
                    if current != last_caption:
                        last_caption = current
                        data = json.dumps({
                            'text': current,
                            'raw': caption_state.get('raw_caption', ''),
                            'corrections': caption_state.get('corrections', []),
                            'timestamp': datetime.now().isoformat()
                        })
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                    import time
                    time.sleep(0.1)  # 100ms polling
            except (BrokenPipeError, ConnectionResetError):
                pass  # Client disconnected

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

        # Session history endpoints
        elif path == '/api/sessions/list':
            self.send_json({"sessions": session_manager.list_sessions()})

        elif path.startswith('/api/sessions/'):
            # /api/sessions/{session_id}
            session_id = path.split('/')[-1]
            if session_id:
                data = session_manager.load_session(session_id)
                if data:
                    self.send_json(data)
                else:
                    self.send_json({"error": "Session not found"}, 404)
            else:
                self.send_json({"error": "Session ID required"}, 400)

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

        # Knowledge Base endpoints (GET)
        elif path == '/api/knowledge/status':
            self.send_json(knowledge_base.get_status())

        elif path == '/api/knowledge/entities':
            self.send_json(knowledge_base.get_all_entities())

        elif path == '/api/knowledge/search':
            query = query.get('q', [''])[0]
            top_k = int(query.get('k', ['5'])[0]) if isinstance(query, dict) else 5
            results = knowledge_base.search(query, top_k)
            self.send_json({"results": results})

        # GPT Post-Processor endpoints (GET)
        elif path == '/api/gpt/status':
            self.send_json(gpt_processor.get_status())

        # Session Analytics endpoints (GET)
        elif path == '/api/session/analytics':
            captions = session_manager.get_captions()
            analytics = session_analytics.analyze_session(captions)
            self.send_json(analytics)

        elif path == '/api/session/analytics/sentiment':
            captions = session_manager.get_captions()
            timeline = session_analytics.generate_sentiment_timeline(captions)
            self.send_json({"timeline": timeline})

        # Video Intelligence endpoints (GET)
        elif path == '/api/video/status':
            self.send_json(video_intelligence.get_status())

        else:
            super().do_GET()
    
    def do_POST(self):
        try:
            self._handle_post()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # Client disconnected
        except Exception as e:
            print(f"Error in POST {self.path}: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.send_json({"error": str(e)}, 500)
            except:
                pass

    def _handle_post(self):
        global whisper_buffer, speechmatics_buffer, local_whisper_api_buffer, browser_caption_buffer, caption_state
        path = urlparse(self.path).path
        data = self.read_json()

        # Caption from browser - uses rolling buffer for 2-line display
        if path == '/api/caption':
            if 'caption' in data:
                new_text = data['caption'].strip()
                is_final = data.get('is_final', True)

                if is_final and new_text:
                    # Save to session FIRST (just this segment)
                    result = process_caption(new_text)
                    if session_manager.is_recording and result["corrected"]:
                        session_manager.add_caption(result["raw"], result["corrected"], result["corrections"])

                    # Append to buffer
                    if browser_caption_buffer:
                        browser_caption_buffer = browser_caption_buffer + " " + new_text
                    else:
                        browser_caption_buffer = new_text

                    # Trim buffer to last ~100 chars, breaking at word boundary
                    if len(browser_caption_buffer) > 100:
                        trimmed = browser_caption_buffer[-100:]
                        space_idx = trimmed.find(' ')
                        if space_idx > 0 and space_idx < 30:
                            browser_caption_buffer = trimmed[space_idx + 1:]
                        else:
                            browser_caption_buffer = trimmed

                    # Update display with trimmed buffer
                    result = process_caption(browser_caption_buffer)
                    caption_state['raw_caption'] = result["raw"]
                    caption_state['caption'] = result["corrected"]
                    caption_state['corrections'] = result["corrections"]
                else:
                    # For partials, show buffer + current partial
                    if browser_caption_buffer:
                        display_text = browser_caption_buffer + " " + new_text
                    else:
                        display_text = new_text

                    result = process_caption(display_text)
                    caption_state['raw_caption'] = result["raw"]
                    caption_state['caption'] = result["corrected"]
                    caption_state['corrections'] = result["corrections"]

            if 'settings' in data:
                caption_state['settings'].update(data['settings'])

            self.send_json({"status": "ok", "corrections": caption_state.get("corrections", [])})
        
        elif path == '/api/clear':
            whisper_buffer = ""
            speechmatics_buffer = ""
            local_whisper_api_buffer = ""
            browser_caption_buffer = ""
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

        # RAG Engine POST endpoints (v4.0)
        elif path == '/api/engine/rag/enable':
            caption_engine.set_rag_enabled(data.get('enabled', True))
            self.send_json({"status": "ok", "rag_enabled": caption_engine.rag_enabled})

        elif path == '/api/engine/suggestion/accept':
            from_text = data.get('from', '')
            to_text = data.get('to', '')
            if from_text and to_text:
                caption_engine.accept_suggestion(from_text, to_text)
                self.send_json({"status": "ok"})
            else:
                self.send_json({"error": "Missing from/to text"}, 400)

        elif path == '/api/engine/suggestion/reject':
            from_text = data.get('from', '')
            to_text = data.get('to', '')
            if from_text and to_text:
                caption_engine.reject_suggestion(from_text, to_text)
                self.send_json({"status": "ok"})
            else:
                self.send_json({"error": "Missing from/to text"}, 400)

        elif path == '/api/engine/learned/clear':
            caption_engine.learner.clear_patterns()
            self.send_json({"status": "ok"})

        elif path == '/api/engine/aliases/generate':
            term = data.get('term', '')
            category = data.get('category', 'other')
            if term:
                aliases = caption_engine.generate_aliases(term, category)
                self.send_json({"status": "ok", "aliases": aliases})
            else:
                self.send_json({"error": "No term provided"}, 400)

        elif path == '/api/session/refine':
            # Post-session refinement - analyze transcript for consistency
            captions = session_manager.get_captions()
            result = caption_engine.refine_session(captions)
            self.send_json(result)

        elif path == '/api/session/refine/apply':
            # Apply a specific refinement
            from_text = data.get('from', '')
            to_text = data.get('to', '')
            if from_text and to_text:
                captions = session_manager.get_captions()
                refined = caption_engine.apply_refinement(captions, from_text, to_text)
                session_manager.captions = refined
                self.send_json({"status": "ok", "refined_count": len(refined)})
            else:
                self.send_json({"error": "Missing from/to text"}, 400)

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
            # Convert device_id to integer if it's a string
            device_id = data.get('device_id')
            if device_id is not None and isinstance(device_id, str):
                try:
                    device_id = int(device_id)
                except (ValueError, TypeError):
                    device_id = None
            result = whisper_engine.start(device_id, on_whisper_text)
            if result.get("status") == "started":
                caption_state["mode"] = "whisper"
                caption_state["whisper_running"] = True
            self.send_json(result)
        
        elif path == '/api/whisper/stop':
            result = whisper_engine.stop()
            caption_state["whisper_running"] = False
            self.send_json(result)

        elif path == '/api/whisper/mode':
            # Set real-time vs accuracy mode
            realtime = data.get('realtime', True)
            result = whisper_engine.set_realtime_mode(realtime)
            self.send_json(result)

        elif path == '/api/whisper/settings':
            # Update Whisper settings
            if 'min_chunk_duration' in data:
                whisper_engine.min_chunk_duration = float(data['min_chunk_duration'])
            if 'max_chunk_duration' in data:
                whisper_engine.max_chunk_duration = float(data['max_chunk_duration'])
            if 'silence_threshold' in data:
                whisper_engine.silence_threshold = float(data['silence_threshold'])
            self.send_json({
                "status": "ok",
                "min_chunk_duration": whisper_engine.min_chunk_duration,
                "max_chunk_duration": whisper_engine.max_chunk_duration,
                "silence_threshold": whisper_engine.silence_threshold
            })

        elif path == '/api/whisper/latency':
            # Quick latency adjustment
            min_ms = data.get('min_ms', 400)
            max_ms = data.get('max_ms', 1500)
            result = whisper_engine.set_latency(min_ms, max_ms)
            self.send_json(result)

        # Speechmatics controls
        elif path == '/api/speechmatics/status':
            self.send_json(speechmatics_engine.get_status())

        elif path == '/api/speechmatics/config':
            api_key = data.get('api_key')
            if api_key:
                result = speechmatics_engine.set_api_key(api_key)
                self.send_json(result)
            else:
                self.send_json({"error": "API key required"}, 400)

        elif path == '/api/speechmatics/start':
            # Convert device_id to integer if it's a string
            device_id = data.get('device_id')
            if device_id is not None and isinstance(device_id, str):
                try:
                    device_id = int(device_id)
                except (ValueError, TypeError):
                    device_id = None
            result = speechmatics_engine.start(device_id=device_id, callback=on_speechmatics_text)
            if "error" not in result:
                caption_state["mode"] = "speechmatics"
                caption_state["speechmatics_running"] = True
            self.send_json(result)

        elif path == '/api/speechmatics/stop':
            result = speechmatics_engine.stop()
            caption_state["speechmatics_running"] = False
            self.send_json(result)

        elif path == '/api/speechmatics/latency':
            max_latency = data.get('max_latency_ms', 2000)
            result = speechmatics_engine.set_max_latency(max_latency)
            self.send_json(result)

        # Local Whisper API controls
        elif path == '/api/localwhisper/config':
            api_url = data.get('api_url')
            api_key = data.get('api_key', '')
            if api_url:
                result = local_whisper_api_engine.set_config(api_url, api_key)
                self.send_json(result)
            else:
                self.send_json({"error": "API URL required"}, 400)

        elif path == '/api/localwhisper/start':
            local_whisper_api_buffer = ""
            # Convert device_id to integer if it's a string
            device_id = data.get('device_id')
            if device_id is not None and isinstance(device_id, str):
                try:
                    device_id = int(device_id)
                except (ValueError, TypeError):
                    device_id = None
            result = local_whisper_api_engine.start(device_id=device_id, callback=on_local_whisper_api_text)
            if "error" not in result:
                caption_state["mode"] = "localwhisper"
                caption_state["localwhisper_running"] = True
            self.send_json(result)

        elif path == '/api/localwhisper/stop':
            result = local_whisper_api_engine.stop()
            caption_state["localwhisper_running"] = False
            self.send_json(result)

        elif path == '/api/localwhisper/latency':
            max_latency = data.get('max_latency_ms', 2000)
            result = local_whisper_api_engine.set_max_latency(max_latency)
            self.send_json(result)

        # Latency management for all modes
        elif path == '/api/latency/config':
            max_latency = data.get('max_latency_ms', 2000)
            latency_manager.set_max_latency(max_latency)
            speechmatics_engine.latency_manager.set_max_latency(max_latency)
            local_whisper_api_engine.latency_manager.set_max_latency(max_latency)
            self.send_json({"status": "ok", "max_latency_ms": max_latency})

        elif path == '/api/latency/stats':
            self.send_json({
                "global": latency_manager.get_stats(),
                "speechmatics": speechmatics_engine.latency_manager.get_stats(),
                "localwhisper": local_whisper_api_engine.latency_manager.get_stats()
            })

        # Session controls
        elif path == '/api/session/start':
            # Clear previous caption state when starting new session
            caption_state["caption"] = ""
            caption_state["raw_caption"] = ""
            caption_state["corrections"] = []

            # Reset engine stats for fresh session
            caption_engine.reset_session_stats()

            session = session_manager.start_session(
                name=data.get('name'),
                record_audio=data.get('record_audio', True),  # Default to True for Whisper second pass
                audio_device=data.get('audio_device')
            )
            self.send_json({"status": "ok", "session": session})

        elif path == '/api/session/stop':
            session = session_manager.stop_session()
            self.send_json({"status": "ok", "session": session})

        elif path == '/api/sessions/delete':
            session_id = data.get('session_id')
            if session_id:
                if session_manager.delete_session(session_id):
                    self.send_json({"status": "ok"})
                else:
                    self.send_json({"error": "Session not found"}, 404)
            else:
                self.send_json({"error": "Session ID required"}, 400)

        elif path == '/api/session/reprocess':
            # Reprocess session with Whisper for higher accuracy
            if not session_manager.has_audio():
                self.send_json({"error": "No audio recording available for this session"}, 400)
                return

            if not whisper_engine.model:
                self.send_json({"error": "Whisper model not loaded. Load a model first."}, 400)
                return

            audio_file = session_manager.get_audio_file()
            if not audio_file or not Path(audio_file).exists():
                self.send_json({"error": "Audio file not found"}, 404)
                return

            try:
                # Load audio and transcribe with Whisper
                import wave
                with wave.open(audio_file, 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe with Whisper using HIGH ACCURACY settings for post-processing
                # Uses beam_size=10 and word_timestamps for best quality
                segments, _ = whisper_engine.model.transcribe(
                    audio_data,
                    beam_size=10,  # Higher beam size for better accuracy
                    language="en",
                    vad_filter=True,
                    word_timestamps=True  # More detailed timing
                )

                # Build new captions from Whisper output
                whisper_captions = []
                for seg in segments:
                    text = seg.text.strip()
                    if text:
                        # Apply caption engine corrections
                        result = caption_engine.correct(text)
                        whisper_captions.append({
                            "timestamp": seg.start,
                            "time_str": session_manager._format_time(seg.start),
                            "raw": text,
                            "corrected": result["corrected"],
                            "corrections": result["corrections"],
                            "source": "whisper_reprocess"
                        })

                # Compare with original captions
                original_count = len(session_manager.captions)
                whisper_count = len(whisper_captions)

                self.send_json({
                    "status": "ok",
                    "original_captions": original_count,
                    "whisper_captions": whisper_count,
                    "captions": whisper_captions,
                    "message": f"Reprocessed {whisper_count} segments with Whisper"
                })

            except Exception as e:
                self.send_json({"error": f"Reprocessing failed: {str(e)}"}, 500)

        elif path == '/api/session/reprocess/apply':
            # Apply reprocessed captions to session
            captions = data.get('captions', [])
            if captions:
                session_manager.captions = captions
                self.send_json({"status": "ok", "caption_count": len(captions)})
            else:
                self.send_json({"error": "No captions provided"}, 400)

        elif path == '/api/session/analytics':
            # POST handler for AI-powered session analytics
            # Accepts OpenAI API key in header for client-side key storage
            api_key = self.headers.get('X-OpenAI-Key', '')
            captions = data.get('captions', session_manager.get_captions())
            transcript = data.get('transcript', '')

            if not transcript and captions:
                transcript = ' '.join(
                    c.get('corrected', c.get('text', c.get('raw', '')))
                    for c in captions if isinstance(c, dict)
                )

            if not transcript:
                self.send_json({"error": "No transcript data available"}, 400)
                return

            # Use provided API key or fall back to configured client
            client = None
            if api_key and OPENAI_AVAILABLE:
                try:
                    client = OpenAI(api_key=api_key)
                except Exception as e:
                    self.send_json({"error": f"Invalid API key: {e}"}, 400)
                    return
            elif openai_client:
                client = openai_client

            if not client:
                self.send_json({"error": "No AI provider available. Please provide an OpenAI API key."}, 400)
                return

            try:
                # Generate comprehensive AI analysis
                response = client.chat.completions.create(
                    model=get_ai_model() if client == openai_client else "gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Analyze this meeting/event transcript comprehensively. Provide:\n"
                                "1. summary: A 3-5 sentence summary of the session\n"
                                "2. highlights: Array of 5 key moments with {title, description, timestamp_estimate}\n"
                                "3. topics: Array of main topics with {name, percentage} (should sum to ~100)\n"
                                "4. sentiment: Overall tone analysis {overall: positive/neutral/negative, score: -1 to 1}\n"
                                "5. speakers: List of people mentioned or speaking\n"
                                "6. action_items: Any action items or decisions made\n\n"
                                "Return ONLY valid JSON."
                            )
                        },
                        {
                            "role": "user",
                            "content": transcript[:12000]  # Limit context
                        }
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )

                try:
                    result = json.loads(response.choices[0].message.content)
                    self.send_json(result)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    content = response.choices[0].message.content
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        try:
                            result = json.loads(json_match.group())
                            self.send_json(result)
                        except:
                            self.send_json({"error": "Failed to parse AI response", "raw": content}, 500)
                    else:
                        self.send_json({"error": "Failed to parse AI response", "raw": content}, 500)

            except Exception as e:
                self.send_json({"error": f"AI analysis failed: {str(e)}"}, 500)

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

        # Knowledge Base endpoints (POST)
        elif path == '/api/knowledge/upload':
            # Handle file upload for knowledge base
            # For now, expect base64-encoded file or text
            filename = data.get('filename', 'document.txt')
            content = data.get('content')  # base64 encoded
            text = data.get('text')  # or plain text

            if content:
                import base64
                try:
                    file_bytes = base64.b64decode(content)
                    result = knowledge_base.add_document(filename, content=file_bytes)
                    self.send_json(result)
                except Exception as e:
                    self.send_json({"error": str(e)}, 400)
            elif text:
                result = knowledge_base.add_document(filename, text=text)
                self.send_json(result)
            else:
                self.send_json({"error": "No content or text provided"}, 400)

        elif path == '/api/knowledge/remove':
            doc_id = data.get('doc_id', '')
            if doc_id:
                result = knowledge_base.remove_document(doc_id)
                self.send_json(result)
            else:
                self.send_json({"error": "No doc_id provided"}, 400)

        elif path == '/api/knowledge/sync':
            # Sync all entities from knowledge base to caption engine
            result = knowledge_base.sync_entities_to_engine(caption_engine)
            self.send_json(result)

        elif path == '/api/knowledge/search':
            query = data.get('query', '')
            top_k = data.get('top_k', 5)
            if query:
                results = knowledge_base.search(query, top_k)
                self.send_json({"results": results})
            else:
                self.send_json({"error": "No query provided"}, 400)

        elif path == '/api/knowledge/import-url':
            # Import text from a URL
            url = data.get('url', '')
            if not url:
                self.send_json({"error": "No URL provided"}, 400)
            else:
                text, error = knowledge_base.extract_text_from_url(url)
                if error:
                    self.send_json({"error": error}, 400)
                else:
                    # Extract domain for filename
                    import urllib.parse
                    parsed = urllib.parse.urlparse(url)
                    filename = f"{parsed.netloc.replace('.', '_')}.txt"
                    result = knowledge_base.add_document(filename, text=text)
                    result['source'] = 'url'
                    result['url'] = url
                    self.send_json(result)

        elif path == '/api/knowledge/import-youtube':
            # Import transcript from YouTube video
            url = data.get('url', '')
            if not url:
                self.send_json({"error": "No URL provided"}, 400)
            else:
                text, error = knowledge_base.extract_youtube_transcript(url)
                if error:
                    self.send_json({"error": error}, 400)
                else:
                    # Extract video ID for filename
                    import re
                    video_id = re.search(r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
                    video_id = video_id.group(1) if video_id else 'video'
                    filename = f"youtube_{video_id}.txt"
                    result = knowledge_base.add_document(filename, text=text)
                    result['source'] = 'youtube'
                    result['url'] = url
                    self.send_json(result)

        # GPT Post-Processor endpoints (POST)
        elif path == '/api/gpt/enable':
            gpt_processor.set_enabled(data.get('enabled', True))
            self.send_json({"status": "ok", "enabled": gpt_processor.enabled})

        elif path == '/api/gpt/process':
            text = data.get('text', '')
            if text:
                result = gpt_processor.process(text)
                self.send_json(result)
            else:
                self.send_json({"error": "No text provided"}, 400)

        elif path == '/api/gpt/extract-entities':
            text = data.get('text', '')
            if text:
                entities = gpt_processor.extract_entities(text)
                self.send_json({"entities": entities})
            else:
                self.send_json({"error": "No text provided"}, 400)

        elif path == '/api/gpt/clear':
            gpt_processor.clear()
            self.send_json({"status": "ok"})

        # Video Intelligence endpoints (POST)
        elif path == '/api/video/upload':
            # Handle video file upload (base64 encoded)
            filename = data.get('filename', 'video.mp4')
            content = data.get('content')  # base64 encoded

            if content:
                import base64
                try:
                    video_bytes = base64.b64decode(content)
                    result = video_intelligence.upload_video(filename, video_bytes)
                    self.send_json(result)
                except Exception as e:
                    self.send_json({"error": str(e)}, 400)
            else:
                self.send_json({"error": "No video content provided"}, 400)

        elif path == '/api/video/highlights':
            # Generate highlight moments from current session transcript
            captions = session_manager.get_captions()
            video_path = data.get('video_path') or (
                video_intelligence.current_video.get('path') if video_intelligence.current_video else None
            )
            result = video_intelligence.generate_highlights(captions, video_path)
            self.send_json(result)

        elif path == '/api/video/clip':
            # Extract a single clip from video
            video_path = data.get('video_path') or (
                video_intelligence.current_video.get('path') if video_intelligence.current_video else None
            )
            start_time = data.get('start', '00:00:00')
            end_time = data.get('end', '00:00:30')

            if not video_path:
                self.send_json({"error": "No video loaded"}, 400)
                return

            result = video_intelligence.extract_clip(video_path, start_time, end_time)
            self.send_json(result)

        elif path == '/api/video/highlight-reel':
            # Generate full highlight reel
            video_path = data.get('video_path') or (
                video_intelligence.current_video.get('path') if video_intelligence.current_video else None
            )
            highlights = data.get('highlights', [])

            if not video_path:
                self.send_json({"error": "No video loaded"}, 400)
                return

            if not highlights:
                # Auto-generate highlights from transcript
                captions = session_manager.get_captions()
                hl_result = video_intelligence.generate_highlights(captions, video_path)
                if hl_result.get('status') == 'ok':
                    highlights = hl_result.get('highlights', [])
                else:
                    self.send_json(hl_result)
                    return

            # Get additional options
            aspect_ratio = data.get('aspect_ratio', '16:9')
            burn_captions = data.get('burn_captions', False)
            target_duration = data.get('duration', 60)

            result = video_intelligence.generate_highlight_reel(
                video_path, highlights,
                aspect_ratio=aspect_ratio,
                burn_captions=burn_captions,
                target_duration=target_duration
            )
            self.send_json(result)

        elif path == '/api/video/youtube-transcript':
            # Fetch YouTube transcript using multiple methods
            url = data.get('url', '')
            if not url:
                self.send_json({"error": "No URL provided"}, 400)
                return

            import re

            # Extract video ID from URL
            video_id = None
            patterns = [
                r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
                r'(?:embed/)([0-9A-Za-z_-]{11})',
                r'(?:youtu\.be/)([0-9A-Za-z_-]{11})'
            ]
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    video_id = match.group(1)
                    break

            if not video_id:
                self.send_json({"error": "Could not extract video ID from URL"}, 400)
                return

            transcript_text = None
            transcript_list = []
            duration = 0
            method_used = None

            # Method 1: Try youtube-transcript-api first (v1.x API with .fetch())
            if YOUTUBE_TRANSCRIPT_AVAILABLE:
                try:
                    from youtube_transcript_api import YouTubeTranscriptApi

                    # New API: YouTubeTranscriptApi().fetch(video_id) returns FetchedTranscript
                    fetched = YouTubeTranscriptApi().fetch(video_id)
                    method_used = "youtube-transcript-api"

                    # Extract text from snippets
                    transcript_text = ' '.join([snippet.text for snippet in fetched.snippets])

                    # Calculate duration from last snippet
                    if fetched.snippets:
                        last_snippet = fetched.snippets[-1]
                        duration = last_snippet.start + last_snippet.duration

                except Exception as e:
                    print(f"youtube-transcript-api failed: {e}")

            # Method 2: Direct HTTP fetch of YouTube's timedtext API (no dependencies)
            if not transcript_text:
                try:
                    import urllib.request
                    import xml.etree.ElementTree as ET

                    # First, fetch the video page to get caption track info
                    video_url = f'https://www.youtube.com/watch?v={video_id}'
                    req = urllib.request.Request(video_url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept-Language': 'en-US,en;q=0.9'
                    })
                    with urllib.request.urlopen(req, timeout=15) as response:
                        html = response.read().decode('utf-8')

                    # Extract timedtext URL from page source
                    # Look for the captionTracks in the ytInitialPlayerResponse
                    caption_match = re.search(r'"captionTracks":\s*\[(.*?)\]', html)
                    if caption_match:
                        caption_tracks = caption_match.group(1)
                        # Find the baseUrl for captions
                        base_url_match = re.search(r'"baseUrl":\s*"([^"]+)"', caption_tracks)
                        if base_url_match:
                            caption_url = base_url_match.group(1).replace('\\u0026', '&')

                            # Fetch the actual caption XML/JSON
                            caption_req = urllib.request.Request(caption_url, headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            })
                            with urllib.request.urlopen(caption_req, timeout=15) as caption_response:
                                caption_content = caption_response.read().decode('utf-8')

                            # Parse XML format captions
                            if caption_content.strip().startswith('<?xml') or caption_content.strip().startswith('<transcript'):
                                root = ET.fromstring(caption_content)
                                text_parts = []
                                for text_elem in root.findall('.//text'):
                                    text = text_elem.text
                                    if text:
                                        # Clean up HTML entities
                                        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                                        text = text.replace('&#39;', "'").replace('&quot;', '"')
                                        text_parts.append(text)
                                if text_parts:
                                    transcript_text = ' '.join(text_parts)
                                    method_used = "youtube-direct-api"
                                    print(f"✅ Fetched transcript via direct YouTube API")
                            else:
                                # Try JSON format
                                try:
                                    caption_json = json.loads(caption_content)
                                    events = caption_json.get('events', [])
                                    text_parts = []
                                    for event in events:
                                        segs = event.get('segs', [])
                                        for seg in segs:
                                            text = seg.get('utf8', '')
                                            if text and text.strip():
                                                text_parts.append(text)
                                    if text_parts:
                                        transcript_text = ' '.join(text_parts)
                                        method_used = "youtube-direct-api-json"
                                        print(f"✅ Fetched transcript via direct YouTube API (JSON)")
                                except:
                                    pass

                except Exception as e:
                    print(f"Direct YouTube caption fetch failed: {e}")

            # Method 3: Try yt-dlp if available and previous methods failed
            if not transcript_text:
                try:
                    result = subprocess.run([
                        'yt-dlp', '--write-auto-sub', '--write-sub', '--skip-download',
                        '--sub-format', 'vtt', '--sub-lang', 'en',
                        '-o', f'/tmp/yt_{video_id}',
                        f'https://www.youtube.com/watch?v={video_id}'
                    ], capture_output=True, text=True, timeout=60)

                    # Look for downloaded subtitle files
                    import glob
                    sub_files = glob.glob(f'/tmp/yt_{video_id}*.vtt')
                    if sub_files:
                        with open(sub_files[0], 'r') as f:
                            vtt_content = f.read()
                        # Parse VTT to extract text
                        lines = vtt_content.split('\n')
                        text_lines = []
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('WEBVTT') and '-->' not in line and not line.isdigit():
                                # Remove VTT tags
                                clean_line = re.sub(r'<[^>]+>', '', line)
                                if clean_line:
                                    text_lines.append(clean_line)
                        transcript_text = ' '.join(text_lines)
                        method_used = "yt-dlp"
                        # Clean up
                        for f in sub_files:
                            try:
                                Path(f).unlink()
                            except:
                                pass
                except Exception as e:
                    print(f"yt-dlp subtitle extraction failed: {e}")

            # Method 4: Use yt-dlp to get video info for duration
            if not duration:
                try:
                    result = subprocess.run([
                        'yt-dlp', '--dump-json', '--skip-download',
                        f'https://www.youtube.com/watch?v={video_id}'
                    ], capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        video_info = json.loads(result.stdout)
                        duration = video_info.get('duration', 0)
                except:
                    pass

            if transcript_text:
                self.send_json({
                    "status": "ok",
                    "transcript": transcript_text,
                    "segments": transcript_list,
                    "video_info": {
                        "video_id": video_id,
                        "duration": duration,
                        "title": f"YouTube Video {video_id}"
                    },
                    "method": method_used
                })
            else:
                self.send_json({
                    "error": "Could not fetch transcript. This video may not have captions enabled, or the captions are restricted. You can try: 1) Verify the video has CC/subtitles on YouTube, 2) Upload the video file directly and use Whisper transcription instead."
                }, 400)

        elif path == '/api/video/transcribe':
            # Transcribe video using Whisper
            video_path = data.get('video_path') or (
                video_intelligence.current_video.get('path') if video_intelligence.current_video else None
            )

            if not video_path:
                self.send_json({"error": "No video loaded. Please upload a video first."}, 400)
                return

            if not Path(video_path).exists():
                self.send_json({"error": "Video file not found"}, 404)
                return

            if not WHISPER_AVAILABLE:
                self.send_json({"error": "Whisper not installed. Run: pip3 install faster-whisper"}, 400)
                return

            if not whisper_engine.model:
                self.send_json({"error": "Whisper model not loaded. Load a model from the dashboard first."}, 400)
                return

            try:
                # Extract audio from video using ffmpeg
                import subprocess
                import tempfile

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    audio_path = tmp.name

                # Extract audio
                result = subprocess.run([
                    'ffmpeg', '-y', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    audio_path
                ], capture_output=True, text=True)

                if result.returncode != 0:
                    self.send_json({"error": f"Audio extraction failed: {result.stderr}"}, 500)
                    return

                # Load audio and transcribe
                import wave
                with wave.open(audio_path, 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe with Whisper using HIGH ACCURACY settings for video transcription
                # Uses beam_size=10 and word_timestamps for best quality
                segments, info = whisper_engine.model.transcribe(
                    audio_data,
                    beam_size=10,  # Higher beam size for better accuracy
                    language="en",
                    vad_filter=True,
                    word_timestamps=True  # More detailed timing for video sync
                )

                # Build transcript with timestamps
                transcript_lines = []
                captions = []
                for seg in segments:
                    text = seg.text.strip()
                    if text:
                        # Apply caption engine corrections
                        corrected = caption_engine.correct(text)
                        timestamp = f"{int(seg.start // 60):02d}:{int(seg.start % 60):02d}"
                        transcript_lines.append(f"[{timestamp}] {corrected['corrected']}")
                        captions.append({
                            "timestamp": seg.start,
                            "end": seg.end,
                            "text": text,
                            "corrected": corrected['corrected'],
                            "corrections": corrected['corrections']
                        })

                # Clean up temp file
                try:
                    Path(audio_path).unlink()
                except:
                    pass

                self.send_json({
                    "status": "ok",
                    "transcript": "\n".join(transcript_lines),
                    "captions": captions,
                    "duration": info.duration if hasattr(info, 'duration') else 0,
                    "language": info.language if hasattr(info, 'language') else 'en'
                })

            except Exception as e:
                self.send_json({"error": f"Transcription failed: {str(e)}"}, 500)

        elif path == '/api/video/download-youtube':
            # Download YouTube video with progress tracking
            url = data.get('url', '')
            if not url:
                self.send_json({"error": "No URL provided"}, 400)
                return

            # Start download (this is synchronous but updates progress)
            result = video_intelligence.download_youtube_video(url)
            self.send_json(result)

        elif path == '/api/video/download-progress':
            # Get current download progress
            progress = video_intelligence.get_download_progress()
            self.send_json(progress)

        elif path == '/api/video/set-transcript':
            # Set transcript for standalone video (without captioning session)
            transcript = data.get('transcript', '')
            if not transcript:
                self.send_json({"error": "No transcript provided"}, 400)
                return

            video_intelligence.set_transcript(transcript)
            self.send_json({
                "status": "ok",
                "segments": len(video_intelligence.transcript_segments)
            })

        elif path == '/api/video/generate-highlights':
            # Generate highlights from transcript text (works without captioning session)
            transcript = data.get('transcript', '')
            if not transcript:
                # Try to use stored transcript
                transcript = video_intelligence.transcript
            if not transcript:
                self.send_json({"error": "No transcript provided"}, 400)
                return

            padding = data.get('padding', 2.0)
            result = video_intelligence.generate_highlights_from_transcript(transcript, clip_padding=padding)
            self.send_json(result)

        elif path == '/api/video/generate-captions':
            # Generate caption file from transcript
            format = data.get('format', 'srt')
            result = video_intelligence.generate_caption_file(format)

            if result.get('status') == 'ok':
                # Return file path for download
                self.send_json(result)
            else:
                self.send_json(result, 400)

        elif path == '/api/video/download-file':
            # Download a generated file (captions, clips, etc.)
            filepath = data.get('path', '')
            if not filepath or not Path(filepath).exists():
                self.send_json({"error": "File not found"}, 404)
                return

            try:
                with open(filepath, 'rb') as f:
                    content = f.read()

                filename = Path(filepath).name
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
                self.send_header('Content-Length', len(content))
                self.end_headers()
                self.wfile.write(content)
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
    embeddings_ok = EMBEDDINGS_AVAILABLE
    ffmpeg_ok = FFMPEG_AVAILABLE
    rag_status = "✅ Active" if caption_engine.rag_enabled else "⚪ Disabled"

    # Detect platform for display
    import platform
    is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
    if whisper_engine.gpu_available:
        gpu_status = "🚀 GPU"
    elif is_apple_silicon:
        gpu_status = "🍎 M-chip"
    else:
        gpu_status = "💻 CPU"

    latency_info = f"{int(whisper_engine.min_chunk_duration*1000)}-{int(whisper_engine.max_chunk_duration*1000)}ms"
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          🎤 COMMUNITY CAPTIONER v4.2.4 - Ultra-Low Latency Edition 🎤        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CAPTIONING MODES:                                                           ║
║    ⚡ Browser Speech      - Instant (~200ms), requires Chrome/Edge           ║
║    🎯 Whisper AI          - Local transcription ({"✅ Ready" if whisper_ok else "❌ pip3 install faster-whisper":<24})  ║
║       └── {gpu_status} mode, {latency_info} chunks, tiny.en default                     ║
║    ☁️  Speechmatics       - Cloud ASR (~300ms), requires API key             ║
║                                                                              ║
║  AI CAPTION ENGINE:                                                          ║
║    🧠 RAG Corrections     - Semantic matching ({rag_status:<24})   ║
║    📚 Vector Embeddings   - Context-aware ({"✅ Ready" if embeddings_ok else "⚪ Optional":<24})  ║
║    🔄 Real-time Learning  - Improves from corrections                        ║
║                                                                              ║
║  FEATURES:                                                                   ║
║    📝 Session Recording   - Audio + captions, SRT/VTT/TXT export             ║
║    📊 Analytics Dashboard - Word cloud, sentiment, topic analysis            ║
║    🎬 Video Intelligence  - Highlight detection ({"✅ Ready" if ffmpeg_ok else "❌ Install ffmpeg":<24})  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Control Panel:  http://localhost:{PORT}                                      ║
║  OBS Overlay:    http://localhost:{PORT}?overlay=true                         ║
║  Network:        http://{LOCAL_IP}:{PORT:<38}  ║
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
