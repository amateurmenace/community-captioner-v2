#!/usr/bin/env python3
"""
Community Captioner v4.0 - Advanced RAG Caption Engine

FEATURES:
  - Dual captioning modes (Web Speech / Whisper)
  - Advanced RAG Caption Engine with semantic similarity matching
  - Vector embeddings for context-aware corrections
  - Fuzzy matching with confidence thresholds
  - Real-time learning from ASR patterns
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

    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            # Try PyPDF2 first
            try:
                import PyPDF2
                text_parts = []
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text_parts.append(page.extract_text() or "")
                return "\n".join(text_parts)
            except ImportError:
                pass

            # Try pdfplumber as fallback
            try:
                import pdfplumber
                text_parts = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text_parts.append(page.extract_text() or "")
                return "\n".join(text_parts)
            except ImportError:
                pass

            return ""
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""

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
        """Extract entities from text using AI"""
        if not openai_client:
            return {"people": [], "places": [], "organizations": []}

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
                return json.loads(response.choices[0].message.content)
            except:
                return {"people": [], "places": [], "organizations": []}

        except Exception as e:
            print(f"Entity extraction error: {e}")
            return {"people": [], "places": [], "organizations": []}

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
        if not extracted_text and content:
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if ext == 'pdf':
                extracted_text = self.extract_text_from_pdf(file_path)
            elif ext in ['docx', 'doc']:
                extracted_text = self.extract_text_from_docx(file_path)
            elif ext in ['txt', 'md', 'csv']:
                extracted_text = content.decode('utf-8', errors='ignore')

        if not extracted_text:
            return {"error": "Could not extract text from document"}

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

    def start_session(self, name=None, record_audio=False, audio_device=None):
        """Start a new captioning session with optional audio recording"""
        session_id = str(uuid.uuid4())[:8]
        self.current_session = {
            "id": session_id,
            "name": name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "started": datetime.now().isoformat(),
            "mode": "browser",
            "audio_recorded": False,
            "audio_file": None
        }
        self.captions = []
        self.start_time = datetime.now()
        self.is_recording = True

        # Start audio recording if requested and available
        if record_audio and AUDIO_AVAILABLE and NUMPY_AVAILABLE:
            self._start_audio_recording(session_id, audio_device)
            self.current_session["audio_recorded"] = True
            self.current_session["audio_file"] = f"{session_id}.wav"

        return self.current_session

    def _start_audio_recording(self, session_id: str, device_id=None):
        """Start recording audio to file"""
        try:
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
    - Transcript-to-video sync
    - AI-powered highlight detection
    - Clip generation with ffmpeg
    - Highlight reel compilation
    """

    def __init__(self, video_dir: Path):
        self.video_dir = video_dir
        self.video_dir.mkdir(exist_ok=True)
        self.clips_dir = video_dir / "clips"
        self.clips_dir.mkdir(exist_ok=True)
        self.current_video = None

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

    def generate_highlights(self, captions: List[Dict], video_path: str = None) -> Dict:
        """Use AI to identify highlight moments from transcript"""
        if not captions:
            return {"error": "No captions to analyze"}

        if not openai_client:
            return {"error": "AI not available for highlight detection"}

        # Build full transcript with timestamps
        transcript = "\n".join([
            f"[{c.get('time_str', '')}] {c.get('corrected', c.get('raw', ''))}"
            for c in captions
        ])

        try:
            response = openai_client.chat.completions.create(
                model=get_ai_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Analyze this timestamped transcript and identify 5-10 highlight moments. "
                            "Look for: key decisions, votes, important announcements, heated discussions, "
                            "memorable quotes, or significant statements. "
                            "Return ONLY a JSON array of highlights:\n"
                            '[\n'
                            '  {"timestamp": "HH:MM:SS", "end_timestamp": "HH:MM:SS", "quote": "exact quote", '
                            '"reason": "why this is a highlight", "importance": 1-10}\n'
                            ']\n'
                            "Order by importance, highest first."
                        )
                    },
                    {
                        "role": "user",
                        "content": transcript[:10000]
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )

            try:
                highlights = json.loads(response.choices[0].message.content)
                return {
                    "status": "ok",
                    "highlights": highlights,
                    "count": len(highlights)
                }
            except:
                return {"error": "Failed to parse AI response", "raw": response.choices[0].message.content}

        except Exception as e:
            return {"error": str(e)}

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

    def generate_highlight_reel(self, video_path: str, highlights: List[Dict], output_name: str = None) -> Dict:
        """Create a highlight reel by concatenating clips"""
        if not FFMPEG_AVAILABLE:
            return {"error": "ffmpeg not available"}

        if not highlights:
            return {"error": "No highlights provided"}

        # First extract all clips
        clips = []
        for i, h in enumerate(highlights):
            start = h.get("timestamp", "00:00:00")
            end = h.get("end_timestamp", self._add_seconds(start, 30))

            clip_result = self.extract_clip(video_path, start, end, f"highlight_{i}.mp4")
            if clip_result.get("status") == "ok":
                clips.append(clip_result["path"])

        if not clips:
            return {"error": "No clips could be extracted"}

        # Create concat file
        concat_file = self.clips_dir / "concat_list.txt"
        with open(concat_file, 'w') as f:
            for clip in clips:
                f.write(f"file '{clip}'\n")

        # Generate highlight reel
        reel_id = str(uuid.uuid4())[:8]
        output_name = output_name or f"highlight_reel_{reel_id}.mp4"
        output_path = self.clips_dir / output_name

        try:
            result = subprocess.run([
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'fast',
                str(output_path)
            ], capture_output=True, text=True)

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
                    "path": str(output_path),
                    "filename": output_name,
                    "clips_count": len(clips)
                }
            else:
                return {"error": result.stderr}

        except Exception as e:
            return {"error": str(e)}

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
gpt_processor = GPTPostProcessor()
session_analytics = SessionAnalytics()
video_intelligence = VideoIntelligence(VIDEO_DIR)

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
                "embeddings_available": EMBEDDINGS_AVAILABLE,
                "rag_enabled": caption_engine.rag_enabled,
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
            session = session_manager.start_session(
                name=data.get('name'),
                record_audio=data.get('record_audio', False),
                audio_device=data.get('audio_device')
            )
            self.send_json({"status": "ok", "session": session})

        elif path == '/api/session/stop':
            session = session_manager.stop_session()
            self.send_json({"status": "ok", "session": session})
        
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

                # Transcribe with Whisper
                segments, _ = whisper_engine.model.transcribe(
                    audio_data,
                    beam_size=5,
                    language="en",
                    vad_filter=True
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

            result = video_intelligence.generate_highlight_reel(video_path, highlights)
            self.send_json(result)

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
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🎤 COMMUNITY CAPTIONER v4.0 - Advanced RAG Engine 🎤            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CORE FEATURES:                                                              ║
║    ⚡ Web Speech Mode     - Real-time browser captions (~200ms)              ║
║    🎯 Whisper Mode        - Accurate local AI ({"✅ Ready" if whisper_ok else "❌ pip3 install faster-whisper":<24})  ║
║    🧠 RAG Caption Engine  - Semantic matching ({rag_status:<24})   ║
║    📚 Vector Embeddings   - Context-aware ({"✅ Ready" if embeddings_ok else "⚪ Optional":<24})  ║
║                                                                              ║
║  AI FEATURES (v4.0):                                                         ║
║    🔄 Real-time Learning  - Improves from corrections                        ║
║    📝 Session Recording   - Audio + captions, SRT/VTT/TXT export             ║
║    📊 Analytics Dashboard - Word cloud, sentiment, topic analysis            ║
║    📁 Knowledge Base      - PDF/DOCX ingestion, entity extraction            ║
║    🎬 Video Intelligence  - Highlight detection ({"✅ Ready" if ffmpeg_ok else "❌ Install ffmpeg":<24})  ║
║    💾 Portable Engines    - Download/upload with learned patterns            ║
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
