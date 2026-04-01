from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List

from .models import Transcript, Turn

WORD_PATTERN = re.compile(r"\b[\w'-]+\b")

class TFIDFVectorizer:
    def __init__(self):
        self.idf: Dict[str, float] = {}
        self.doc_count: int = 0
        self.category_vectors: Dict[str, Dict[str, float]] = {}

    def fit(self, categories: Dict[str, Iterable[str]]) -> None:
        self.doc_count = len(categories)
        doc_freqs: Dict[str, int] = Counter()
        docs: Dict[str, List[str]] = {}

        for cat, phrases in categories.items():
            tokens = []
            for phrase in phrases:
                tokens.extend(self._tokenize(phrase))
            docs[cat] = tokens
            for token in set(tokens):
                doc_freqs[token] += 1

        for word, freq in doc_freqs.items():
            self.idf[word] = math.log((self.doc_count + 1) / (freq + 1)) + 1.0

        for cat, tokens in docs.items():
            self.category_vectors[cat] = self._transform(tokens)

    def _transform(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        vec: Dict[str, float] = {}
        norm_sq = 0.0
        for w, count in tf.items():
            val = count * self.idf.get(w, 0.0)
            vec[w] = val
            norm_sq += val * val
        
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for w in vec:
                vec[w] /= norm
        return vec

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        for w in WORD_PATTERN.findall(text.lower()):
            for suffix in ("ing", "ed", "es", "s", "ion"):
                if w.endswith(suffix) and len(w) > len(suffix) + 2:
                    w = w[:-len(suffix)]
                    break
            tokens.append(w)
        return tokens

    def compute_similarities(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        input_vec = self._transform(tokens)
        
        scores: Dict[str, float] = {}
        for cat, cat_vec in self.category_vectors.items():
            sim = 0.0
            for w, val in input_vec.items():
                sim += val * cat_vec.get(w, 0.0)
            scores[cat] = sim
        return scores


class PhaseDetector:
    """Assign coarse workflow phases to each transcript turn using TF-IDF semantic heuristics."""

    KEYWORDS: Dict[str, Iterable[str]] = {
        "planning": (
            "plan", "architecture", "design", "approach", "strategy", 
            "requirements", "outline", "concept", "spec", "blueprint"
        ),
        "implementation": (
            "implement", "write code", "function", "class", "refactor", 
            "build", "code", "module", "script", "api", "logic", "develop"
        ),
        "debugging": (
            "error", "bug", "fix", "traceback", "fails", "not working", 
            "issue", "exception", "stack", "crash", "resolve", "patch"
        ),
        "testing": (
            "test", "unit test", "integration", "assert", "coverage", 
            "pytest", "verify", "mock", "validate", "fixture"
        ),
        "review": (
            "review", "security", "audit", "improve", "optimize", 
            "cleanup", "harden", "performance", "analyze", "evaluate"
        ),
    }

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.vectorizer = TFIDFVectorizer()
        self.vectorizer.fit(self.KEYWORDS)

    def annotate(self, transcript: Transcript) -> None:
        if not self.enabled:
            for turn in transcript.turns:
                turn.phase = "discussion"
            return

        previous_phase = "discussion"
        for turn in transcript.turns:
            turn.phase = self.detect_turn_phase(turn, previous_phase=previous_phase)
            previous_phase = turn.phase

    def detect_turn_phase(self, turn: Turn, previous_phase: str = "discussion") -> str:
        if turn.code_blocks and turn.speaker in {"AI", "Tool"}:
            return "implementation"

        text = turn.content.lower()
        if not text.strip():
            return "discussion"

        scores = self.vectorizer.compute_similarities(text)

        # Reward continuity to prevent noisy phase oscillation
        if previous_phase in scores:
            scores[previous_phase] += 0.1  # small boost

        best_phase = max(scores, key=lambda phase_name: scores[phase_name], default=None)
        if best_phase and scores[best_phase] > 0.05:  # threshold
            return best_phase

        if turn.speaker == "Tool":
            return "implementation"

        return "discussion"

    @staticmethod
    def build_timeline(turns: List[Turn]) -> str:
        marker_by_phase = {
            "planning": "P",
            "implementation": "I",
            "debugging": "D",
            "testing": "T",
            "review": "R",
            "discussion": "U",
        }
        return " ".join(marker_by_phase.get(turn.phase, "U") for turn in turns)
