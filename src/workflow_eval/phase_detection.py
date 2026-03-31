from __future__ import annotations

import re
from typing import Dict, Iterable, List

from .models import Transcript, Turn

WORD_PATTERN = re.compile(r"\b[\w'-]+\b")


class PhaseDetector:
    """Assign coarse workflow phases to each transcript turn using deterministic heuristics."""

    KEYWORDS: Dict[str, Iterable[str]] = {
        "planning": (
            "plan",
            "architecture",
            "design",
            "approach",
            "strategy",
            "requirements",
            "outline",
        ),
        "implementation": (
            "implement",
            "write code",
            "function",
            "class",
            "refactor",
            "build",
            "code",
            "module",
            "script",
            "api",
        ),
        "debugging": (
            "error",
            "bug",
            "fix",
            "traceback",
            "fails",
            "not working",
            "issue",
            "exception",
            "stack",
        ),
        "testing": (
            "test",
            "unit test",
            "integration",
            "assert",
            "coverage",
            "pytest",
            "verify",
        ),
        "review": (
            "review",
            "security",
            "audit",
            "improve",
            "optimize",
            "cleanup",
            "harden",
            "performance",
        ),
    }

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

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

        normalized_text = self._normalize_text(text)
        token_set = set(WORD_PATTERN.findall(normalized_text))

        scores: Dict[str, int] = {}
        for phase, keywords in self.KEYWORDS.items():
            scores[phase] = sum(
                self._keyword_weight(keyword, normalized_text, token_set)
                for keyword in keywords
            )

        # Reward continuity to prevent noisy phase oscillation when lexical signals are weak.
        if previous_phase in scores:
            scores[previous_phase] += 1

        best_phase = max(scores, key=lambda phase_name: scores[phase_name])
        if scores[best_phase] > 0:
            return best_phase

        if turn.speaker == "Tool":
            return "implementation"

        return "discussion"

    @staticmethod
    def _stem(word: str) -> str:
        word = word.lower()
        for suffix in ("ing", "ed", "es", "s", "ion"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

    @staticmethod
    def _stem(word: str) -> str:
        word = word.lower()
        for suffix in ("ing", "ed", "es", "s", "ion"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join([PhaseDetector._stem(w) for w in WORD_PATTERN.findall(text.lower())])

    @staticmethod
    def _keyword_weight(keyword: str, normalized_text: str, token_set: set[str]) -> int:
        normalized_keyword = " ".join([PhaseDetector._stem(w) for w in WORD_PATTERN.findall(keyword.lower())])
        if not normalized_keyword:
            return 0
        if " " in normalized_keyword:
            return 2 if f" {normalized_keyword} " in f" {normalized_text} " else 0
        return 1 if normalized_keyword in token_set else 0

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
