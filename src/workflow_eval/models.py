from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    speaker: str
    content: str
    timestamp: Optional[str] = None
    code_blocks: List[str] = field(default_factory=list)
    code_block_languages: List[Optional[str]] = field(default_factory=list)
    phase: str = "discussion"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "speaker": self.speaker,
            "content": self.content,
            "timestamp": self.timestamp,
            "phase": self.phase,
            "code_blocks": self.code_blocks,
        }
        if self.code_block_languages:
            payload["code_block_languages"] = self.code_block_languages
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class Transcript:
    name: str
    source_path: str
    turns: List[Turn]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricScore:
    value: Any
    score: float
    confidence: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 2),
            "rationale": self.rationale,
        }


@dataclass
class TranscriptEvaluation:
    transcript: str
    metrics: Dict[str, Any]
    scores: Dict[str, MetricScore]
    overall_score: float
    overall_confidence: float
    timeline: str = ""
    errors: List[str] = field(default_factory=list)
    normalized_turns: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    analysis_artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_turns: bool = False, include_timeline: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "transcript": self.transcript,
            "metrics": self.metrics,
            "scores": {name: metric.to_dict() for name, metric in self.scores.items()},
            "overall_score": round(self.overall_score, 2),
            "overall_confidence": round(self.overall_confidence, 2),
            "errors": self.errors,
            "recommendations": self.recommendations,
        }

        if include_timeline and self.timeline:
            payload["phase_timeline"] = self.timeline

        if include_turns:
            payload["normalized_turns"] = self.normalized_turns

        if self.analysis_artifacts:
            payload["analysis_artifacts"] = self.analysis_artifacts

        return payload
