from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence

from .metrics import MetricsEngine
from .models import TranscriptEvaluation
from .parsing import TranscriptParser
from .phase_detection import PhaseDetector
from .scoring import ScoreCalculator


class WorkflowEvaluator:
    """Orchestrates transcript parsing, phase labeling, metric extraction, and scoring."""

    def __init__(
        self,
        parser: TranscriptParser,
        phase_detector: PhaseDetector,
        metrics_engine: MetricsEngine,
        score_calculator: ScoreCalculator,
    ) -> None:
        self.parser = parser
        self.phase_detector = phase_detector
        self.metrics_engine = metrics_engine
        self.score_calculator = score_calculator

    def evaluate_file(self, path: Path) -> TranscriptEvaluation:
        try:
            transcript = self.parser.parse_file(path)
            self.phase_detector.annotate(transcript)
            metrics = self.metrics_engine.compute(transcript)
            scores, overall_score, overall_confidence = self.score_calculator.score(metrics)
            recommendations = self.score_calculator.recommendations(metrics, scores)

            timeline = self.phase_detector.build_timeline(transcript.turns)
            normalized_turns = [turn.to_dict() for turn in transcript.turns]
            analysis_artifacts = {
                "phase_transition_edges": metrics.get("phase_transition_edges", []),
                "speaker_transition_edges": metrics.get("speaker_transition_edges", []),
                "workflow_findings": metrics.get("workflow_findings", []),
                "security_findings": metrics.get("security_findings", []),
            }

            return TranscriptEvaluation(
                transcript=transcript.name,
                metrics=metrics,
                scores=scores,
                overall_score=overall_score,
                overall_confidence=overall_confidence,
                timeline=timeline,
                normalized_turns=normalized_turns,
                recommendations=recommendations,
                analysis_artifacts=analysis_artifacts,
            )

        except Exception as exc:  # pragma: no cover - defensive safety net
            return TranscriptEvaluation(
                transcript=path.name,
                metrics={},
                scores={},
                overall_score=0.0,
                overall_confidence=0.0,
                timeline="",
                errors=[str(exc)],
                normalized_turns=[],
                recommendations=[],
                analysis_artifacts={},
            )

    def evaluate_files(self, paths: Sequence[Path]) -> List[TranscriptEvaluation]:
        return list(self.iter_evaluations(paths))

    def iter_evaluations(self, paths: Sequence[Path]) -> Iterator[TranscriptEvaluation]:
        for path in paths:
            yield self.evaluate_file(path)
