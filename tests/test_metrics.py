import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from workflow_eval.metrics import MetricsEngine
from workflow_eval.metrics import MetricEvaluator
from workflow_eval.models import Transcript, Turn
from workflow_eval.scoring import ScoreCalculator


class MetricsEngineTests(unittest.TestCase):
    def _build_transcript(self) -> Transcript:
        turns = [
            Turn(speaker="User", content="Let's plan architecture first.", phase="planning"),
            Turn(speaker="AI", content="Plan sounds good.", phase="planning"),
            Turn(speaker="User", content="Now implement parser code.", phase="implementation"),
            Turn(
                speaker="AI",
                content="```python\ndef parse(value):\n    return value\n```",
                code_blocks=["def parse(value):\n    return value"],
                phase="implementation",
            ),
            Turn(speaker="User", content="Not quite, this fails. Fix this.", phase="debugging"),
            Turn(speaker="AI", content="I fixed the parsing bug.", phase="debugging"),
            Turn(speaker="User", content="Great, thanks.", phase="review"),
        ]
        return Transcript(name="demo", source_path="demo", turns=turns)

    def test_metrics_compute_expected_values(self) -> None:
        transcript = self._build_transcript()
        metrics = MetricsEngine().compute(transcript)

        self.assertEqual(7, metrics["total_turns"])
        self.assertEqual(4, metrics["user_turns"])
        self.assertEqual(3, metrics["ai_turns"])
        self.assertGreater(metrics["estimated_total_tokens"], 0)
        self.assertGreaterEqual(metrics["code_turn_count"], 1)
        self.assertIn("tool_call_turn_count", metrics)
        self.assertIn("context_retention_score", metrics)
        self.assertIn("phase_transition_matrix", metrics)
        self.assertIn("security_risk_count", metrics)
        self.assertIn("workflow_findings", metrics)

    def test_score_calculator_ranges(self) -> None:
        transcript = self._build_transcript()
        metrics = MetricsEngine().compute(transcript)
        calculator = ScoreCalculator()
        score_map, overall_score, overall_confidence = calculator.score(metrics)
        recommendations = calculator.recommendations(metrics, score_map)

        self.assertGreaterEqual(overall_score, 0)
        self.assertLessEqual(overall_score, 100)
        self.assertGreaterEqual(overall_confidence, 0)
        self.assertLessEqual(overall_confidence, 100)
        self.assertIn("correction_efficiency", score_map)
        self.assertIn("security_hygiene", score_map)
        self.assertGreaterEqual(len(recommendations), 1)

    def test_security_detection_avoids_cross_line_false_positive(self) -> None:
        transcript = Transcript(
            name="security-demo",
            source_path="demo",
            turns=[
                Turn(
                    speaker="AI",
                    content="```python\nimport subprocess\nsubprocess.run(['echo', 'hi'])\nconfig.shell = True\n```",
                    code_blocks=["import subprocess\nsubprocess.run(['echo', 'hi'])\nconfig.shell = True"],
                    code_block_languages=["python"],
                    phase="implementation",
                )
            ],
        )

        metrics = MetricsEngine().compute(transcript)

        self.assertEqual(0, metrics["security_risk_breakdown"].get("dangerous_subprocess", 0))

    def test_python_syntax_ratio_only_uses_python_candidates(self) -> None:
        transcript = Transcript(
            name="mixed-code",
            source_path="demo",
            turns=[
                Turn(
                    speaker="AI",
                    content="```python\ndef parse(value):\n    return value\n```\n```json\n{\"ok\": true}\n```",
                    code_blocks=["def parse(value):\n    return value", "{\"ok\": true}"],
                    code_block_languages=["python", "json"],
                    phase="implementation",
                )
            ],
        )

        metrics = MetricsEngine().compute(transcript)

        self.assertEqual(2, metrics["ai_code_block_count"])
        self.assertEqual(1, metrics["python_candidate_block_count"])
        self.assertEqual(1, metrics["non_python_code_block_count"])
        self.assertEqual(1.0, metrics["python_syntax_valid_ratio"])

    def test_custom_metric_evaluator_can_be_registered(self) -> None:
        class CustomMetric(MetricEvaluator):
            name = "custom"

            def compute(self, context, metrics, engine):
                return {"custom_turn_count": context.total_turns}

        transcript = self._build_transcript()
        engine = MetricsEngine()
        engine.register_evaluator(CustomMetric())

        metrics = engine.compute(transcript)

        self.assertEqual(7, metrics["custom_turn_count"])


if __name__ == "__main__":
    unittest.main()
