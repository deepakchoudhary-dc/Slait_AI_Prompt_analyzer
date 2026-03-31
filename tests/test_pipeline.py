import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from workflow_eval.evaluator import WorkflowEvaluator
from workflow_eval.ingestion import DataIngestion
from workflow_eval.metrics import MetricsEngine
from workflow_eval.parsing import TranscriptParser
from workflow_eval.phase_detection import PhaseDetector
from workflow_eval.reporting import ReportGenerator
from workflow_eval.scoring import ScoreCalculator


class EndToEndPipelineTests(unittest.TestCase):
    def test_end_to_end_evaluation_report(self) -> None:
        ingestion = DataIngestion()
        sample_dir = ROOT / "transcripts"
        paths = ingestion.collect_input_files([str(sample_dir)])

        evaluator = WorkflowEvaluator(
            parser=TranscriptParser(strip_sensitive=True),
            phase_detector=PhaseDetector(enabled=True),
            metrics_engine=MetricsEngine(),
            score_calculator=ScoreCalculator(),
        )

        evaluations = evaluator.evaluate_files(paths)
        self.assertGreaterEqual(len(evaluations), 3)

        report_generator = ReportGenerator(include_turns=False, include_timeline=True)
        report = report_generator.build_report(evaluations)

        summary = report.get("summary", {})
        self.assertEqual(len(paths), summary.get("transcript_count"))
        self.assertGreaterEqual(summary.get("successful_evaluations", 0), 1)
        self.assertIn("total_security_risks", summary)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.json"
            report_generator.write(report, output_path, "json")

            self.assertTrue(output_path.exists())
            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("results", data)
            first_result = data["results"][0]
            self.assertIn("recommendations", first_result)


if __name__ == "__main__":
    unittest.main()
