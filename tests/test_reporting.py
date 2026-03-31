import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from workflow_eval.models import MetricScore, TranscriptEvaluation
from workflow_eval.reporting import ReportGenerator


class ReportGeneratorTests(unittest.TestCase):
    def test_csv_includes_dynamic_metric_and_score_columns(self) -> None:
        evaluation = TranscriptEvaluation(
            transcript="session.txt",
            metrics={"total_turns": 3, "custom_metric": {"value": 42}},
            scores={"custom_score": MetricScore(value=42, score=88.0, confidence=70.0, rationale="demo")},
            overall_score=88.0,
            overall_confidence=70.0,
        )

        report = ReportGenerator().build_report([evaluation])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.csv"
            ReportGenerator().write(report, output_path, "csv")

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertEqual(1, len(rows))
        self.assertIn("metric_custom_metric", rows[0])
        self.assertIn("score_custom_score", rows[0])


if __name__ == "__main__":
    unittest.main()
