import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from workflow_eval.metrics import MetricsEngine
from workflow_eval.models import Transcript, Turn
from workflow_eval.scoring import ScoreCalculator


class ScoreCalculatorTests(unittest.TestCase):
    def test_score_config_file_overrides_weights(self) -> None:
        transcript = Transcript(
            name="demo",
            source_path="demo",
            turns=[
                Turn(speaker="User", content="Plan and implement."),
                Turn(speaker="AI", content="```python\ndef run():\n    return True\n```", code_blocks=["def run():\n    return True"], code_block_languages=["python"]),
            ],
        )
        metrics = MetricsEngine().compute(transcript)

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "weights.json"
            weights = {name: 0.0 for name in ScoreCalculator.DEFAULT_WEIGHTS}
            weights["security_hygiene"] = 1.0
            config_path.write_text(
                json.dumps({"weights": weights}),
                encoding="utf-8",
            )
            calculator = ScoreCalculator.from_config_file(config_path)

        score_map, overall_score, _overall_confidence = calculator.score(metrics)

        self.assertAlmostEqual(score_map["security_hygiene"].score, overall_score, places=2)


if __name__ == "__main__":
    unittest.main()
    def test_score_constants_are_used(self) -> None:
        from workflow_eval.scoring import C
        self.assertEqual(C.TURN_RATIO_IDEAL, 1.0)
        self.assertEqual(C.CODE_RATIO_LOW, 0.30)
