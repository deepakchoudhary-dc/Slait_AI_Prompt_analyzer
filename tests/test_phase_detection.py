import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from workflow_eval.models import Turn
from workflow_eval.phase_detection import PhaseDetector


class PhaseDetectionTests(unittest.TestCase):
    def test_substring_matches_do_not_trigger_phase_hits(self) -> None:
        detector = PhaseDetector(enabled=True)
        turn = Turn(speaker="User", content="The planet has a barcode on it.")

        self.assertEqual("discussion", detector.detect_turn_phase(turn))

    def test_explicit_planning_language_is_detected(self) -> None:
        detector = PhaseDetector(enabled=True)
        turn = Turn(speaker="User", content="Let's plan the architecture and design first.")

        self.assertEqual("planning", detector.detect_turn_phase(turn))

    def test_ai_code_turn_defaults_to_implementation(self) -> None:
        detector = PhaseDetector(enabled=True)
        turn = Turn(
            speaker="AI",
            content="def run():\n    return True",
            code_blocks=["def run():\n    return True"],
        )

        self.assertEqual("implementation", detector.detect_turn_phase(turn))


if __name__ == "__main__":
    unittest.main()
