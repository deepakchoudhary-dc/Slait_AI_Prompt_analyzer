import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from workflow_eval.parsing import TranscriptParser


class TranscriptParserTests(unittest.TestCase):
    def test_parse_plain_text_transcript_with_code_blocks(self) -> None:
        transcript_text = """User: Write a Python function to reverse a string.
AI: Sure, here is code:
```python
def reverse_text(value: str) -> str:
    return value[::-1]
```
User: Great, add validation.
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session.txt"
            path.write_text(transcript_text, encoding="utf-8")

            parser = TranscriptParser()
            transcript = parser.parse_file(path)

        self.assertEqual(3, len(transcript.turns))
        self.assertEqual("User", transcript.turns[0].speaker)
        self.assertEqual("AI", transcript.turns[1].speaker)
        self.assertEqual(1, len(transcript.turns[1].code_blocks))
        self.assertEqual(["python"], transcript.turns[1].code_block_languages)

    def test_parse_json_transcript_messages(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "Plan architecture."},
                {"role": "assistant", "content": "Use parser, metrics, reporting."},
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            parser = TranscriptParser()
            transcript = parser.parse_file(path)

        self.assertEqual(2, len(transcript.turns))
        self.assertEqual("User", transcript.turns[0].speaker)
        self.assertEqual("AI", transcript.turns[1].speaker)

    def test_strip_sensitive_redacts_common_patterns(self) -> None:
        transcript_text = """User: Email me at jane@example.com
AI: Token is ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sensitive.txt"
            path.write_text(transcript_text, encoding="utf-8")

            parser = TranscriptParser(strip_sensitive=True)
            transcript = parser.parse_file(path)

        joined_content = "\n".join(turn.content for turn in transcript.turns)
        self.assertNotIn("jane@example.com", joined_content)
        self.assertNotIn("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456", joined_content)
        self.assertIn("[REDACTED]", joined_content)

    def test_parse_jsonl_transcript_messages(self) -> None:
        jsonl_payload = "\n".join(
            [
                json.dumps({"role": "user", "content": "Plan a parser."}),
                json.dumps({"role": "assistant", "content": "Use role normalization."}),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session.jsonl"
            path.write_text(jsonl_payload, encoding="utf-8")

            parser = TranscriptParser()
            transcript = parser.parse_file(path)

        self.assertEqual(2, len(transcript.turns))
        self.assertEqual("User", transcript.turns[0].speaker)
        self.assertEqual("AI", transcript.turns[1].speaker)

    def test_parse_json_content_chunks(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"text": "First line"},
                        {"content": [{"text": "Nested line"}]},
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chunks.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            parser = TranscriptParser()
            transcript = parser.parse_file(path)

        self.assertEqual("First line\nNested line", transcript.turns[0].content)

    def test_invalid_jsonl_reports_line_number(self) -> None:
        payload = "\n".join(
            [
                json.dumps({"role": "user", "content": "hello"}),
                "{bad json",
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.jsonl"
            path.write_text(payload, encoding="utf-8")

            parser = TranscriptParser()
            with self.assertRaisesRegex(ValueError, "line 2"):
                parser.parse_file(path)


if __name__ == "__main__":
    unittest.main()
