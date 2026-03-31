from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import Transcript, Turn

SPEAKER_LINE_PATTERN = re.compile(
    r"^\s*(?:\[(?P<timestamp>[^\]]+)\]\s*)?(?P<speaker>User|Human|Developer|Assistant|AI|Copilot|Claude|ChatGPT|System|Tool)\s*[:\-]\s*(?P<content>.*)$",
    re.IGNORECASE,
)
CODE_BLOCK_PATTERN = re.compile(
    r"(?:```|~~~)(?P<language>[\w.+\-]+)?\n(?P<code>.*?)(?:```|~~~)",
    re.DOTALL,
)
SMART_QUOTES_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
)

SENSITIVE_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b"),
    re.compile(
        r"-----BEGIN [A-Z ]+PRIVATE KEY-----.*?-----END [A-Z ]+PRIVATE KEY-----",
        re.DOTALL,
    ),
]


class TranscriptParser:
    """Parse transcript files from different export formats into a normalized schema."""

    def __init__(
        self,
        remove_system_messages: bool = True,
        strip_sensitive: bool = False,
        redaction_token: str = "[REDACTED]",
    ) -> None:
        self.remove_system_messages = remove_system_messages
        self.strip_sensitive = strip_sensitive
        self.redaction_token = redaction_token

    def parse_file(self, path: Path) -> Transcript:
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                raw_turns = self._parse_json_file(path)
            except ValueError:
                raw_turns = self._parse_json_lines_file(path)
        elif suffix in {".jsonl", ".ndjson"}:
            raw_turns = self._parse_json_lines_file(path)
        else:
            raw_turns = self._parse_text_file(path)

        normalized_turns: List[Turn] = []
        for turn in raw_turns:
            speaker = self._normalize_speaker(turn.speaker)
            content = self._normalize_content(turn.content)

            if self.strip_sensitive:
                content = self._redact_sensitive(content)

            extracted_blocks = self._extract_code_blocks(content)
            code_blocks = [block for block, _language in extracted_blocks]
            code_languages = [language for _block, language in extracted_blocks]
            if not code_blocks and self._looks_like_unfenced_code(content):
                code_blocks = [content]
                code_languages = [None]

            if self.remove_system_messages and speaker == "System":
                continue

            if not content and not code_blocks:
                continue

            metadata = dict(turn.metadata)
            if code_languages:
                metadata["code_block_languages"] = code_languages

            normalized_turns.append(
                Turn(
                    speaker=speaker,
                    content=content,
                    timestamp=turn.timestamp,
                    code_blocks=code_blocks,
                    code_block_languages=code_languages,
                    metadata=metadata,
                )
            )

        if not normalized_turns:
            raise ValueError(f"Transcript '{path.name}' did not contain parseable turns.")

        return Transcript(
            name=path.name,
            source_path=str(path),
            turns=normalized_turns,
            metadata={"turn_count": len(normalized_turns)},
        )

    def _parse_json_file(self, path: Path) -> List[Turn]:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON transcript: {exc}") from exc

        messages = list(self._extract_messages(payload))
        if not messages:
            raise ValueError("JSON transcript did not contain parseable messages.")

        return [self._to_turn(message, index) for index, message in enumerate(messages)]

    def _parse_json_lines_file(self, path: Path) -> Iterable[Turn]:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL transcript at line {index}: {exc}") from exc
                
                messages = list(self._extract_messages(payload))
                for msg in messages:
                    yield self._to_turn(msg, index)

    def _parse_text_file(self, path: Path) -> List[Turn]:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return self._parse_text_lines(handle)

    def _parse_text_lines(self, lines: Iterable[str]) -> List[Turn]:
        turns: List[Turn] = []
        current_speaker: Optional[str] = None
        current_timestamp: Optional[str] = None
        buffer: List[str] = []
        fallback_lines: List[str] = []

        for raw_line in lines:
            line = raw_line.rstrip("\n").rstrip("\r")
            fallback_lines.append(line)
            match = SPEAKER_LINE_PATTERN.match(line)
            if match:
                if current_speaker is not None:
                    turns.append(
                        Turn(
                            speaker=current_speaker,
                            content="\n".join(buffer).strip(),
                            timestamp=current_timestamp,
                        )
                    )

                current_speaker = match.group("speaker")
                current_timestamp = match.group("timestamp")
                buffer = [match.group("content")]
                continue

            if current_speaker is None:
                continue

            buffer.append(line)

        if current_speaker is not None:
            turns.append(
                Turn(
                    speaker=current_speaker,
                    content="\n".join(buffer).strip(),
                    timestamp=current_timestamp,
                )
            )

        if not turns:
            raw_content = "\n".join(fallback_lines).strip()
            if raw_content:
                return [Turn(speaker="Unknown", content=raw_content)]

        return turns

    def _extract_messages(self, payload: Any) -> Iterable[Any]:
        if isinstance(payload, list):
            if payload and self._is_message_candidate(payload[0]):
                return payload
            extracted: List[Any] = []
            for item in payload:
                extracted.extend(list(self._extract_messages(item)))
            return extracted

        if isinstance(payload, dict):
            preferred_keys = (
                "messages",
                "turns",
                "conversation",
                "chat",
                "items",
                "entries",
                "data",
            )
            for key in preferred_keys:
                if key in payload:
                    nested = list(self._extract_messages(payload[key]))
                    if nested:
                        return nested

            if self._is_message_candidate(payload):
                return [payload]

            collected: List[Any] = []
            for value in payload.values():
                nested = list(self._extract_messages(value))
                if nested:
                    collected.extend(nested)
            return collected

        return []

    @staticmethod
    def _is_message_candidate(item: Any) -> bool:
        if isinstance(item, str):
            return True
        if not isinstance(item, dict):
            return False

        speaker_keys = {"speaker", "role", "author", "participant", "from"}
        content_keys = {"content", "message", "text", "value", "body"}
        return bool(speaker_keys.intersection(item.keys())) and bool(
            content_keys.intersection(item.keys())
        )

    def _to_turn(self, message: Any, index: int) -> Turn:
        if isinstance(message, str):
            return Turn(speaker="Unknown", content=message)

        if not isinstance(message, dict):
            return Turn(speaker="Unknown", content=str(message))

        speaker = str(
            self._first_present(
                message,
                ["speaker", "role", "author", "participant", "from"],
                "Unknown",
            )
        )
        timestamp_value = self._first_present(
            message,
            ["timestamp", "time", "created_at", "datetime"],
            None,
        )
        content = self._extract_message_content(message)

        metadata: Dict[str, Any] = {"index": index}
        for key in ("id", "channel", "model", "tool"):
            if key in message:
                metadata[key] = message[key]

        return Turn(
            speaker=speaker,
            content=content,
            timestamp=str(timestamp_value) if timestamp_value is not None else None,
            metadata=metadata,
        )

    @staticmethod
    def _first_present(payload: Dict[str, Any], keys: Iterable[str], default: Any) -> Any:
        for key in keys:
            if key in payload:
                return payload[key]
        return default

    def _extract_message_content(self, message: Dict[str, Any]) -> str:
        for key in ("content", "message", "text", "value", "body"):
            if key not in message:
                continue

            value = message[key]
            if isinstance(value, str):
                return value

            if isinstance(value, list):
                chunks = [self._normalize_json_content_chunk(chunk) for chunk in value]
                return "\n".join(chunk for chunk in chunks if chunk).strip()

            if isinstance(value, dict):
                return self._normalize_json_content_chunk(value)

        return json.dumps(message, ensure_ascii=True)

    def _normalize_json_content_chunk(self, chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk

        if isinstance(chunk, dict):
            for key in ("text", "content", "value", "body"):
                if key in chunk and isinstance(chunk[key], str):
                    return chunk[key]

            if "content" in chunk and isinstance(chunk["content"], list):
                nested = [self._normalize_json_content_chunk(item) for item in chunk["content"]]
                return "\n".join(item for item in nested if item).strip()

        return ""

    @staticmethod
    def _normalize_speaker(raw_speaker: str) -> str:
        speaker = raw_speaker.strip().lower()
        user_aliases = {"user", "human", "developer"}
        ai_aliases = {"assistant", "ai", "copilot", "claude", "chatgpt", "model"}

        if speaker in user_aliases:
            return "User"
        if speaker in ai_aliases:
            return "AI"
        if speaker in {"system"}:
            return "System"
        if speaker in {"tool", "function"}:
            return "Tool"
        return raw_speaker.strip().title() or "Unknown"

    def _normalize_content(self, content: str) -> str:
        normalized = content.translate(SMART_QUOTES_TRANSLATION)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in normalized.split("\n")]
        return "\n".join(lines).strip()

    def _redact_sensitive(self, content: str) -> str:
        redacted = content
        for pattern in SENSITIVE_PATTERNS:
            redacted = pattern.sub(self.redaction_token, redacted)
        return redacted

    @staticmethod
    def _extract_code_blocks(content: str) -> List[Tuple[str, Optional[str]]]:
        extracted: List[Tuple[str, Optional[str]]] = []
        for match in CODE_BLOCK_PATTERN.finditer(content):
            code = match.group("code").strip()
            language = match.group("language")
            if not code:
                continue
            extracted.append((code, language.lower() if language else None))
        return extracted

    @staticmethod
    def _looks_like_unfenced_code(content: str) -> bool:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        code_markers = (
            "def ",
            "class ",
            "import ",
            "from ",
            "if ",
            "for ",
            "while ",
            "return ",
            "try:",
            "except ",
        )
        marker_hits = sum(1 for line in lines if line.startswith(code_markers))
        punctuation_hits = sum(
            1
            for line in lines
            if line.endswith(":") or ("(" in line and ")" in line)
        )
        return marker_hits >= 1 and punctuation_hits >= 1
