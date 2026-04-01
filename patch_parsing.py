import re

with open("E:/Slait/src/workflow_eval/parsing.py", "r", encoding="utf-8") as f:
    text = f.read()

# Convert _parse_text_file to streaming yield
text_func_start = text.find('def _parse_text_file(self, path: Path) -> List[Turn]:')
text_func_end = text.find('def _extract_messages', text_func_start)
new_text_func = """def _parse_text_file(self, path: Path) -> Iterable[Turn]:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            yield from self._parse_text_lines(handle)

    def _parse_text_lines(self, lines: Iterable[str]) -> Iterable[Turn]:
        current_speaker: Optional[str] = None
        current_timestamp: Optional[str] = None
        buffer: List[str] = []
        fallback_lines: List[str] = []
        
        has_turns = False
        for raw_line in lines:
            line = raw_line.rstrip("\\n").rstrip("\\r")
            fallback_lines.append(line)
            match = SPEAKER_LINE_PATTERN.match(line)
            if match:
                if current_speaker is not None:
                    has_turns = True
                    yield Turn(
                        speaker=current_speaker,
                        content="\\n".join(buffer).strip(),
                        timestamp=current_timestamp,
                    )
                current_speaker = match.group("speaker")
                current_timestamp = match.group("timestamp")
                buffer = [match.group("content")]
                continue
            if current_speaker is not None:
                buffer.append(line)
                
        if current_speaker is not None:
            has_turns = True
            yield Turn(
                speaker=current_speaker,
                content="\\n".join(buffer).strip(),
                timestamp=current_timestamp,
            )
            
        if not has_turns:
            raw_content = "\\n".join(fallback_lines).strip()
            if raw_content:
                yield Turn(speaker="Unknown", content=raw_content)

    """
text = text[:text_func_start] + new_text_func + text[text_func_end:]

# Fix SPEAKER_LINE_PATTERN to accept any generic name
text = re.sub(
    r'SPEAKER_LINE_PATTERN = re\.compile\([\s\S]*?re\.IGNORECASE,\n\)',
    'SPEAKER_LINE_PATTERN = re.compile(\n    r"^\\\\s*(?:\\\\[(?P<timestamp>[^\\\\]]+)\\\\]\\\\s*)?(?P<speaker>[a-zA-Z0-9_\\\\-\\\\s]{2,30}?)\\\\s*[:\\\\-]\\\\s*(?P<content>.*)$",\n    re.IGNORECASE,\n)',
    text,
    flags=re.MULTILINE
)

with open("E:/Slait/src/workflow_eval/parsing.py", "w", encoding="utf-8") as f:
    f.write(text)
print("done")
