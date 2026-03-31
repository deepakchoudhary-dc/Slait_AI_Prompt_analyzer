from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Set

LOGGER = logging.getLogger(__name__)


class DataIngestion:
    """Collect and validate transcript input files."""

    SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".ndjson", ".txt", ".log", ".md"}

    def collect_input_files(self, raw_inputs: Sequence[str]) -> List[Path]:
        files: List[Path] = []
        for raw_input in raw_inputs:
            files.extend(self._expand_input(raw_input))

        unique_files: List[Path] = []
        seen: Set[Path] = set()
        for path in sorted(files, key=lambda item: str(item).lower()):
            normalized = path.resolve()
            if normalized not in seen:
                seen.add(normalized)
                unique_files.append(normalized)

        if not unique_files:
            raise ValueError(
                "No transcript files found. Supported extensions: "
                + ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            )

        return unique_files

    def _expand_input(self, raw_input: str) -> Iterable[Path]:
        candidate = Path(raw_input)

        if self._looks_like_glob(raw_input):
            return self._from_glob(raw_input)

        if candidate.exists():
            if candidate.is_file():
                if self._is_supported(candidate):
                    return [candidate]
                LOGGER.warning("Skipping unsupported file type: %s", candidate)
                return []

            if candidate.is_dir():
                return [
                    path
                    for path in candidate.rglob("*")
                    if path.is_file() and self._is_supported(path)
                ]

        # Fallback: treat unresolved path as glob pattern for convenience.
        return self._from_glob(raw_input)

    def _from_glob(self, pattern: str) -> List[Path]:
        matched: List[Path] = []
        for hit in glob.glob(pattern, recursive=True):
            path = Path(hit)
            if path.is_file() and self._is_supported(path):
                matched.append(path)
        return matched

    @staticmethod
    def _looks_like_glob(value: str) -> bool:
        return any(marker in value for marker in ("*", "?", "[", "]"))

    def _is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
