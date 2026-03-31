from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from math import ceil
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Transcript, Turn

WORD_PATTERN = re.compile(r"\b[\w'-]+\b")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}

SECRET_NAME_PATTERN = re.compile(
    r"(password|token|secret|api[_-]?key|client[_-]?secret)",
    re.IGNORECASE,
)
LINE_SECURITY_PATTERNS = {
    "dynamic_code_execution": re.compile(r"\b(eval|exec)\s*\("),
    "unsafe_deserialization": re.compile(r"\b(pickle\.loads|yaml\.load)\s*\("),
    "dangerous_subprocess": re.compile(
        r"\bsubprocess\.(run|Popen|call)\s*\([^)\n]{0,400}\bshell\s*=\s*True\b"
    ),
    "weak_hashing": re.compile(r"\b(hashlib\.(md5|sha1)|md5\s*\(|sha1\s*\()"),
    "plaintext_secret_assignment": re.compile(
        r"\b(password|token|secret|api[_-]?key)\b\s*[:=]\s*['\"][^'\"]+['\"]",
        re.IGNORECASE,
    ),
    "tls_verification_disabled": re.compile(r"\bverify\s*=\s*False\b"),
}

SECRET_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b"),
]

SECURITY_DISCUSSION_TERMS = (
    "security",
    "sanitize",
    "escape",
    "validate",
    "csrf",
    "xss",
    "sql injection",
    "auth",
    "authorization",
    "least privilege",
    "secrets",
)

SPECIFICITY_TERMS = (
    "requirements",
    "constraint",
    "validate",
    "test",
    "edge case",
    "error handling",
    "performance",
    "security",
    "modular",
    "scalable",
    "json",
    "cli",
    "frontend",
    "report",
)

PHASES = ["planning", "implementation", "debugging", "testing", "review", "discussion"]
SPEAKERS = ["User", "AI", "Tool", "System", "Unknown"]

COMPLEXITY_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.ExceptHandler,
    ast.BoolOp,
    ast.IfExp,
    ast.With,
    ast.AsyncWith,
    ast.comprehension,
    ast.Match,
)

PYTHON_LANGUAGE_HINTS = {"python", "py", "python3", "py3"}
NON_PYTHON_LANGUAGE_HINTS = {
    "bash",
    "sh",
    "shell",
    "powershell",
    "ps1",
    "javascript",
    "js",
    "typescript",
    "ts",
    "tsx",
    "jsx",
    "json",
    "yaml",
    "yml",
    "toml",
    "xml",
    "html",
    "css",
    "sql",
    "go",
    "rust",
    "java",
    "kotlin",
    "swift",
    "c",
    "cpp",
    "csharp",
    "cs",
    "ruby",
    "php",
}


@dataclass(frozen=True)
class MetricContext:
    transcript: Transcript
    turns: List[Turn]
    user_turns: List[Turn]
    ai_turns: List[Turn]
    tool_turns: List[Turn]
    total_turns: int


@dataclass(frozen=True)
class CodeBlockRecord:
    block: str
    language: Optional[str]
    speaker: str


class MetricEvaluator(ABC):
    """Unit of metric computation that can be registered into the engine."""

    name: str

    @abstractmethod
    def compute(
        self,
        context: MetricContext,
        metrics: Dict[str, Any],
        engine: "MetricsEngine",
    ) -> Dict[str, Any]:
        raise NotImplementedError


class ConversationMetricsEvaluator(MetricEvaluator):
    name = "conversation"

    def compute(
        self,
        context: MetricContext,
        metrics: Dict[str, Any],
        engine: "MetricsEngine",
    ) -> Dict[str, Any]:
        user_turn_count = len(context.user_turns)
        ai_turn_count = len(context.ai_turns)
        tool_turn_count = len(context.tool_turns)
        user_word_count = sum(engine._word_count(turn.content) for turn in context.user_turns)
        ai_word_count = sum(engine._word_count(turn.content) for turn in context.ai_turns)
        code_turn_count = sum(1 for turn in context.turns if engine._is_code_turn(turn))
        discussion_turn_count = max(context.total_turns - code_turn_count, 0)
        ai_to_user_turn_ratio = (
            round(ai_turn_count / user_turn_count, 3) if user_turn_count else None
        )
        code_to_discussion_ratio = (
            round(code_turn_count / discussion_turn_count, 3)
            if discussion_turn_count
            else None
        )
        tool_call_turn_count = sum(
            1
            for turn in context.turns
            if turn.speaker == "Tool" or bool(engine.TOOL_CALL_PATTERN.search(turn.content))
        )
        return {
            "total_turns": context.total_turns,
            "user_turns": user_turn_count,
            "ai_turns": ai_turn_count,
            "tool_turns": tool_turn_count,
            "ai_to_user_turn_ratio": ai_to_user_turn_ratio,
            "user_word_count": user_word_count,
            "ai_word_count": ai_word_count,
            "estimated_user_tokens": engine._estimate_tokens(user_word_count),
            "estimated_ai_tokens": engine._estimate_tokens(ai_word_count),
            "estimated_total_tokens": engine._estimate_tokens(user_word_count)
            + engine._estimate_tokens(ai_word_count),
            "avg_user_prompt_words": round(user_word_count / user_turn_count, 2)
            if user_turn_count
            else 0.0,
            "avg_ai_response_words": round(ai_word_count / ai_turn_count, 2)
            if ai_turn_count
            else 0.0,
            "code_turn_count": code_turn_count,
            "discussion_turn_count": discussion_turn_count,
            "code_turn_ratio": round(code_turn_count / context.total_turns, 3)
            if context.total_turns
            else 0.0,
            "code_to_discussion_ratio": code_to_discussion_ratio,
            "tool_call_turn_count": tool_call_turn_count,
        }


class SuggestionMetricsEvaluator(MetricEvaluator):
    name = "suggestions"

    def compute(
        self,
        context: MetricContext,
        metrics: Dict[str, Any],
        engine: "MetricsEngine",
    ) -> Dict[str, Any]:
        correction_count = sum(
            1
            for turn in context.user_turns
            if engine._contains_any(turn.content, engine.CORRECTION_TERMS)
        )
        positive_feedback_count = sum(
            1
            for turn in context.user_turns
            if engine._contains_any(turn.content, engine.POSITIVE_TERMS)
        )
        ai_suggestion_count, accepted_count, rejected_count, unresolved_count = engine._suggestion_metrics(
            context.turns
        )
        user_turn_count = len(context.user_turns)
        acceptance_rate = (
            round(accepted_count / ai_suggestion_count, 3) if ai_suggestion_count else None
        )
        rejection_rate = (
            round(rejected_count / ai_suggestion_count, 3) if ai_suggestion_count else None
        )
        return {
            "avg_response_latency_turns": engine._average_response_latency_turns(context.turns),
            "correction_count": correction_count,
            "correction_ratio": round(correction_count / user_turn_count, 3)
            if user_turn_count
            else 0.0,
            "positive_feedback_count": positive_feedback_count,
            "positive_feedback_ratio": round(positive_feedback_count / user_turn_count, 3)
            if user_turn_count
            else 0.0,
            "ai_suggestion_count": ai_suggestion_count,
            "accepted_suggestion_count": accepted_count,
            "rejected_suggestion_count": rejected_count,
            "unresolved_suggestion_count": unresolved_count,
            "acceptance_rate": acceptance_rate,
            "rejection_rate": rejection_rate,
        }


class WorkflowDynamicsEvaluator(MetricEvaluator):
    name = "workflow_dynamics"

    def compute(
        self,
        context: MetricContext,
        metrics: Dict[str, Any],
        engine: "MetricsEngine",
    ) -> Dict[str, Any]:
        phase_sequence = [turn.phase if turn.phase in PHASES else "discussion" for turn in context.turns]
        phase_transition_matrix = engine._build_transition_matrix(phase_sequence, PHASES)
        speaker_sequence = [
            turn.speaker if turn.speaker in SPEAKERS else "Unknown" for turn in context.turns
        ]
        speaker_transition_matrix = engine._build_transition_matrix(speaker_sequence, SPEAKERS)
        context_switch_count = engine._count_context_switches(phase_sequence)
        output = {
            "context_retention_score": engine._context_retention_score(context.turns),
            "prompt_specificity_score": engine._prompt_specificity_score(context.user_turns),
            "context_switch_count": context_switch_count,
            "phase_switch_rate": round(context_switch_count / max(context.total_turns - 1, 1), 3)
            if context.total_turns > 1
            else 0.0,
            "phase_loopback_count": engine._count_phase_loopbacks(phase_sequence),
            "phase_transition_matrix": phase_transition_matrix,
            "phase_transition_edges": engine._matrix_to_edges(phase_transition_matrix),
            "speaker_transition_matrix": speaker_transition_matrix,
            "speaker_transition_edges": engine._matrix_to_edges(speaker_transition_matrix),
            "security_discussion_count": sum(
                1
                for turn in context.turns
                if engine._contains_any(turn.content, SECURITY_DISCUSSION_TERMS)
            ),
            "insecure_dependency_mention_count": sum(
                1
                for turn in context.turns
                if engine._contains_any(turn.content, engine.INSECURE_DEPENDENCY_TERMS)
            ),
        }
        output.update(engine._phase_distribution(context.turns, context.total_turns))
        return output


class CodeQualityMetricsEvaluator(MetricEvaluator):
    name = "code_quality"

    def compute(
        self,
        context: MetricContext,
        metrics: Dict[str, Any],
        engine: "MetricsEngine",
    ) -> Dict[str, Any]:
        return engine._code_quality_metrics(context.ai_turns)


class SecurityMetricsEvaluator(MetricEvaluator):
    name = "security"

    def compute(
        self,
        context: MetricContext,
        metrics: Dict[str, Any],
        engine: "MetricsEngine",
    ) -> Dict[str, Any]:
        return engine._security_metrics(context.turns, context.ai_turns)


class WorkflowFindingsEvaluator(MetricEvaluator):
    name = "workflow_findings"

    def compute(
        self,
        context: MetricContext,
        metrics: Dict[str, Any],
        engine: "MetricsEngine",
    ) -> Dict[str, Any]:
        return {"workflow_findings": engine._workflow_findings(metrics)}


class PythonSecurityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: Counter[str] = Counter()

    def visit_Call(self, node: ast.Call) -> None:
        qualified_name = self._qualified_name(node.func)
        if qualified_name in {"eval", "exec"}:
            self.findings["dynamic_code_execution"] += 1
        if qualified_name in {"pickle.loads", "yaml.load"} or "load" in qualified_name and "yaml" in qualified_name:
            self.findings["unsafe_deserialization"] += 1
        if any(name in qualified_name for name in {"subprocess.run", "subprocess.Popen", "subprocess.call", "os.popen", "os.system"}):
            if self._has_truthy_arg(node, "shell", 8) or self._has_bool_keyword(node, "shell", True):
                self.findings["dangerous_subprocess"] += 1
        if any(name in qualified_name for name in {"hashlib.md5", "hashlib.sha1", "md5", "sha1"}):    
            self.findings["weak_hashing"] += 1
        if self._has_bool_keyword(node, "verify", False) or self._has_falsy_arg(node, "verify", 99):
            self.findings["tls_verification_disabled"] += 1
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if any(self._secret_name_matches(target) for target in node.targets) and self._is_string_literal(node.value):
            self.findings["plaintext_secret_assignment"] += 1
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self._secret_name_matches(node.target) and self._is_string_literal(node.value):
            self.findings["plaintext_secret_assignment"] += 1
        self.generic_visit(node)

    @staticmethod
    def _qualified_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = PythonSecurityVisitor._qualified_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    @staticmethod
    def _has_bool_keyword(node: ast.Call, keyword_name: str, expected: bool) -> bool:
        for keyword in node.keywords:
            if keyword.arg == keyword_name:
                if isinstance(keyword.value, ast.Constant):
                    return bool(keyword.value.value) is expected
        return False

    @staticmethod
    def _has_truthy_arg(node: ast.Call, keyword_name: str, pos_idx: int) -> bool:
        if len(node.args) > pos_idx:
            arg = node.args[pos_idx]
            if isinstance(arg, ast.Constant):
                return bool(arg.value) is True
        return False

    @staticmethod
    def _has_falsy_arg(node: ast.Call, keyword_name: str, pos_idx: int) -> bool:
        if len(node.args) > pos_idx:
            arg = node.args[pos_idx]
            if isinstance(arg, ast.Constant):
                return bool(arg.value) is False
        return False

    @staticmethod
    def _secret_name_matches(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return bool(SECRET_NAME_PATTERN.search(node.id))
        if isinstance(node, ast.Attribute):
            return bool(SECRET_NAME_PATTERN.search(node.attr))
        return False

    @staticmethod
    def _is_string_literal(node: Optional[ast.AST]) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, str) and bool(node.value.strip())

class MetricsEngine:
    """Compute transcript-level workflow metrics using a pluggable evaluator registry."""

    CORRECTION_TERMS = (
        "not quite",
        "try again",
        "incorrect",
        "wrong",
        "fix this",
        "does not work",
        "failed",
        "error",
        "change that",
    )

    REJECTION_TERMS = (
        "no",
        "not this",
        "different approach",
        "start over",
        "discard",
        "replace",
        "that is wrong",
    )

    POSITIVE_TERMS = (
        "good",
        "great",
        "perfect",
        "nice",
        "thanks",
        "thank you",
        "looks good",
        "works",
    )

    CONTINUATION_TERMS = (
        "now",
        "next",
        "also",
        "can you add",
        "please add",
        "extend",
        "continue",
        "looks good",
        "proceed",
    )

    INSECURE_DEPENDENCY_TERMS = (
        "event-stream",
        "ua-parser-js",
        "axios@",
        "left-pad",
    )

    TOOL_CALL_PATTERN = re.compile(
        r"\b(tool|run_in_terminal|function call|execute command|shell command|terminal)\b",
        re.IGNORECASE,
    )

    def __init__(self, evaluators: Optional[Iterable[MetricEvaluator]] = None) -> None:
        self.evaluators: List[MetricEvaluator] = list(evaluators or self.default_evaluators())

    @staticmethod
    def default_evaluators() -> List[MetricEvaluator]:
        return [
            ConversationMetricsEvaluator(),
            SuggestionMetricsEvaluator(),
            WorkflowDynamicsEvaluator(),
            CodeQualityMetricsEvaluator(),
            SecurityMetricsEvaluator(),
            WorkflowFindingsEvaluator(),
        ]

    def register_evaluator(self, evaluator: MetricEvaluator) -> None:
        self.evaluators.append(evaluator)

    def compute(self, transcript: Transcript) -> Dict[str, Any]:
        context = self._build_context(transcript)
        metrics: Dict[str, Any] = {}
        for evaluator in self.evaluators:
            result = evaluator.compute(context, metrics, self)
            overlap = set(metrics).intersection(result)
            if overlap:
                overlap_list = ", ".join(sorted(overlap))
                raise ValueError(
                    f"Metric evaluator '{evaluator.name}' attempted to overwrite existing metrics: {overlap_list}"
                )
            metrics.update(result)
        return metrics

    def _build_context(self, transcript: Transcript) -> MetricContext:
        turns = transcript.turns
        return MetricContext(
            transcript=transcript,
            turns=turns,
            user_turns=[turn for turn in turns if turn.speaker == "User"],
            ai_turns=[turn for turn in turns if turn.speaker == "AI"],
            tool_turns=[turn for turn in turns if turn.speaker == "Tool"],
            total_turns=len(turns),
        )

    @staticmethod
    def _word_count(content: str) -> int:
        return len(WORD_PATTERN.findall(content))

    @staticmethod
    def _estimate_tokens(words: int) -> int:
        # Approximate GPT-style token count from words.
        return int(ceil(words * 1.35))

    @staticmethod
    def _normalize_text(content: str) -> str:
        return " ".join(token.lower() for token in WORD_PATTERN.findall(content))

    @staticmethod
    def _extract_keywords(content: str) -> List[str]:
        tokens = [token.lower() for token in WORD_PATTERN.findall(content)]
        keywords = [token for token in tokens if len(token) > 3 and token not in STOPWORDS]
        return keywords

    def _context_retention_score(self, turns: List[Turn]) -> float:
        overlap_scores: List[float] = []

        for index, turn in enumerate(turns):
            if turn.speaker != "User":
                continue

            next_ai_turn = self._find_next_ai_turn(turns, index + 1)
            if next_ai_turn is None:
                continue

            user_keywords = set(self._extract_keywords(turn.content))
            ai_keywords = set(self._extract_keywords(next_ai_turn.content))
            if not user_keywords:
                continue

            overlap = len(user_keywords.intersection(ai_keywords)) / len(user_keywords)
            overlap_scores.append(overlap)

        if not overlap_scores:
            return 0.0

        return round(mean(overlap_scores) * 100.0, 2)

    def _prompt_specificity_score(self, user_turns: List[Turn]) -> float:
        if not user_turns:
            return 0.0

        scores: List[float] = []
        for turn in user_turns:
            words = self._word_count(turn.content)
            normalized_text = self._normalize_text(turn.content)
            token_set = set(WORD_PATTERN.findall(normalized_text))

            if words < 4:
                length_score = 25.0
            elif words <= 14:
                length_score = 70.0
            elif words <= 60:
                length_score = 100.0
            elif words <= 120:
                length_score = 80.0
            else:
                length_score = 55.0

            keyword_hits = 0
            for term in SPECIFICITY_TERMS:
                normalized_term = self._normalize_text(term)
                if " " in normalized_term:
                    keyword_hits += int(f" {normalized_term} " in f" {normalized_text} ")
                elif normalized_term in token_set:
                    keyword_hits += 1
            keyword_score = min(keyword_hits, 6) / 6 * 100.0

            structure_signals = 0
            if ":" in turn.content:
                structure_signals += 1
            if "\n" in turn.content:
                structure_signals += 1
            if any(marker in turn.content for marker in ("must", "should", "need", "required")):
                structure_signals += 1
            if any(char.isdigit() for char in turn.content):
                structure_signals += 1

            structure_score = min(structure_signals, 4) / 4 * 100.0
            score = (0.4 * length_score) + (0.4 * keyword_score) + (0.2 * structure_score)
            scores.append(score)

        return round(mean(scores), 2)

    def _average_response_latency_turns(self, turns: List[Turn]) -> float:
        latencies: List[int] = []
        for index, turn in enumerate(turns):
            if turn.speaker != "User":
                continue

            for lookahead, candidate in enumerate(turns[index + 1 :], start=1):
                if candidate.speaker == "AI":
                    latencies.append(lookahead)
                    break

        if not latencies:
            return 0.0

        return round(mean(latencies), 2)

    def _suggestion_metrics(self, turns: List[Turn]) -> Tuple[int, int, int, int]:
        ai_suggestion_count = 0
        accepted_count = 0
        rejected_count = 0
        unresolved_count = 0

        for index, turn in enumerate(turns):
            if turn.speaker != "AI":
                continue

            ai_suggestion_count += 1
            next_user_turn = self._find_next_user_turn(turns, index + 1)
            if next_user_turn is None:
                unresolved_count += 1
                continue

            next_user_text = next_user_turn.content.lower()
            if self._contains_any(next_user_text, self.REJECTION_TERMS) or self._contains_any(
                next_user_text, self.CORRECTION_TERMS
            ):
                rejected_count += 1
                continue

            if self._contains_any(next_user_text, self.POSITIVE_TERMS) or self._contains_any(
                next_user_text, self.CONTINUATION_TERMS
            ):
                accepted_count += 1
                continue

            unresolved_count += 1

        return ai_suggestion_count, accepted_count, rejected_count, unresolved_count

    @staticmethod
    def _find_next_user_turn(turns: List[Turn], start_index: int) -> Optional[Turn]:
        for turn in turns[start_index:]:
            if turn.speaker == "User":
                return turn
        return None

    @staticmethod
    def _find_next_ai_turn(turns: List[Turn], start_index: int) -> Optional[Turn]:
        for turn in turns[start_index:]:
            if turn.speaker == "AI":
                return turn
        return None

    @classmethod
    def _contains_any(cls, content: str, keywords: Sequence[str]) -> bool:
        normalized_text = cls._normalize_text(content)
        tokens = set(WORD_PATTERN.findall(normalized_text))
        for keyword in keywords:
            normalized_keyword = cls._normalize_text(keyword)
            if not normalized_keyword:
                continue
            if " " in normalized_keyword:
                if f" {normalized_keyword} " in f" {normalized_text} ":
                    return True
            elif normalized_keyword in tokens:
                return True
        return False

    @staticmethod
    def _is_code_turn(turn: Turn) -> bool:
        if turn.code_blocks:
            return True

        lines = [line.strip() for line in turn.content.splitlines() if line.strip()]
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
        return marker_hits >= 1

    def _iter_code_blocks(self, turns: Iterable[Turn]) -> List[CodeBlockRecord]:
        records: List[CodeBlockRecord] = []
        for turn in turns:
            for index, block in enumerate(turn.code_blocks):
                language = turn.code_block_languages[index] if index < len(turn.code_block_languages) else None
                records.append(CodeBlockRecord(block=block, language=language, speaker=turn.speaker))
        return records

    @staticmethod
    def _count_non_empty_lines(content: str) -> int:
        return sum(1 for line in content.splitlines() if line.strip())

    def _is_python_candidate(self, record: CodeBlockRecord) -> bool:
        language = (record.language or "").lower()
        if language in PYTHON_LANGUAGE_HINTS:
            return True
        if language in NON_PYTHON_LANGUAGE_HINTS:
            return False

        code = record.block.strip()
        if not code:
            return False

        if re.search(r"^\s*<\w+", code, re.MULTILINE):
            return False
        if re.search(r"^\s*\{", code) and ":" in code and '"' in code:
            return False

        lines = [line.strip() for line in code.splitlines() if line.strip()]
        python_prefixes = (
            "def ",
            "class ",
            "import ",
            "from ",
            "if ",
            "elif ",
            "for ",
            "while ",
            "try:",
            "except ",
            "with ",
            "async ",
            "@",
        )
        non_python_prefixes = ("const ", "let ", "function ", "public ", "private ", "SELECT ", "INSERT ")

        python_signals = sum(1 for line in lines if line.startswith(python_prefixes))
        non_python_signals = sum(1 for line in lines if line.startswith(non_python_prefixes))
        python_signals += int("self" in code or "None" in code or "True" in code or "False" in code)
        non_python_signals += int("{" in code and "}" in code)
        non_python_signals += int("=>" in code or "console.log" in code or "println!(" in code)

        return python_signals > 0 and python_signals >= non_python_signals

    @staticmethod
    def _line_security_findings(content: str) -> Counter[str]:
        findings: Counter[str] = Counter()
        for line in content.splitlines():
            for risk_name, pattern in LINE_SECURITY_PATTERNS.items():
                findings[risk_name] += len(pattern.findall(line))
        return findings

    def _code_quality_metrics(self, ai_turns: List[Turn]) -> Dict[str, Any]:
        records = self._iter_code_blocks(ai_turns)
        block_count = len(records)
        code_line_count = sum(self._count_non_empty_lines(record.block) for record in records)
        python_records = [record for record in records if self._is_python_candidate(record)]
        python_candidate_count = len(python_records)
        language_breakdown: Counter[str] = Counter(
            record.language or "unknown" for record in records if record.language
        )

        parseable_block_count = 0
        complexities: List[int] = []
        function_lengths: List[int] = []
        function_count = 0
        fully_typed_function_count = 0
        docstring_function_count = 0
        try_count = 0
        long_line_count = 0
        comment_line_count = 0

        for record in records:
            lines = record.block.splitlines()
            long_line_count += sum(1 for line in lines if len(line.rstrip()) > 120)
            comment_line_count += sum(1 for line in lines if line.strip().startswith("#"))

        for record in python_records:
            try:
                tree = ast.parse(record.block)
            except SyntaxError:
                continue

            parseable_block_count += 1
            complexities.append(self._estimate_cyclomatic_complexity(tree))
            try_count += sum(1 for node in ast.walk(tree) if isinstance(node, ast.Try))

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_count += 1

                    if self._is_fully_typed_function(node):
                        fully_typed_function_count += 1

                    if ast.get_docstring(node):
                        docstring_function_count += 1

                    if hasattr(node, "lineno") and hasattr(node, "end_lineno") and node.end_lineno:
                        function_lengths.append(max(node.end_lineno - node.lineno + 1, 1))

        syntax_valid_ratio: Optional[float] = None
        if python_candidate_count > 0:
            syntax_valid_ratio = round(parseable_block_count / python_candidate_count, 3)

        avg_complexity = round(mean(complexities), 2) if complexities else 0.0
        max_complexity = max(complexities) if complexities else 0
        avg_function_length = round(mean(function_lengths), 2) if function_lengths else 0.0

        type_hint_coverage: Optional[float] = None
        docstring_coverage: Optional[float] = None
        if function_count > 0:
            type_hint_coverage = round(fully_typed_function_count / function_count, 3)
            docstring_coverage = round(docstring_function_count / function_count, 3)

        exception_handling_density = (
            round(try_count / parseable_block_count, 3) if parseable_block_count > 0 else 0.0
        )
        comment_density = round(comment_line_count / max(code_line_count, 1), 3)

        return {
            "ai_code_block_count": block_count,
            "ai_code_line_count": code_line_count,
            "python_candidate_block_count": python_candidate_count,
            "non_python_code_block_count": max(block_count - python_candidate_count, 0),
            "parseable_python_block_count": parseable_block_count,
            "python_syntax_valid_ratio": syntax_valid_ratio,
            "avg_estimated_cyclomatic_complexity": avg_complexity,
            "max_estimated_cyclomatic_complexity": max_complexity,
            "function_count": function_count,
            "avg_function_length": avg_function_length,
            "type_hint_coverage": type_hint_coverage,
            "docstring_coverage": docstring_coverage,
            "exception_handling_density": exception_handling_density,
            "long_line_count": long_line_count,
            "comment_density": comment_density,
            "declared_code_language_breakdown": dict(sorted(language_breakdown.items())),
        }

    @staticmethod
    def _is_fully_typed_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        has_all_arg_hints = all(arg.annotation is not None for arg in args)
        return has_all_arg_hints and node.returns is not None

    @staticmethod
    def _estimate_cyclomatic_complexity(tree: ast.AST) -> int:
        complexity = 1
        complexity += sum(1 for node in ast.walk(tree) if isinstance(node, COMPLEXITY_NODES))
        return complexity

    def _security_metrics(self, turns: List[Turn], ai_turns: List[Turn]) -> Dict[str, Any]:
        content_corpus = "\n".join(turn.content for turn in turns)
        code_records = self._iter_code_blocks(ai_turns)
        risk_breakdown: Counter[str] = Counter()
        for record in code_records:
            if self._is_python_candidate(record):
                try:
                    tree = ast.parse(record.block)
                except SyntaxError:
                    risk_breakdown.update(self._line_security_findings(record.block))
                    continue

                visitor = PythonSecurityVisitor()
                visitor.visit(tree)
                risk_breakdown.update(visitor.findings)
            else:
                risk_breakdown.update(self._line_security_findings(record.block))

        insecure_code_pattern_count = sum(risk_breakdown.values())
        secret_exposure_count = sum(len(pattern.findall(content_corpus)) for pattern in SECRET_PATTERNS)
        security_risk_count = insecure_code_pattern_count + secret_exposure_count

        code_block_count = max(len(code_records), 1)
        security_issue_density = round(security_risk_count / code_block_count, 3)

        findings: List[Dict[str, Any]] = []
        for risk_name, count in sorted(risk_breakdown.items()):
            if count > 0:
                findings.append({"type": risk_name, "count": count, "severity": self._risk_severity(risk_name)})

        if secret_exposure_count > 0:
            findings.append({"type": "secret_exposure", "count": secret_exposure_count, "severity": "critical"})

        return {
            "security_risk_count": security_risk_count,
            "insecure_code_pattern_count": insecure_code_pattern_count,
            "secret_exposure_count": secret_exposure_count,
            "security_issue_density": security_issue_density,
            "security_risk_breakdown": dict(risk_breakdown),
            "security_findings": findings,
        }

    @staticmethod
    def _risk_severity(risk_name: str) -> str:
        if risk_name in {"plaintext_secret_assignment", "unsafe_deserialization"}:
            return "high"
        if risk_name in {"dynamic_code_execution", "dangerous_subprocess", "tls_verification_disabled"}:
            return "high"
        return "medium"

    @staticmethod
    def _build_transition_matrix(sequence: List[str], labels: Iterable[str]) -> Dict[str, Dict[str, int]]:
        matrix: Dict[str, Dict[str, int]] = {
            source: {target: 0 for target in labels}
            for source in labels
        }
        for source, target in zip(sequence, sequence[1:]):
            if source not in matrix:
                continue
            if target not in matrix[source]:
                continue
            matrix[source][target] += 1
        return matrix

    @staticmethod
    def _matrix_to_edges(matrix: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        for source, targets in matrix.items():
            for target, weight in targets.items():
                if weight <= 0:
                    continue
                edges.append({"source": source, "target": target, "weight": weight})
        return edges

    @staticmethod
    def _count_context_switches(sequence: List[str]) -> int:
        return sum(1 for current, nxt in zip(sequence, sequence[1:]) if current != nxt)

    @staticmethod
    def _count_phase_loopbacks(sequence: List[str]) -> int:
        phase_rank = {
            "planning": 1,
            "implementation": 2,
            "debugging": 3,
            "testing": 4,
            "review": 5,
            "discussion": 0,
        }
        loopbacks = 0
        for current, nxt in zip(sequence, sequence[1:]):
            if current not in phase_rank or nxt not in phase_rank:
                continue
            if phase_rank[nxt] < phase_rank[current] and nxt != "discussion":
                loopbacks += 1
        return loopbacks

    @staticmethod
    def _workflow_findings(metrics: Dict[str, Any]) -> List[str]:
        findings: List[str] = []

        if metrics.get("security_risk_count", 0) > 0:
            findings.append(
                "Security risks detected in transcript code/content. Review security_findings for high-impact patterns."
            )

        if float(metrics.get("context_retention_score", 0.0)) < 25.0:
            findings.append(
                "Low context retention between user prompts and AI responses suggests alignment drift."
            )

        acceptance_rate = metrics.get("acceptance_rate")
        if acceptance_rate is not None and float(acceptance_rate) < 0.35:
            findings.append(
                "Low suggestion acceptance indicates poor first-pass output quality or unclear prompts."
            )

        if float(metrics.get("prompt_specificity_score", 0.0)) < 45.0:
            findings.append(
                "Prompt specificity is low; adding constraints and acceptance criteria should improve outcomes."
            )

        if float(metrics.get("phase_switch_rate", 0.0)) > 0.75:
            findings.append(
                "High phase switching suggests context fragmentation and potential workflow thrashing."
            )

        syntax_ratio = metrics.get("python_syntax_valid_ratio")
        if syntax_ratio is not None and float(syntax_ratio) < 0.7:
            findings.append(
                "A significant portion of Python-targeted code snippets appears non-parseable; increase validation before acceptance."
            )

        return findings

    @staticmethod
    def _phase_distribution(turns: List[Turn], total_turns: int) -> Dict[str, float]:
        phase_counts = {
            "planning": 0,
            "implementation": 0,
            "debugging": 0,
            "testing": 0,
            "review": 0,
            "discussion": 0,
        }

        for turn in turns:
            phase = turn.phase if turn.phase in phase_counts else "discussion"
            phase_counts[phase] += 1

        if total_turns == 0:
            return {
                "planning_phase_percentage": 0.0,
                "implementation_phase_percentage": 0.0,
                "debugging_phase_percentage": 0.0,
                "testing_phase_percentage": 0.0,
                "review_phase_percentage": 0.0,
                "discussion_phase_percentage": 0.0,
            }

        return {
            "planning_phase_percentage": round((phase_counts["planning"] / total_turns) * 100, 2),
            "implementation_phase_percentage": round(
                (phase_counts["implementation"] / total_turns) * 100, 2
            ),
            "debugging_phase_percentage": round((phase_counts["debugging"] / total_turns) * 100, 2),
            "testing_phase_percentage": round((phase_counts["testing"] / total_turns) * 100, 2),
            "review_phase_percentage": round((phase_counts["review"] / total_turns) * 100, 2),
            "discussion_phase_percentage": round((phase_counts["discussion"] / total_turns) * 100, 2),
        }
