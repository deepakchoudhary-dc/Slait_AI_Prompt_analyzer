from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence

from .evaluator import WorkflowEvaluator
from .ingestion import DataIngestion
from .metrics import MetricsEngine
from .parsing import TranscriptParser
from .phase_detection import PhaseDetector
from .reporting import ReportGenerator
from .scoring import ScoreCalculator

LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workflow-eval",
        description=(
            "Evaluate AI-assisted coding transcripts and generate workflow quality insights."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input transcript paths, directories, or glob patterns.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="report.json",
        help="Output report path. Default: report.json",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "md"],
        default=None,
        help="Output format. If omitted, format is inferred from output extension.",
    )
    parser.add_argument(
        "--strip-sensitive",
        action="store_true",
        help="Redact common secrets and sensitive tokens from normalized content.",
    )
    parser.add_argument(
        "--keep-system-messages",
        action="store_true",
        help="Retain system turns instead of filtering them out during normalization.",
    )
    parser.add_argument(
        "--disable-phase-detection",
        action="store_true",
        help="Disable planning/implementation/debugging phase detection.",
    )
    parser.add_argument(
        "--include-turns",
        action="store_true",
        help="Include normalized turns in the report payload.",
    )
    parser.add_argument(
        "--score-config",
        default=None,
        help="Optional JSON score-weight configuration file.",
    )
    parser.add_argument(
        "--no-timeline",
        action="store_true",
        help="Do not include phase timeline markers in the report.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO",
    )
    return parser


def infer_output_format(output_path: Path) -> str:
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".md", ".markdown"}:
        return "md"
    return "json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    ingestion = DataIngestion()
    try:
        input_files = ingestion.collect_input_files(args.inputs)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 1

    try:
        score_calculator = (
            ScoreCalculator.from_config_file(Path(args.score_config))
            if args.score_config
            else ScoreCalculator()
        )
    except (OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    evaluator = WorkflowEvaluator(
        parser=TranscriptParser(
            remove_system_messages=not args.keep_system_messages,
            strip_sensitive=args.strip_sensitive,
        ),
        phase_detector=PhaseDetector(enabled=not args.disable_phase_detection),
        metrics_engine=MetricsEngine(),
        score_calculator=score_calculator,
    )

    evaluations = evaluator.evaluate_files(input_files)

    output_path = Path(args.output)
    output_format = args.output_format or infer_output_format(output_path)

    report_generator = ReportGenerator(
        include_turns=args.include_turns,
        include_timeline=not args.no_timeline,
    )
    report = report_generator.build_report(evaluations)
    report_generator.write(report, output_path, output_format)

    summary = report.get("summary", {})
    LOGGER.info(
        "Evaluated %s transcripts (%s succeeded, %s failed).",
        summary.get("transcript_count", 0),
        summary.get("successful_evaluations", 0),
        summary.get("failed_evaluations", 0),
    )
    LOGGER.info(
        "Average overall score: %s",
        summary.get("average_overall_score", 0),
    )
    LOGGER.info("Report written to: %s", output_path.resolve())

    return 2 if summary.get("failed_evaluations", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
