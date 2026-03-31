from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .models import TranscriptEvaluation


class ReportGenerator:
    """Render evaluation results in JSON, CSV, or Markdown format."""

    def __init__(self, include_turns: bool = False, include_timeline: bool = True) -> None:
        self.include_turns = include_turns
        self.include_timeline = include_timeline

    def build_report(self, evaluations: Sequence[TranscriptEvaluation]) -> Dict[str, Any]:
        successful = [evaluation for evaluation in evaluations if not evaluation.errors]
        failed = [evaluation for evaluation in evaluations if evaluation.errors]

        average_score = (
            round(sum(item.overall_score for item in successful) / len(successful), 2)
            if successful
            else 0.0
        )
        average_confidence = (
            round(sum(item.overall_confidence for item in successful) / len(successful), 2)
            if successful
            else 0.0
        )

        ranked = sorted(successful, key=lambda item: item.overall_score, reverse=True)

        total_security_risks = sum(
            int(item.metrics.get("security_risk_count", 0))
            for item in successful
        )
        average_context_retention = (
            round(
                sum(float(item.metrics.get("context_retention_score", 0.0)) for item in successful)
                / len(successful),
                2,
            )
            if successful
            else 0.0
        )

        summary: Dict[str, Any] = {
            "transcript_count": len(evaluations),
            "successful_evaluations": len(successful),
            "failed_evaluations": len(failed),
            "average_overall_score": average_score,
            "average_overall_confidence": average_confidence,
            "best_transcript": ranked[0].transcript if ranked else None,
            "total_security_risks": total_security_risks,
            "average_context_retention": average_context_retention,
        }

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "results": [
                evaluation.to_dict(
                    include_turns=self.include_turns,
                    include_timeline=self.include_timeline,
                )
                for evaluation in evaluations
            ],
        }

    def write(self, report: Dict[str, Any], output_path: Path, output_format: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "json":
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return

        if output_format == "csv":
            self._write_csv(report, output_path)
            return

        if output_format == "md":
            self._write_markdown(report, output_path)
            return

        raise ValueError(f"Unsupported output format: {output_format}")

    def _write_csv(self, report: Dict[str, Any], output_path: Path) -> None:
        rows: List[Dict[str, Any]] = []
        for result in report.get("results", []):
            row: Dict[str, Any] = {
                "transcript": result.get("transcript"),
                "overall_score": result.get("overall_score"),
                "overall_confidence": result.get("overall_confidence"),
                "recommendations": " | ".join(result.get("recommendations", [])),
                "errors": " | ".join(result.get("errors", [])),
            }
            for metric_name, metric_value in result.get("metrics", {}).items():
                row[f"metric_{metric_name}"] = self._serialize_tabular_value(metric_value)
            for score_name, score_payload in result.get("scores", {}).items():
                row[f"score_{score_name}"] = score_payload.get("score")
                row[f"score_confidence_{score_name}"] = score_payload.get("confidence")
            rows.append(row)

        base_fields = [
            "transcript",
            "overall_score",
            "overall_confidence",
            "recommendations",
            "errors",
        ]
        dynamic_fields = sorted(
            {
                key
                for row in rows
                for key in row
                if key not in base_fields
            }
        )
        fieldnames = base_fields + dynamic_fields

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _serialize_tabular_value(value: Any) -> Any:
        if isinstance(value, (str, int, float)) or value is None:
            return value
        return json.dumps(value, sort_keys=True)

    def _write_markdown(self, report: Dict[str, Any], output_path: Path) -> None:
        summary = report.get("summary", {})
        lines = [
            "# Workflow Evaluation Report",
            "",
            f"Generated at: {report.get('generated_at', 'unknown')}",
            "",
            "## Summary",
            "",
            f"- Transcript count: {summary.get('transcript_count', 0)}",
            f"- Successful evaluations: {summary.get('successful_evaluations', 0)}",
            f"- Failed evaluations: {summary.get('failed_evaluations', 0)}",
            f"- Average overall score: {summary.get('average_overall_score', 0)}",
            f"- Average confidence: {summary.get('average_overall_confidence', 0)}",
            f"- Best transcript: {summary.get('best_transcript', 'n/a')}",
            f"- Total security risks: {summary.get('total_security_risks', 0)}",
            f"- Average context retention: {summary.get('average_context_retention', 0)}",
            "",
            "## Results",
            "",
            "| Transcript | Score | Confidence | Turns | Tokens | Acceptance | Context Retention | Security Risks |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]

        for result in report.get("results", []):
            metrics = result.get("metrics", {})
            lines.append(
                "| {transcript} | {score} | {confidence} | {turns} | {tokens} | {acceptance} | {context_retention} | {security_risks} |".format(
                    transcript=result.get("transcript", "unknown"),
                    score=result.get("overall_score", 0),
                    confidence=result.get("overall_confidence", 0),
                    turns=metrics.get("total_turns", 0),
                    tokens=metrics.get("estimated_total_tokens", 0),
                    acceptance=metrics.get("acceptance_rate", "n/a"),
                    context_retention=metrics.get("context_retention_score", 0),
                    security_risks=metrics.get("security_risk_count", 0),
                )
            )

            recommendations = result.get("recommendations", [])
            if recommendations:
                lines.append("\nRecommendations:")
                for recommendation in recommendations:
                    lines.append(f"- {recommendation}")

            errors = result.get("errors", [])
            if errors:
                lines.append(f"\nErrors for {result.get('transcript', 'unknown')}: {'; '.join(errors)}")

        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
