# Prompt Template for Copilot

Write a Python CLI app that reads transcript files from AI coding assistants (ChatGPT, Claude, Copilot, and similar tools), parses the conversation, computes workflow metrics, and outputs a JSON report of insights.

## Goal
Build an evaluation harness that runs end-to-end transcript analysis and supports future extension without major rewrites.

## Context
- Input can be one or more transcript files.
- Transcript sources can vary in schema and formatting.
- Outputs should be machine-readable for benchmarking and regression tracking.

## Requirements
1. Architecture
- Use modular components:
  - `DataIngestion`
  - `TranscriptParser`
  - `PhaseDetector`
  - `MetricsEngine`
  - `ScoreCalculator`
  - `ReportGenerator`
- Keep parsing, analysis, scoring, and output logic separated.

2. Input and normalization
- Support `.json`, `.txt`, and `.log` transcript formats.
- Normalize into turns: `{speaker, timestamp, content, code_blocks, phase}`.
- Detect speaker roles (`User`, `AI`, `System`, `Tool`).
- Extract code blocks from markdown fences.
- Handle malformed records safely without crashing entire batch.

3. Metrics
- Include transcript-level metrics:
  - total turns
  - AI-to-user ratio
  - estimated tokens
  - tool call count
- Include workflow quality metrics:
  - correction ratio
  - positive feedback count
  - suggestion acceptance/rejection rate
  - code vs discussion ratio
  - phase distribution

4. Scoring
- Score each metric from 0-100.
- Include confidence per metric.
- Compute weighted overall score.
- Document aggregation logic in code or README.

5. Output
- Output JSON report by default.
- Optional CSV/Markdown summaries.
- Support multiple transcripts and aggregate summary.
- Include parse errors per transcript without aborting entire run.

6. Quality and maintainability
- Use type hints and docstrings.
- Add unit tests for parser, metrics, and end-to-end pipeline.
- Provide sample transcripts under `transcripts/`.
- Document usage and output interpretation in README.

7. Security and supply chain
- Use Python standard library whenever possible.
- Do not add third-party dependencies unless strictly necessary.
- If adding dependencies, pin versions and document vulnerability checks.
