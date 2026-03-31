# AI Coding Workflow Evaluation Harness

A secure, modular evaluation harness that ingests AI-assisted coding transcripts (ChatGPT, Claude, Copilot, and similar tools), normalizes conversation data, computes advanced workflow-quality metrics, and emits reports for both automated pipelines and interactive exploration.

## Why this project
- Build an end-to-end evaluation harness for transcript analysis.
- Quantify workflow effectiveness using repeatable heuristics.
- Keep architecture maintainable and extensible for future metrics.
- Start from a strict security baseline with minimal supply-chain risk.

## Security posture
- Core CLI and evaluation engine use Python standard library only.
- Optional frontend dependency is pinned in `requirements-ui.txt`.
- Dependency vetting notes are captured in `docs/dependency_vetting.md`.
- See `SECURITY.md` for governance, hardening, and intake policy.

## Architecture
Pipeline components are separated for maintainability:

1. Data ingestion (`DataIngestion`)
- Resolves input paths, directories, and glob patterns.
- Accepts `.json`, `.jsonl`, `.ndjson`, `.txt`, `.log`, and `.md` transcripts.

2. Parsing and preprocessing (`TranscriptParser`)
- Normalizes different transcript schemas into a common turn format.
- Detects speaker roles (`User`, `AI`, `System`, `Tool`).
- Extracts code blocks from fenced markdown.
- Preserves code block language hints when present (for safer language-aware analysis).
- Optionally redacts sensitive values (`--strip-sensitive`).
- Streams JSON, JSONL, and text inputs from disk instead of duplicating whole files in memory.

3. Phase detection (`PhaseDetector`)
- Labels turns as `planning`, `implementation`, `debugging`, `testing`, `review`, or `discussion`.
- Uses token-aware keyword/phrase heuristics, code cues, and continuity smoothing.

4. Metrics engine (`MetricsEngine`)
- Computes transcript-level, phase-level, code-quality, and security-risk metrics.
- Runs as a registry of evaluator components, so new metric groups can be added without rewriting one monolithic compute method.
- Includes transition matrices and relationship edges for graph visualizations.
- Produces context retention, prompt specificity, phase stability, and workflow findings.
- Uses AST-based Python security checks and language-aware code analysis to reduce regex false positives.

5. Scoring (`ScoreCalculator`)
- Converts advanced metrics into 0-100 scores plus confidence.
- Applies stronger security-weighted aggregation.
- Generates actionable recommendations per transcript.
- Supports external JSON weight overrides via `--score-config`.

6. Output reporting (`ReportGenerator`)
- Produces JSON, CSV, or Markdown reports.
- CSV exports include all discovered metrics and score dimensions dynamically.
- Supports multi-transcript comparison and advanced summary statistics.

7. Streamlit frontend (`streamlit_app.py`)
- Accepts transcript input from file uploads, pasted text, pasted JSON, or filesystem paths/globs.
- Provides interactive dashboards, heatmaps, and relationship graphs.
- Exports JSON, CSV, and Markdown reports.

## Normalized transcript shape
Each transcript is normalized to turns such as:

```json
[
  {
    "speaker": "User",
    "content": "Write a Python function to reverse a string.",
    "timestamp": "2026-03-31T09:00:00Z"
  },
  {
    "speaker": "AI",
    "content": "def reverse_string(s): return s[::-1]",
    "timestamp": "2026-03-31T09:00:05Z"
  }
]
```

## Supported input formats
- Plain text logs with prefixed turns:
  - `User: ...`
  - `AI: ...`
  - `[timestamp] User: ...`
- JSON exports with message records:
  - `speaker` or `role`
  - `content` / `message` / `text`
  - optional `timestamp`
- JSON Lines (`.jsonl` / `.ndjson`) where each line is a transcript object.

## Metrics included
- Turn counts:
  - `total_turns`, `user_turns`, `ai_turns`, `tool_turns`
  - `ai_to_user_turn_ratio`
- Text volume:
  - `user_word_count`, `ai_word_count`
  - `estimated_user_tokens`, `estimated_ai_tokens`, `estimated_total_tokens`
- Quality signals:
  - `correction_count`, `correction_ratio`
  - `positive_feedback_count`, `positive_feedback_ratio`
  - `ai_suggestion_count`, `accepted_suggestion_count`, `rejected_suggestion_count`, `acceptance_rate`
- Code/discussion balance:
  - `code_turn_count`, `discussion_turn_count`, `code_turn_ratio`
  - `ai_code_block_count`, `ai_code_line_count`
- Advanced workflow dynamics:
  - `context_retention_score`
  - `prompt_specificity_score`
  - `avg_response_latency_turns`
  - `context_switch_count`, `phase_switch_rate`, `phase_loopback_count`
  - `phase_transition_matrix`, `speaker_transition_matrix`
- Code quality heuristics:
  - `python_syntax_valid_ratio`
  - `avg_estimated_cyclomatic_complexity`
  - `type_hint_coverage`, `docstring_coverage`
  - `avg_function_length`, `long_line_count`, `comment_density`
- Security metrics:
  - `security_risk_count`
  - `insecure_code_pattern_count`
  - `secret_exposure_count`
  - `security_issue_density`
  - `security_findings`
- Phase distribution:
  - `planning_phase_percentage`
  - `implementation_phase_percentage`
  - `debugging_phase_percentage`
  - `testing_phase_percentage`
  - `review_phase_percentage`
  - `discussion_phase_percentage`
- Tooling traceability:
  - `tool_call_turn_count`

## Scoring model
The scoring engine maps raw metrics to 0-100 scores and computes:
- Metric-level confidence (heuristic confidence estimate)
- Overall weighted score
- Overall weighted confidence

Current weighted dimensions:
- Turn balance
- Correction efficiency
- Suggestion acceptance
- Feedback quality
- Code/discussion balance
- Phase coverage
- Prompt efficiency
- Observability
- Context retention
- Phase stability
- Code quality
- Security hygiene

## Recommendations and findings
Each transcript report includes:
- `recommendations`: prioritized improvement actions
- `analysis_artifacts.workflow_findings`: heuristic findings about flow quality
- `analysis_artifacts.security_findings`: pattern-level risk findings for code/content

## CLI usage
From project root:

```bash
python -m workflow_eval --help
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH='src'; python -m workflow_eval --help
```

Evaluate one or more transcripts:

```bash
PYTHONPATH=src python -m workflow_eval \
  transcripts/sample_session_1.txt \
  transcripts/sample_session_2.json \
  -o reports/report.json
```

Evaluate an entire directory and include normalized turns:

```bash
PYTHONPATH=src python -m workflow_eval transcripts --include-turns --strip-sensitive -o reports/full_report.json
```

Generate CSV or Markdown output:

```bash
PYTHONPATH=src python -m workflow_eval transcripts --output-format csv -o reports/report.csv
PYTHONPATH=src python -m workflow_eval transcripts --output-format md -o reports/report.md
```

Use a custom score-weight configuration:

```bash
PYTHONPATH=src python -m workflow_eval transcripts --score-config docs/scoring_config.example.json -o reports/weighted_report.json
```

## Example JSON result
```json
{
  "transcript": "session1.json",
  "metrics": {
    "total_turns": 20,
    "ai_to_user_turn_ratio": 1.5,
    "correction_count": 3,
    "positive_feedback_count": 2,
    "planning_phase_percentage": 10.0,
    "implementation_phase_percentage": 70.0,
    "overall_score": 78
  }
}
```

## Tests
Run unit tests with standard library `unittest`:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Streamlit frontend
Install optional UI dependency:

```bash
pip install -r requirements.txt
```

Run the interactive dashboard:

```bash
streamlit run streamlit_app.py
```

Key UI capabilities:
- Ingest transcripts via uploads, pasted text, pasted JSON, or path/glob input.
- Compare sessions with score/confidence overview charts.
- Visualize phase distribution and phase-transition heatmaps.
- Inspect relationship graphs for speaker and phase transitions.
- Download complete JSON/CSV/Markdown reports.

## Extensibility notes
- Add new metric groups by registering a `MetricEvaluator` with `MetricsEngine.register_evaluator(...)`.
- Keep parsing logic format-specific inside `TranscriptParser`.
- Use `ScoreCalculator.from_config_file(...)` to customize weighting without editing source.
- Use `ReportGenerator` to add new output formats or visualizations.
- The evaluator already processes transcripts one file at a time and can be adapted for parallel execution later.

Hope you all find it useful to upskill your prompting skills!!
