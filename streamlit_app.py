from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from workflow_eval.evaluator import WorkflowEvaluator
from workflow_eval.ingestion import DataIngestion
from workflow_eval.metrics import PHASES
from workflow_eval.metrics import MetricsEngine
from workflow_eval.parsing import TranscriptParser
from workflow_eval.phase_detection import PhaseDetector
from workflow_eval.reporting import ReportGenerator
from workflow_eval.scoring import ScoreCalculator


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #1a1a1b;
            --bg-panel: #ffffff;
            --border: #eaebed;
            --accent: #2563eb;
            --accent-light: #eff6ff;
            --success: #059669;
            --warning: #d97706;
            --danger: #dc2626;
        }
        
        .main .block-container {
            max-width: 1400px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        /* God-level styling modifications */
        h1 {
            font-size: 2.8rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.05em !important;
            color: #0f172a !important;
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem !important;
        }
        h2 { font-size: 1.8rem !important; font-weight: 700 !important; color: #1e293b !important; }
        h3 { font-size: 1.4rem !important; font-weight: 600 !important; color: #334155 !important; }

        .stButton>button {
            border-radius: 8px !important;
            border: none !important;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            padding: 0.6rem 1.4rem !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2), 0 2px 4px -1px rgba(37, 99, 235, 0.1) !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3), 0 4px 6px -2px rgba(37, 99, 235, 0.1) !important;
        }

        .metric-card {
            background-color: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05), 0 1px 2px 0 rgba(0,0,0,0.03);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        }
        
        .metric-title { font-size: 0.95rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
        .metric-value { font-size: 2.2rem; font-weight: 800; color: #0f172a; line-height: 1.2; }
        .metric-value.excellent { color: var(--success); }
        .metric-value.warning { color: var(--warning); }
        .metric-value.critical { color: var(--danger); }
        .metric-desc { font-size: 0.9rem; color: #94a3b8; margin-top: 0.2rem; }

        .recommendation-box {
            background: var(--accent-light);
            border-left: 4px solid var(--accent);
            padding: 1.2rem;
            border-radius: 0 8px 8px 0;
            margin: 1.5rem 0;
            color: #1e3a8a;
            font-weight: 500;
        }
        
        .findings-box {
            background: #fef2f2;
            border-left: 4px solid var(--danger);
            padding: 1.2rem;
            border-radius: 0 8px 8px 0;
            margin: 1.5rem 0;
            color: #991b1b;
        }

        /* Override st metrics */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 800 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            font-weight: 600 !important;
            color: #64748b !important;
        }

        hr {
            border-color: #e2e8f0;
            margin: 3rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _collect_paths_from_mode(mode: str) -> Dict[str, Any]:
    paths: List[Path] = []
    raw_candidates: List[str] = []

    if mode == "Upload files":
        uploads = st.file_uploader(
            "Upload transcript files",
            type=["json", "jsonl", "ndjson", "txt", "log", "md"],
            accept_multiple_files=True,
        )
        if uploads:
            temp_dir = Path(tempfile.mkdtemp(prefix="workflow_eval_upload_"))
            for upload in uploads:
                target = temp_dir / upload.name
                target.write_bytes(upload.getvalue())
                paths.append(target)

    elif mode == "Paste plain text":
        raw_text = st.text_area(
            "Paste one or more transcripts",
            height=220,
            placeholder=(
                "Use lines like 'User: ...' and 'AI: ...'.\n"
                "To split multiple transcripts, use this delimiter on its own line:\n"
                "---TRANSCRIPT---"
            ),
        )
        if raw_text.strip():
            temp_dir = Path(tempfile.mkdtemp(prefix="workflow_eval_text_"))
            chunks = [chunk.strip() for chunk in raw_text.split("\n---TRANSCRIPT---\n") if chunk.strip()]
            for index, chunk in enumerate(chunks, start=1):
                target = temp_dir / f"pasted_text_{index}.txt"
                target.write_text(chunk, encoding="utf-8")
                paths.append(target)

    elif mode == "Paste JSON":
        raw_json = st.text_area(
            "Paste JSON transcript payload",
            height=220,
            placeholder=(
                "Paste a JSON object/array, or multiple JSON objects separated by the delimiter:\n"
                "---TRANSCRIPT---"
            ),
        )
        if raw_json.strip():
            temp_dir = Path(tempfile.mkdtemp(prefix="workflow_eval_json_"))
            chunks = [chunk.strip() for chunk in raw_json.split("\n---TRANSCRIPT---\n") if chunk.strip()]
            for index, chunk in enumerate(chunks, start=1):
                target = temp_dir / f"pasted_json_{index}.json"
                target.write_text(chunk, encoding="utf-8")
                paths.append(target)

    else:
        raw_paths = st.text_area(
            "Enter paths or glob patterns (one per line)",
            height=160,
            placeholder=(
                "Example:\n"
                "transcripts\n"
                "transcripts/**/*.json\n"
                "C:/work/logs/session_*.txt"
            ),
        )
        if raw_paths.strip():
            raw_candidates = [line.strip() for line in raw_paths.splitlines() if line.strip()]

    return {
        "mode": mode,
        "paths": paths,
        "raw_candidates": raw_candidates,
    }


def _resolve_paths_from_payload(payload: Dict[str, Any]) -> List[Path]:
    mode = str(payload.get("mode", ""))
    if mode == "Filesystem paths/globs":
        candidates = payload.get("raw_candidates", [])
        if not candidates:
            return []
        ingestion = DataIngestion()
        return ingestion.collect_input_files(candidates)

    return list(payload.get("paths", []))


def _flatten_phase_percentages(metrics: Dict[str, object], transcript_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for phase in PHASES:
        key = f"{phase}_phase_percentage"
        if key in metrics:
            rows.append(
                {
                    "transcript": transcript_name,
                    "phase": phase,
                    "percentage": float(metrics.get(key, 0.0)),
                }
            )
    return rows


def _phase_overview_chart(results: List[Dict[str, object]]) -> None:
    rows: List[Dict[str, object]] = []
    for result in results:
        rows.extend(_flatten_phase_percentages(result.get("metrics", {}), str(result.get("transcript", "unknown"))))

    if not rows:
        return

    spec = {
        "background": "#ffffff",
        "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
        "encoding": {
            "x": {"field": "transcript", "type": "nominal", "title": "Transcript"},
            "y": {"field": "percentage", "type": "quantitative", "title": "Phase %", "stack": "normalize"},
            "color": {
                "field": "phase",
                "type": "nominal",
                "scale": {
                    "range": ["#0f8b8d", "#2a9d8f", "#264653", "#f4a261", "#e76f51", "#8aa29e"]
                },
            },
            "tooltip": [
                {"field": "transcript", "type": "nominal"},
                {"field": "phase", "type": "nominal"},
                {"field": "percentage", "type": "quantitative"},
            ],
        },
        "config": {
            "axis": {
                "labelColor": "#12303b",
                "titleColor": "#12303b",
                "gridColor": "#d8e4e8",
                "tickColor": "#9fb4bc",
            },
            "legend": {"labelColor": "#12303b", "titleColor": "#12303b"},
        },
    }
    st.vega_lite_chart(rows, spec, use_container_width=True)


def _score_overview_chart(results: List[Dict[str, object]]) -> None:
    rows = [
        {
            "transcript": result.get("transcript", "unknown"),
            "overall_score": result.get("overall_score", 0),
            "overall_confidence": result.get("overall_confidence", 0),
        }
        for result in results
    ]

    spec = {
        "background": "#ffffff",
        "layer": [
            {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 6, "cornerRadiusTopRight": 6},
                "encoding": {
                    "x": {"field": "transcript", "type": "nominal", "title": "Transcript"},
                    "y": {"field": "overall_score", "type": "quantitative", "title": "Score"},
                    "color": {
                        "field": "overall_score",
                        "type": "quantitative",
                        "scale": {"scheme": "teals"},
                    },
                    "tooltip": [
                        {"field": "transcript", "type": "nominal"},
                        {"field": "overall_score", "type": "quantitative"},
                        {"field": "overall_confidence", "type": "quantitative"},
                    ],
                },
            },
            {
                "mark": {"type": "line", "color": "#e76f51", "point": True, "strokeWidth": 2},
                "encoding": {
                    "x": {"field": "transcript", "type": "nominal"},
                    "y": {"field": "overall_confidence", "type": "quantitative", "title": "Confidence"},
                },
            },
        ]
        ,
        "config": {
            "axis": {
                "labelColor": "#12303b",
                "titleColor": "#12303b",
                "gridColor": "#d8e4e8",
                "tickColor": "#9fb4bc",
            },
            "legend": {"labelColor": "#12303b", "titleColor": "#12303b"},
        },
    }
    st.vega_lite_chart(rows, spec, use_container_width=True)


def _phase_transition_heatmap(result: Dict[str, object]) -> None:
    metrics = result.get("metrics", {})
    matrix = metrics.get("phase_transition_matrix", {})
    rows: List[Dict[str, object]] = []
    for source, targets in matrix.items():
        for target, count in targets.items():
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "count": count,
                }
            )

    if not rows:
        st.info("No transition matrix available for this transcript.")
        return

    spec = {
        "background": "#ffffff",
        "layer": [
            {
                "mark": "rect",
                "encoding": {
                    "x": {"field": "target", "type": "nominal", "title": "To phase"},
                    "y": {"field": "source", "type": "nominal", "title": "From phase"},
                    "color": {
                        "field": "count",
                        "type": "quantitative",
                        "scale": {"scheme": "yellowgreenblue"},
                    },
                    "tooltip": [
                        {"field": "source", "type": "nominal"},
                        {"field": "target", "type": "nominal"},
                        {"field": "count", "type": "quantitative"},
                    ],
                },
            },
            {
                "mark": {"type": "text", "baseline": "middle", "fontSize": 11, "fontWeight": "bold"},
                "encoding": {
                    "x": {"field": "target", "type": "nominal"},
                    "y": {"field": "source", "type": "nominal"},
                    "text": {"field": "count", "type": "quantitative"},
                    "color": {
                        "condition": {"test": "datum.count >= 2", "value": "white"},
                        "value": "#1a2f37",
                    },
                },
            },
        ],
        "config": {
            "axis": {
                "labelColor": "#12303b",
                "titleColor": "#12303b",
                "tickColor": "#9fb4bc",
            }
        },
    }
    st.vega_lite_chart(rows, spec, use_container_width=True)


def _render_relationship_graph(result: Dict[str, object], key_name: str, title: str) -> None:
    artifacts = result.get("analysis_artifacts", {})
    edges = artifacts.get(key_name, [])
    if not edges:
        st.info(f"No {title.lower()} data available.")
        return

    lines = [
        "digraph G {",
        "rankdir=LR;",
        'graph [bgcolor="transparent", splines=true, overlap=false, pad="0.25"];',
        'node [shape=ellipse, style="filled,bold", fillcolor="#eef8fa", color="#0f8b8d", fontname="Helvetica", fontcolor="#0d2c37", penwidth=1.3];',
        'edge [color="#264653", fontname="Helvetica", fontcolor="#102d38", fontsize=11];',
    ]

    for edge in edges:
        source = str(edge.get("source", "unknown")).replace('"', "")
        target = str(edge.get("target", "unknown")).replace('"', "")
        weight = int(edge.get("weight", 0))
        width = max(1.4, min(weight, 8) * 0.75)
        lines.append(
            f'"{source}" -> "{target}" [label="{weight}", penwidth={width:.2f}, arrowsize=0.85];'
        )

    lines.append("}")
    st.graphviz_chart("\n".join(lines), use_container_width=True)


def _relationship_edge_table(result: Dict[str, object], key_name: str) -> List[Dict[str, object]]:
    artifacts = result.get("analysis_artifacts", {})
    edges = artifacts.get(key_name, [])
    rows = [
        {
            "source": edge.get("source", "unknown"),
            "target": edge.get("target", "unknown"),
            "weight": int(edge.get("weight", 0)),
        }
        for edge in edges
    ]
    return sorted(rows, key=lambda item: int(item.get("weight", 0)), reverse=True)


def _scores_table(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metric_name, payload in result.get("scores", {}).items():
        rows.append(
            {
                "metric": metric_name,
                "score": payload.get("score"),
                "confidence": payload.get("confidence"),
                "value": payload.get("value"),
                "rationale": payload.get("rationale"),
            }
        )
    return sorted(rows, key=lambda item: float(item.get("score") or 0), reverse=True)


def _build_report(paths: List[Path], strip_sensitive: bool, keep_system_messages: bool, enable_phase_detection: bool) -> Dict[str, object]:
    evaluator = _build_evaluator(strip_sensitive, keep_system_messages, enable_phase_detection)
    evaluations = evaluator.evaluate_files(paths)
    generator = ReportGenerator(include_turns=True, include_timeline=True)
    return generator.build_report(evaluations)


def _render_summary(summary: Dict[str, object]) -> None:
    st.markdown("### Evaluation Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Transcripts", int(summary.get("transcript_count", 0)))
    col2.metric("Successful", int(summary.get("successful_evaluations", 0)))
    col3.metric("Avg Score", float(summary.get("average_overall_score", 0.0)))
    col4.metric("Avg Confidence", float(summary.get("average_overall_confidence", 0.0)))
    col5.metric("Security Risks", int(summary.get("total_security_risks", 0)))


def main() -> None:
    st.set_page_config(
        page_title="Workflow Evaluation Control Center",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    st.title("Workflow Evaluation Control Center")
    st.markdown(
        "<span class='caption-chip'>Advanced Transcript Intelligence</span>"
        "<span class='caption-chip'>Security-Weighted Scoring</span>"
        "<span class='caption-chip'>Graph-Driven Diagnostics</span>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Input Control")
        input_mode = st.radio(
            "How do you want to provide transcripts?",
            options=["Upload files", "Paste plain text", "Paste JSON", "Filesystem paths/globs"],
        )

        input_payload = _collect_paths_from_mode(input_mode)

        st.subheader("Analysis Settings")
        strip_sensitive = st.checkbox("Strip sensitive content", value=True)
        keep_system_messages = st.checkbox("Keep system messages", value=False)
        enable_phase_detection = st.checkbox("Enable phase detection", value=True)

        run_button = st.button("Run Full Evaluation", type="primary", use_container_width=True)

    report = st.session_state.get("workflow_report")

    if run_button:
        try:
            paths = _resolve_paths_from_payload(input_payload)
            if not paths:
                st.warning("No transcript inputs were found. Provide at least one transcript source.")
            else:
                report = _build_report(paths, strip_sensitive, keep_system_messages, enable_phase_detection)
                st.session_state["workflow_report"] = report
                st.success("Evaluation completed successfully.")
        except Exception as exc:
            st.exception(exc)
            return

    if not report:
        st.info("Upload or paste transcripts, then click Run Full Evaluation.")
        return

    summary = report.get("summary", {})
    results: List[Dict[str, object]] = report.get("results", [])

    _render_summary(summary)

    st.markdown("### Portfolio-Level Charts")
    c1, c2 = st.columns([1.1, 1.0])
    with c1:
        st.markdown("#### Score and Confidence by Transcript")
        _score_overview_chart(results)
    with c2:
        st.markdown("#### Phase Distribution Across Transcripts")
        _phase_overview_chart(results)

    transcript_names = [str(result.get("transcript", "unknown")) for result in results]
    selected_name = st.selectbox("Inspect transcript", options=transcript_names)
    selected_result = next((item for item in results if item.get("transcript") == selected_name), results[0])

    st.markdown("### Deep Dive")
    deep_1, deep_2 = st.columns([1.0, 1.1])

    with deep_1:
        st.markdown("#### Dimension Scores")
        st.dataframe(_scores_table(selected_result), use_container_width=True, hide_index=True)

        st.markdown("#### Recommendations")
        for recommendation in selected_result.get("recommendations", []):
            st.write(f"- {recommendation}")

        findings = selected_result.get("analysis_artifacts", {}).get("workflow_findings", [])
        if findings:
            st.markdown("#### Workflow Findings")
            for finding in findings:
                st.write(f"- {finding}")

    with deep_2:
        st.markdown("#### Phase Transition Heatmap")
        _phase_transition_heatmap(selected_result)

        st.markdown("#### Speaker Relationship Graph")
        _render_relationship_graph(selected_result, "speaker_transition_edges", "Speaker Relationship Graph")
        speaker_edges = _relationship_edge_table(selected_result, "speaker_transition_edges")
        if speaker_edges:
            st.dataframe(speaker_edges, use_container_width=True, hide_index=True)

        st.markdown("#### Phase Relationship Graph")
        _render_relationship_graph(selected_result, "phase_transition_edges", "Phase Relationship Graph")
        phase_edges = _relationship_edge_table(selected_result, "phase_transition_edges")
        if phase_edges:
            st.dataframe(phase_edges, use_container_width=True, hide_index=True)

    st.markdown("### Security Findings")
    security_findings = selected_result.get("analysis_artifacts", {}).get("security_findings", [])
    if security_findings:
        st.dataframe(security_findings, use_container_width=True, hide_index=True)
    else:
        st.success("No direct security anti-pattern findings were detected in this transcript.")

    st.markdown("### Export")
    report_json = json.dumps(report, indent=2)
    st.download_button(
        "Download JSON report",
        data=report_json,
        file_name="workflow_report.json",
        mime="application/json",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        generator = ReportGenerator(include_turns=False, include_timeline=True)

        csv_path = output_dir / "workflow_report.csv"
        generator.write(report, csv_path, "csv")
        st.download_button(
            "Download CSV summary",
            data=csv_path.read_text(encoding="utf-8"),
            file_name="workflow_report.csv",
            mime="text/csv",
        )

        md_path = output_dir / "workflow_report.md"
        generator.write(report, md_path, "md")
        st.download_button(
            "Download Markdown report",
            data=md_path.read_text(encoding="utf-8"),
            file_name="workflow_report.md",
            mime="text/markdown",
        )

    with st.expander("Raw report JSON"):
        st.code(report_json, language="json")


if __name__ == "__main__":
    main()
