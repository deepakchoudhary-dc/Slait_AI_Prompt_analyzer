from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ScoringConstants:
    CLAMP_MIN: float = 0.0
    CLAMP_MAX: float = 100.0

C = ScoringConstants()

from typing import Any, Dict, List, Optional, Tuple

from .models import MetricScore


def _clamp(value: float, lower: float = C.CLAMP_MIN, upper: float = C.CLAMP_MAX) -> float:
    return max(lower, min(upper, value))


class ScoreCalculator:
    """Convert raw metrics into normalized quality scores."""

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "turn_balance": 0.07,
        "correction_efficiency": 0.11,
        "suggestion_acceptance": 0.10,
        "feedback_quality": 0.04,
        "code_discussion_balance": 0.08,
        "phase_coverage": 0.07,
        "prompt_efficiency": 0.07,
        "observability": 0.04,
        "context_retention": 0.09,
        "phase_stability": 0.06,
        "code_quality": 0.09,
        "security_hygiene": 0.18,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:     
        self.weights = self._resolve_weights(weights or {})

    @classmethod
    def from_config_file(cls, path: Path) -> "ScoreCalculator":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Score configuration must be a JSON object.")      

        weight_payload = payload.get("weights", payload)
        if not isinstance(weight_payload, dict):
            raise ValueError("Score configuration must provide a 'weights' object.")

        weights: Dict[str, float] = {}
        for key, value in weight_payload.items():
            try:
                weights[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid weight for '{key}': {value!r}") from exc

        return cls(weights=weights)

    def _resolve_weights(self, overrides: Dict[str, float]) -> Dict[str, float]:
        unknown = sorted(set(overrides).difference(self.DEFAULT_WEIGHTS))
        if unknown:
            raise ValueError(f"Unknown score weights: {', '.join(unknown)}")    

        merged = dict(self.DEFAULT_WEIGHTS)
        merged.update(overrides)
        total = sum(weight for weight in merged.values() if weight > 0)
        if total <= 0:
            raise ValueError("Score weights must sum to a positive value.")     

        return {
            name: (max(weight, 0.0) / total)
            for name, weight in merged.items()
        }

    def score(self, metrics: Dict[str, Any]) -> Tuple[Dict[str, MetricScore], float, float]:
        scores: Dict[str, MetricScore] = {}

        total_turns = int(metrics.get("total_turns", 0))
        user_turns = int(metrics.get("user_turns", 0))
        ai_suggestions = int(metrics.get("ai_suggestion_count", 0))

        # 1) User/AI interaction balance.
        ratio = metrics.get("ai_to_user_turn_ratio")
        if ratio is None:
            turn_balance = MetricScore(
                value=None,
                score=50.0,
                confidence=45.0,
                rationale="No user turns found; interaction balance is underdetermined.",
            )
        else:
            distance = abs(float(ratio) - 1.0)
            turn_balance = MetricScore(
                value=ratio,
                score=_clamp(100.0 - (distance * 40.0)),
                confidence=_clamp(55.0 + min(total_turns, 20) * 2.0),
                rationale="Closer to a 1:1 conversational cadence generally indicates tighter iteration loops.",
            )
        scores["turn_balance"] = turn_balance

        # 2) Lower correction ratio is better.
        correction_ratio = float(metrics.get("correction_ratio", 0.0))
        scores["correction_efficiency"] = MetricScore(
            value=correction_ratio,
            score=_clamp((1.0 - correction_ratio) * 100.0),
            confidence=_clamp(65.0 + min(user_turns, 20) * 1.5),
            rationale="Frequent correction prompts often indicate low first-pass utility.",
        )

        # 3) Suggestion acceptance reflects whether AI outputs are usable.      
        acceptance_rate = metrics.get("acceptance_rate")
        if acceptance_rate is None:
            suggestion_acceptance = MetricScore(
                value=None,
                score=50.0,
                confidence=50.0,
                rationale="No AI suggestions detected, so acceptance cannot be estimated.",
            )
        else:
            suggestion_acceptance = MetricScore(
                value=acceptance_rate,
                score=_clamp(float(acceptance_rate) * 100.0),
                confidence=_clamp(50.0 + min(ai_suggestions, 20) * 2.0),        
                rationale="Higher acceptance usually indicates outputs are aligned with developer intent.",
            )
        scores["suggestion_acceptance"] = suggestion_acceptance

        # 4) Some positive feedback is a useful signal, but excessive praise is neutralized.
        feedback_ratio = float(metrics.get("positive_feedback_ratio", 0.0))     
        if feedback_ratio == 0 and user_turns > 0:
            feedback_score = 40.0
        elif feedback_ratio <= 0.35:
            feedback_score = _clamp((feedback_ratio / 0.35) * 100.0)
        else:
            feedback_score = _clamp(100.0 - (feedback_ratio - 0.35) * 180.0)    

        scores["feedback_quality"] = MetricScore(
            value=feedback_ratio,
            score=feedback_score,
            confidence=_clamp(60.0 + min(user_turns, 20) * 1.2),
            rationale="Measured acknowledgements are treated as a weak success proxy.",
        )

        # 5) Balanced code/discussion turns generally indicate healthy flow.    
        code_ratio = float(metrics.get("code_turn_ratio", 0.0))
        if code_ratio < 0.30:
            code_balance_score = _clamp((code_ratio / 0.30) * 100.0)
        elif code_ratio <= 0.75:
            code_balance_score = 100.0
        else:
            code_balance_score = _clamp(100.0 - (code_ratio - 0.75) * 300.0)    

        scores["code_discussion_balance"] = MetricScore(
            value=code_ratio,
            score=code_balance_score,
            confidence=_clamp(65.0 + min(total_turns, 20) * 1.5),
            rationale="An effective workflow alternates implementation with reasoning and verification.",
        )

        # 6) Reward sessions that cover multiple phases.
        phase_keys = (
            "planning_phase_percentage",
            "implementation_phase_percentage",
            "debugging_phase_percentage",
            "testing_phase_percentage",
        )
        covered_phases = sum(1 for key in phase_keys if float(metrics.get(key, 0.0)) > 0.0)
        phase_coverage_ratio = covered_phases / len(phase_keys)

        scores["phase_coverage"] = MetricScore(
            value=phase_coverage_ratio,
            score=_clamp(phase_coverage_ratio * 100.0),
            confidence=_clamp(70.0 + min(total_turns, 20) * 1.0),
            rationale="Coverage across planning, implementation, debugging, and testing is rewarded.",
        )

        # 7) Penalize very short or excessively verbose prompts.
        avg_prompt_words = float(metrics.get("avg_user_prompt_words", 0.0))     
        if avg_prompt_words == 0:
            prompt_score = 50.0
            prompt_confidence = 40.0
        elif avg_prompt_words < 4:
            prompt_score = 45.0
            prompt_confidence = 60.0
        elif avg_prompt_words <= 60:
            prompt_score = 100.0
            prompt_confidence = 80.0
        elif avg_prompt_words <= 120:
            prompt_score = _clamp(100.0 - (avg_prompt_words - 60.0) * 1.2)      
            prompt_confidence = 80.0
        else:
            prompt_score = 25.0
            prompt_confidence = 75.0

        scores["prompt_efficiency"] = MetricScore(
            value=avg_prompt_words,
            score=prompt_score,
            confidence=prompt_confidence,
            rationale="Concise but explicit prompts tend to improve assistant reliability.",
        )

        # 8) Track explicit tooling activity (neutral-to-positive signal).      
        tool_call_turn_count = int(metrics.get("tool_call_turn_count", 0))      
        observability_score = 70.0 if tool_call_turn_count == 0 else _clamp(70.0 + min(tool_call_turn_count, 5) * 6.0)
        scores["observability"] = MetricScore(
            value=tool_call_turn_count,
            score=observability_score,
            confidence=80.0,
            rationale="Transparent tool usage can improve reproducibility and debugging traceability.",
        )

        # 9) Context retention from user intent to AI responses.
        context_retention = float(metrics.get("context_retention_score", 0.0))  
        scores["context_retention"] = MetricScore(
            value=context_retention,
            score=_clamp(context_retention),
            confidence=_clamp(60.0 + min(user_turns, 20) * 1.5),
            rationale="Higher lexical overlap across turn pairs indicates stronger intent retention.",
        )

        # 10) Penalize excessive phase thrashing and workflow loopbacks.        
        phase_switch_rate = float(metrics.get("phase_switch_rate", 0.0))        
        phase_loopback_count = int(metrics.get("phase_loopback_count", 0))      
        phase_stability_score = _clamp(100.0 - (phase_switch_rate * 70.0) - (phase_loopback_count * 5.0))
        scores["phase_stability"] = MetricScore(
            value={"phase_switch_rate": phase_switch_rate, "phase_loopback_count": phase_loopback_count},
            score=phase_stability_score,
            confidence=_clamp(70.0 + min(total_turns, 25) * 1.0),
            rationale="Stable progression through phases usually reduces context-switch overhead.",
        )

        # 11) Code quality synthesis from syntax validity, complexity, typing, docs, and readability.
        code_block_count = int(metrics.get("ai_code_block_count", 0))
        syntax_ratio = metrics.get("python_syntax_valid_ratio")
        avg_complexity = float(metrics.get("avg_estimated_cyclomatic_complexity", 0.0))
        type_hint_coverage = metrics.get("type_hint_coverage")
        docstring_coverage = metrics.get("docstring_coverage")
        long_line_count = int(metrics.get("long_line_count", 0))

        if code_block_count == 0:
            code_quality_score = 65.0
            code_quality_confidence = 45.0
        else:
            syntax_score = 55.0 if syntax_ratio is None else _clamp(float(syntax_ratio) * 100.0)
            if avg_complexity <= 8:
                complexity_score = 100.0
            else:
                complexity_score = _clamp(100.0 - ((avg_complexity - 8.0) * 8.0))

            hint_score = 60.0 if type_hint_coverage is None else _clamp(float(type_hint_coverage) * 100.0)
            doc_score = 60.0 if docstring_coverage is None else _clamp(float(docstring_coverage) * 100.0)
            readability_score = _clamp(100.0 - (long_line_count * 3.0))

            code_quality_score = (
                (0.35 * syntax_score)
                + (0.20 * complexity_score)
                + (0.20 * hint_score)
                + (0.15 * doc_score)
                + (0.10 * readability_score)
            )
            code_quality_confidence = _clamp(55.0 + min(code_block_count, 20) * 2.0)

        scores["code_quality"] = MetricScore(
            value={
                "syntax_valid_ratio": syntax_ratio,
                "avg_complexity": avg_complexity,
                "type_hint_coverage": type_hint_coverage,
                "docstring_coverage": docstring_coverage,
                "long_line_count": long_line_count,
            },
            score=code_quality_score,
            confidence=code_quality_confidence,
            rationale="Code quality rewards parseable, maintainable, and well-documented outputs.",
        )

        # 12) Security hygiene strongly weighted in the global score.
        security_risk_count = int(metrics.get("security_risk_count", 0))        
        secret_exposure_count = int(metrics.get("secret_exposure_count", 0))    
        insecure_dependency_mention_count = int(metrics.get("insecure_dependency_mention_count", 0))
        security_discussion_count = int(metrics.get("security_discussion_count", 0))

        security_score = 100.0
        security_score -= security_risk_count * 18.0
        security_score -= secret_exposure_count * 28.0
        security_score -= insecure_dependency_mention_count * 10.0
        security_score += min(security_discussion_count * 2.0, 10.0)
        security_score = _clamp(security_score)

        security_confidence = _clamp(65.0 + min(total_turns, 25) * 1.2)
        scores["security_hygiene"] = MetricScore(
            value={
                "security_risk_count": security_risk_count,
                "secret_exposure_count": secret_exposure_count,
                "security_discussion_count": security_discussion_count,
            },
            score=security_score,
            confidence=security_confidence,
            rationale="Security anti-patterns and exposed secrets are heavily penalized.",
        )

        weighted_score = 0.0
        weighted_confidence = 0.0
        for name, metric in scores.items():
            weight = self.weights.get(name, 0.0)
            weighted_score += metric.score * weight
            weighted_confidence += metric.confidence * weight

        return scores, round(weighted_score, 2), round(weighted_confidence, 2)  

    def recommendations(self, metrics: Dict[str, Any], scores: Dict[str, MetricScore]) -> List[str]:
        output: List[str] = []

        if int(metrics.get("security_risk_count", 0)) > 0:
            output.append(
                "Resolve detected security risks in transcript code patterns before adoption (review security_findings)."
            )

        if int(metrics.get("secret_exposure_count", 0)) > 0:
            output.append(
                "Secrets or credential-like material were detected; rotate secrets and enforce redaction in transcripts."
            )

        acceptance_rate = metrics.get("acceptance_rate")
        if acceptance_rate is not None and float(acceptance_rate) < 0.4:        
            output.append(
                "Increase prompt precision with explicit constraints and acceptance tests to improve first-pass acceptance."
            )

        if float(metrics.get("context_retention_score", 0.0)) < 30.0:
            output.append(
                "Context retention is weak; ask the assistant to restate goals and preserve identifiers across iterations."
            )

        syntax_ratio = metrics.get("python_syntax_valid_ratio")
        if syntax_ratio is not None and float(syntax_ratio) < 0.8:
            output.append(
                "A significant portion of code snippets is not parseable Python; add syntax checks before accepting outputs."
            )

        phase_stability = scores.get("phase_stability")
        if phase_stability and phase_stability.score < 55.0:
            output.append(
                "Workflow phase thrashing detected; separate planning, implementation, and debugging into explicit steps."
            )

        if not output:
            output.append(
                "Workflow quality appears strong; continue with the same structure and enforce automated security checks in CI."
            )

        return output