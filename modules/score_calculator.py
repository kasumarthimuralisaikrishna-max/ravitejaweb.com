"""Score normalization, JSON schema validation, and ATS analytics calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel, Field, field_validator

from config import MATCH_LEVELS


class MissingSkillDetail(BaseModel):
    """Detailed explanation for why a missing skill matters for the target role."""

    skill: str = Field(..., min_length=1)
    why_important: str = Field(..., min_length=1)


class ResumeAnalysisSchema(BaseModel):
    """Strict structured schema for validated resume analysis results."""

    ats_score: int = Field(..., ge=0, le=100)
    match_level: str = Field(default="")
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    technical_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    experience_analysis: str = Field(..., min_length=1)
    education_analysis: str = Field(..., min_length=1)
    resume_strengths: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    professional_summary: str = Field(..., min_length=1)
    final_assessment: str = Field(..., min_length=1)

    # Rich context fields supporting explicit vs. inferred skill clarity & missing skill rationale
    explicit_skills: list[str] = Field(default_factory=list)
    inferred_skills: list[str] = Field(default_factory=list)
    missing_skill_explanations: list[MissingSkillDetail] = Field(default_factory=list)

    @field_validator(
        "matching_skills",
        "missing_skills",
        "matched_keywords",
        "missing_keywords",
        "technical_skills",
        "soft_skills",
        "resume_strengths",
        "improvement_suggestions",
        "explicit_skills",
        "inferred_skills",
        mode="before",
    )
    @classmethod
    def _clean_string_list(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                cleaned.append(text)
        return cleaned

    @field_validator("ats_score", mode="before")
    @classmethod
    def _clamp_score(cls, value: Any) -> int:
        try:
            score = int(round(float(value)))
        except (TypeError, ValueError):
            raise ValueError("ats_score must be an integer between 0 and 100.")
        return max(0, min(100, score))


@dataclass(frozen=True)
class ScoreBreakdown:
    """Deterministic compatibility breakdown metrics computed from validated analysis."""

    ats_score: int
    match_level: str
    match_color: str
    keyword_coverage_pct: float
    skill_match_pct: float
    matched_keywords_count: int
    missing_keywords_count: int
    matching_skills_count: int
    missing_skills_count: int


def determine_match_level(score: int) -> tuple[str, str]:
    """Return (match_level_label, hex_color) for an integer ATS score (0-100)."""
    clamped = max(0, min(100, int(score)))
    for low, high, label, color in MATCH_LEVELS:
        if low <= clamped <= high:
            return label, color
    return "Needs Improvement", "#EF4444"


def normalize_analysis_result(raw_data: dict[str, Any]) -> ResumeAnalysisSchema:
    """
    Validate and normalize raw AI JSON output into a guaranteed ResumeAnalysisSchema.

    Enforces:
    - ATS score integer within [0, 100]
    - Canonical match_level aligned to score thresholds
    - Exactly 5 personalized improvement suggestions
    - Consistent missing_skills and missing_skill_explanations alignment
    """
    validated = ResumeAnalysisSchema.model_validate(raw_data)

    # Enforce canonical match level based on score thresholds
    canonical_level, _ = determine_match_level(validated.ats_score)
    validated.match_level = canonical_level

    # Ensure exactly 5 actionable improvement suggestions
    suggestions = list(validated.improvement_suggestions)
    fallback_suggestions = [
        "Align your technical skills section with the highest-priority tools and technologies named in the job description.",
        "Integrate missing role-specific keywords naturally into your experience bullet points and project descriptions.",
        "Tailor your professional summary to spotlight your most relevant experience and target role fit.",
        "Quantify achievements in your work and academic projects using measurable outcomes, scale, or impact metrics.",
        "Use clean, single-column ATS-friendly formatting with standard section headings such as Experience, Skills, and Education.",
    ]
    if len(suggestions) > 5:
        suggestions = suggestions[:5]
    elif len(suggestions) < 5:
        existing_lower = {s.lower() for s in suggestions}
        for fb in fallback_suggestions:
            if len(suggestions) >= 5:
                break
            if fb.lower() not in existing_lower:
                suggestions.append(fb)
                existing_lower.add(fb.lower())
    validated.improvement_suggestions = suggestions

    # Ensure every missing skill has a readable explanation if details were partially omitted
    detail_map = {d.skill.lower(): d for d in validated.missing_skill_explanations}
    normalized_details: list[MissingSkillDetail] = []
    for skill in validated.missing_skills:
        existing = detail_map.get(skill.lower())
        if existing:
            normalized_details.append(existing)
        else:
            normalized_details.append(
                MissingSkillDetail(
                    skill=skill,
                    why_important=(
                        f"Listed as a relevant capability in the target job description; "
                        f"demonstrating or building familiarity with {skill} can improve role alignment."
                    ),
                )
            )
    # Also ensure any skills in missing_skill_explanations appear in missing_skills if omitted
    missing_lower = {s.lower() for s in validated.missing_skills}
    for d in validated.missing_skill_explanations:
        if d.skill.lower() not in missing_lower:
            validated.missing_skills.append(d.skill)
            missing_lower.add(d.skill.lower())
            normalized_details.append(d)

    validated.missing_skill_explanations = normalized_details
    return validated


def compute_score_breakdown(analysis: ResumeAnalysisSchema) -> ScoreBreakdown:
    """Compute keyword coverage and skill match percentages deterministically."""
    label, color = determine_match_level(analysis.ats_score)

    matched_kw = len(analysis.matched_keywords)
    missing_kw = len(analysis.missing_keywords)
    total_kw = matched_kw + missing_kw
    keyword_coverage = round((matched_kw / total_kw) * 100, 1) if total_kw > 0 else 0.0

    matched_sk = len(analysis.matching_skills)
    missing_sk = len(analysis.missing_skills)
    total_sk = matched_sk + missing_sk
    skill_match = round((matched_sk / total_sk) * 100, 1) if total_sk > 0 else 0.0

    return ScoreBreakdown(
        ats_score=analysis.ats_score,
        match_level=label,
        match_color=color,
        keyword_coverage_pct=keyword_coverage,
        skill_match_pct=skill_match,
        matched_keywords_count=matched_kw,
        missing_keywords_count=missing_kw,
        matching_skills_count=matched_sk,
        missing_skills_count=missing_sk,
    )


def to_standard_json_dict(analysis: ResumeAnalysisSchema) -> dict[str, Any]:
    """Return the structured JSON dictionary matching the RESUMATE AI specification."""
    return {
        "ats_score": analysis.ats_score,
        "match_level": analysis.match_level,
        "matching_skills": analysis.matching_skills,
        "missing_skills": analysis.missing_skills,
        "matched_keywords": analysis.matched_keywords,
        "missing_keywords": analysis.missing_keywords,
        "technical_skills": analysis.technical_skills,
        "soft_skills": analysis.soft_skills,
        "experience_analysis": analysis.experience_analysis,
        "education_analysis": analysis.education_analysis,
        "resume_strengths": analysis.resume_strengths,
        "improvement_suggestions": analysis.improvement_suggestions,
        "professional_summary": analysis.professional_summary,
        "final_assessment": analysis.final_assessment,
        "explicit_skills": analysis.explicit_skills,
        "inferred_skills": analysis.inferred_skills,
        "missing_skill_explanations": [
            {"skill": d.skill, "why_important": d.why_important}
            for d in analysis.missing_skill_explanations
        ],
    }
