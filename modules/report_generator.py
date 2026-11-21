"""PDF report generation for RESUMATE AI analysis results using ReportLab."""

from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import APP_NAME, ATS_DISCLAIMER
from modules.score_calculator import (
    ResumeAnalysisSchema,
    compute_score_breakdown,
)


class ReportGenerationError(Exception):
    """Raised when PDF report generation fails."""


def _safe(text: str) -> str:
    """Escape XML/HTML reserved characters for ReportLab Paragraph safety."""
    return escape(str(text or "")).replace("\n", "<br/>")


def _build_bullet_list(items: list[str], style: ParagraphStyle, empty_text: str = "None identified.") -> list:
    if not items:
        return [Paragraph(f"<i>{_safe(empty_text)}</i>", style)]
    return [Paragraph(f"&bull;&nbsp;&nbsp;{_safe(item)}", style) for item in items]


def generate_analysis_pdf_report(
    analysis: ResumeAnalysisSchema,
    filename: str,
    analyzed_at: str | None = None,
) -> bytes:
    """
    Generate a polished, multi-section PDF analysis report in memory.

    Returns PDF bytes ready for download.
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=0.65 * inch,
            rightMargin=0.65 * inch,
            topMargin=0.65 * inch,
            bottomMargin=0.65 * inch,
            title=f"{APP_NAME} - ATS Analysis Report",
            author=APP_NAME,
        )

        styles = getSampleStyleSheet()
        navy = colors.HexColor("#0F172A")
        violet = colors.HexColor("#6D28D9")
        blue = colors.HexColor("#2563EB")
        slate = colors.HexColor("#475569")
        light_bg = colors.HexColor("#F8FAFC")
        border_col = colors.HexColor("#CBD5E1")
        green = colors.HexColor("#059669")
        amber = colors.HexColor("#D97706")
        red = colors.HexColor("#DC2626")

        title_style = ParagraphStyle(
            "ResumateTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=colors.white,
            spaceAfter=4,
            alignment=TA_LEFT,
        )
        subtitle_style = ParagraphStyle(
            "ResumateSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            textColor=colors.HexColor("#E2E8F0"),
            leading=13,
        )
        section_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            textColor=violet,
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "BodyTextCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            textColor=navy,
            leading=14,
            spaceAfter=4,
        )
        muted_style = ParagraphStyle(
            "MutedText",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            textColor=slate,
            leading=12,
        )

        breakdown = compute_score_breakdown(analysis)
        score_color = (
            green
            if breakdown.ats_score >= 85
            else blue
            if breakdown.ats_score >= 70
            else amber
            if breakdown.ats_score >= 50
            else red
        )

        timestamp_str = analyzed_at or datetime.now().strftime("%Y-%m-%d %H:%M")

        story = []

        # Branded Banner Header
        header_data = [
            [
                Paragraph(f"{APP_NAME} — ATS Compatibility Report", title_style),
            ],
            [
                Paragraph(
                    f"Resume: <b>{_safe(filename)}</b> &nbsp;|&nbsp; Generated: {_safe(timestamp_str)}",
                    subtitle_style,
                )
            ],
        ]
        header_table = Table(header_data, colWidths=[7.2 * inch])
        header_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1E1B4B")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 16),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 16),
                    ("TOPPADDING", (0, 0), (-1, 0), 14),
                    ("BOTTOMPADDING", (0, 1), (-1, 1), 14),
                    ("TOPPADDING", (0, 1), (-1, 1), 2),
                ]
            )
        )
        story.append(header_table)
        story.append(Spacer(1, 12))

        # Score KPI Strip
        score_label_style = ParagraphStyle(
            "ScoreLabel", parent=body_style, fontSize=8.5, textColor=slate, spaceAfter=2
        )
        score_val_style = ParagraphStyle(
            "ScoreVal",
            parent=body_style,
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=score_color,
        )
        kpi_val_style = ParagraphStyle(
            "KpiVal",
            parent=body_style,
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=navy,
        )

        kpi_table = Table(
            [
                [
                    [
                        Paragraph("ESTIMATED ATS SCORE", score_label_style),
                        Paragraph(f"{breakdown.ats_score} / 100", score_val_style),
                    ],
                    [
                        Paragraph("MATCH LEVEL", score_label_style),
                        Paragraph(_safe(breakdown.match_level), kpi_val_style),
                    ],
                    [
                        Paragraph("KEYWORD COVERAGE", score_label_style),
                        Paragraph(f"{breakdown.keyword_coverage_pct:.1f}%", kpi_val_style),
                    ],
                    [
                        Paragraph("SKILL MATCH RATE", score_label_style),
                        Paragraph(f"{breakdown.skill_match_pct:.1f}%", kpi_val_style),
                    ],
                ]
            ],
            colWidths=[1.8 * inch, 1.9 * inch, 1.75 * inch, 1.75 * inch],
        )
        kpi_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), light_bg),
                    ("BOX", (0, 0), (-1, -1), 0.75, border_col),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, border_col),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(kpi_table)
        story.append(Spacer(1, 8))
        story.append(Paragraph(_safe(ATS_DISCLAIMER), muted_style))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.6, color=border_col))

        # AI-Generated Professional Summary
        story.append(Paragraph("AI-Generated Professional Summary", section_style))
        summary_box = Table(
            [[Paragraph(_safe(analysis.professional_summary), body_style)]],
            colWidths=[7.2 * inch],
        )
        summary_box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF2FF")),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, violet),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(summary_box)

        # Matching Skills & Missing Skills
        story.append(Paragraph("Matching Skills", section_style))
        matching_str = ", ".join(analysis.matching_skills) if analysis.matching_skills else "None identified"
        story.append(Paragraph(_safe(matching_str), body_style))

        if analysis.explicit_skills or analysis.inferred_skills:
            exp_str = ", ".join(analysis.explicit_skills) if analysis.explicit_skills else "None"
            inf_str = ", ".join(analysis.inferred_skills) if analysis.inferred_skills else "None"
            story.append(
                Paragraph(
                    f"<b>Explicitly Stated:</b> {_safe(exp_str)}<br/>"
                    f"<b>Context-Inferred:</b> {_safe(inf_str)}",
                    body_style,
                )
            )

        story.append(Paragraph("Missing Skills & Why They Matter", section_style))
        if analysis.missing_skill_explanations:
            for detail in analysis.missing_skill_explanations:
                story.append(
                    Paragraph(
                        f"&bull;&nbsp;&nbsp;<b>{_safe(detail.skill)}:</b> {_safe(detail.why_important)}",
                        body_style,
                    )
                )
        else:
            story.extend(_build_bullet_list(analysis.missing_skills, body_style, "No critical skill gaps detected."))

        # Keyword Analysis
        story.append(Paragraph("Keyword Analysis", section_style))
        matched_kw_str = ", ".join(analysis.matched_keywords) if analysis.matched_keywords else "None"
        missing_kw_str = ", ".join(analysis.missing_keywords) if analysis.missing_keywords else "None"
        story.append(Paragraph(f"<b>Matched Keywords:</b> {_safe(matched_kw_str)}", body_style))
        story.append(Paragraph(f"<b>Missing Keywords:</b> {_safe(missing_kw_str)}", body_style))

        # Technical & Soft Skills
        story.append(Paragraph("Skills Inventory", section_style))
        tech_str = ", ".join(analysis.technical_skills) if analysis.technical_skills else "None identified"
        soft_str = ", ".join(analysis.soft_skills) if analysis.soft_skills else "None identified"
        story.append(Paragraph(f"<b>Technical Skills:</b> {_safe(tech_str)}", body_style))
        story.append(Paragraph(f"<b>Soft Skills:</b> {_safe(soft_str)}", body_style))

        # Resume Strengths
        story.append(Paragraph("Resume Strengths", section_style))
        story.extend(_build_bullet_list(analysis.resume_strengths, body_style))

        # Experience & Education Analysis
        story.append(Paragraph("Experience & Education Alignment", section_style))
        story.append(Paragraph(f"<b>Experience Analysis:</b> {_safe(analysis.experience_analysis)}", body_style))
        story.append(Paragraph(f"<b>Education Analysis:</b> {_safe(analysis.education_analysis)}", body_style))

        # Five Improvement Suggestions
        story.append(Paragraph("Top 5 Improvement Suggestions", section_style))
        for idx, suggestion in enumerate(analysis.improvement_suggestions, start=1):
            story.append(Paragraph(f"<b>{idx}.</b> {_safe(suggestion)}", body_style))

        # Final Assessment
        story.append(Paragraph("Final Assessment", section_style))
        story.append(Paragraph(_safe(analysis.final_assessment), body_style))

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        if not pdf_bytes:
            raise ReportGenerationError("Generated PDF report was empty.")
        return pdf_bytes

    except ReportGenerationError:
        raise
    except Exception as exc:
        raise ReportGenerationError(
            f"Unable to generate the PDF report ({type(exc).__name__}). You can still download the JSON report."
        ) from exc
