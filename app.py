"""RESUMATE AI — AI Resume Analyzer & ATS Score Checker (Streamlit Application)."""

from __future__ import annotations

import html
import json
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from config import (
    APP_NAME,
    APP_TAGLINE,
    ATS_DISCLAIMER,
    CSS_PATH,
    MAX_PDF_SIZE_MB,
    MIN_JD_WORDS,
)
from modules.database import (
    AnalysisRecord,
    clear_history,
    delete_analysis,
    get_all_history,
    get_dashboard_stats,
    init_db,
    save_analysis,
)
from modules.gemini_analyzer import (
    AnalysisError,
    analyze_resume_with_gemini,
    validate_job_description,
)
from modules.pdf_extractor import (
    PDFExtractionError,
    PDFExtractionResult,
    extract_resume_from_pdf,
)
from modules.report_generator import (
    ReportGenerationError,
    generate_analysis_pdf_report,
)
from modules.score_calculator import (
    ResumeAnalysisSchema,
    compute_score_breakdown,
    to_standard_json_dict,
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RESUMATE AI — AI Resume Analyzer & ATS Score Checker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_PAGES = [
    "Dashboard",
    "Analyze Resume",
    "Analysis History",
    "Resume Improvement Tips",
    "About RESUMATE AI",
]


def load_custom_css() -> None:
    if CSS_PATH.exists():
        css = CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    defaults = {
        "nav_page": "Dashboard",
        "extracted_resume": None,  # PDFExtractionResult | None
        "uploaded_file_sig": None,
        "job_description_input": "",
        "current_analysis": None,  # ResumeAnalysisSchema | None
        "current_filename": "",
        "current_analyzed_at": "",
        "uploader_reset_key": 0,
        "confirm_clear_history": False,
        "selected_history_id": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def navigate_to(page: str) -> None:
    if page in NAV_PAGES:
        st.session_state["nav_page"] = page


def esc(text: str) -> str:
    return html.escape(str(text or ""))


def render_copy_button(text_to_copy: str, button_label: str, key_suffix: str) -> None:
    """Render an iframe-safe clipboard copy button with visual confirmation."""
    payload_js = json.dumps(text_to_copy)
    safe_label = esc(button_label)
    btn_id = f"copy_btn_{key_suffix}"
    components.html(
        f"""
        <div style="margin: 2px 0 6px 0;">
          <button id="{btn_id}" style="
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(59, 130, 246, 0.25));
            border: 1px solid rgba(139, 92, 246, 0.55);
            color: #F8FAFC;
            font-family: Inter, -apple-system, sans-serif;
            font-size: 13px;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
          ">{safe_label}</button>
        </div>
        <script>
          const btn = document.getElementById("{btn_id}");
          const textVal = {payload_js};
          btn.addEventListener("click", async () => {{
            try {{
              await navigator.clipboard.writeText(textVal);
              btn.textContent = "✓ Copied to Clipboard";
              btn.style.borderColor = "#10B981";
              btn.style.background = "rgba(16, 185, 129, 0.22)";
            }} catch (e) {{
              const ta = document.createElement("textarea");
              ta.value = textVal;
              document.body.appendChild(ta);
              ta.select();
              document.execCommand("copy");
              document.body.removeChild(ta);
              btn.textContent = "✓ Copied to Clipboard";
              btn.style.borderColor = "#10B981";
            }}
            setTimeout(() => {{
              btn.textContent = "{safe_label}";
              btn.style.borderColor = "rgba(139, 92, 246, 0.55)";
              btn.style.background = "linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(59, 130, 246, 0.25))";
            }}, 2200);
          }});
        </script>
        """,
        height=48,
    )


# ---------------------------------------------------------------------------
# Plotly Visualizations
# ---------------------------------------------------------------------------
def build_ats_gauge_chart(score: int, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 34, "color": "#F8FAFC", "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B", "tickfont": {"color": "#94A3B8", "size": 11}},
                "bar": {"color": color, "thickness": 0.72},
                "bgcolor": "rgba(15, 23, 42, 0.65)",
                "borderwidth": 1,
                "bordercolor": "rgba(139, 92, 246, 0.30)",
                "steps": [
                    {"range": [0, 49], "color": "rgba(239, 68, 68, 0.14)"},
                    {"range": [50, 69], "color": "rgba(245, 158, 11, 0.14)"},
                    {"range": [70, 84], "color": "rgba(59, 130, 246, 0.14)"},
                    {"range": [85, 100], "color": "rgba(16, 185, 129, 0.16)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=24, r=24, t=24, b=16),
        height=250,
    )
    return fig


def build_coverage_chart(
    matched_kw: int,
    missing_kw: int,
    matching_sk: int,
    missing_sk: int,
) -> go.Figure:
    categories = ["Job Keywords", "Core Skills"]
    fig = go.Figure(
        data=[
            go.Bar(
                name="Matched",
                y=categories,
                x=[matched_kw, matching_sk],
                orientation="h",
                marker=dict(color="#10B981", line=dict(color="rgba(16,185,129,0.9)", width=1)),
                text=[matched_kw, matching_sk],
                textposition="inside",
            ),
            go.Bar(
                name="Missing",
                y=categories,
                x=[missing_kw, missing_sk],
                orientation="h",
                marker=dict(color="#F97316", line=dict(color="rgba(249,115,22,0.9)", width=1)),
                text=[missing_kw, missing_sk],
                textposition="inside",
            ),
        ]
    )
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0", family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
        margin=dict(l=10, r=16, t=34, b=16),
        height=210,
        xaxis=dict(title="Count", gridcolor="rgba(148, 163, 184, 0.14)", zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_history_trend_chart(records: list[AnalysisRecord]) -> go.Figure:
    ordered = list(reversed(records[:15]))
    labels = [f"#{r.id} ({r.filename[:14]})" for r in ordered]
    scores = [r.ats_score for r in ordered]
    colors_list = [
        "#10B981" if s >= 85 else "#3B82F6" if s >= 70 else "#F59E0B" if s >= 50 else "#EF4444"
        for s in scores
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=scores,
            marker=dict(color=colors_list, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=[f"{s}%" for s in scores],
            textposition="outside",
            name="ATS Score",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0", family="Inter"),
        margin=dict(l=16, r=16, t=24, b=28),
        height=260,
        yaxis=dict(range=[0, 112], title="Score", gridcolor="rgba(148, 163, 184, 0.14)"),
        xaxis=dict(tickangle=-18, gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-sidebar">
              <div class="brand-logo-row">
                <div class="brand-icon">R</div>
                <div>
                  <div class="brand-title">{esc(APP_NAME)}</div>
                  <div class="brand-sub">ATS &amp; Career Intelligence</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = st.radio(
            "Navigation",
            options=NAV_PAGES,
            index=NAV_PAGES.index(st.session_state["nav_page"]),
            key="sidebar_nav_radio",
        )
        if selected != st.session_state["nav_page"]:
            st.session_state["nav_page"] = selected

        st.markdown("---")
        st.markdown(
            """
            <div class="glass-card" style="padding: 0.95rem 1rem;">
              <div style="font-weight:700; font-size:0.86rem; color:#C4B5FD; margin-bottom:0.35rem;">
                Privacy-First Processing
              </div>
              <div style="font-size:0.8rem; color:#94A3B8; line-height:1.45;">
                Uploaded PDFs are parsed in memory and never permanently stored. Only structured compatibility reports are saved to local history.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return st.session_state["nav_page"]


# ---------------------------------------------------------------------------
# Shared Results Dashboard Renderer
# ---------------------------------------------------------------------------
def render_analysis_results(
    analysis: ResumeAnalysisSchema,
    filename: str,
    analyzed_at: str = "",
    key_prefix: str = "active",
    show_reanalyze_controls: bool = True,
) -> None:
    breakdown = compute_score_breakdown(analysis)
    badge_bg = {
        "#10B981": "rgba(16, 185, 129, 0.18)",
        "#3B82F6": "rgba(59, 130, 246, 0.18)",
        "#F59E0B": "rgba(245, 158, 11, 0.18)",
        "#EF4444": "rgba(239, 68, 68, 0.18)",
    }.get(breakdown.match_color, "rgba(139, 92, 246, 0.18)")

    st.markdown("---")
    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:0.75rem; margin-bottom:0.6rem;">
          <div>
            <h2 style="margin:0; font-size:1.55rem;">Analysis Results Dashboard</h2>
            <div style="color:#94A3B8; font-size:0.88rem; margin-top:2px;">
              Resume: <strong style="color:#E2E8F0;">{esc(filename)}</strong>
              {f" &nbsp;|&nbsp; Analyzed: {esc(analyzed_at)}" if analyzed_at else ""}
            </div>
          </div>
          <span class="match-badge" style="background:{badge_bg}; color:{breakdown.match_color}; border:1px solid {breakdown.match_color};">
            {esc(breakdown.match_level)}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action Bar: PDF export, JSON export, Reanalyze, Clear
    act_cols = st.columns([1.2, 1.1, 1.5, 1.0])
    with act_cols[0]:
        try:
            pdf_bytes = generate_analysis_pdf_report(
                analysis=analysis,
                filename=filename,
                analyzed_at=analyzed_at or datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"RESUMATE_Report_{filename.rsplit('.', 1)[0]}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"{key_prefix}_pdf_dl",
            )
        except ReportGenerationError as exc:
            st.warning(str(exc))

    with act_cols[1]:
        json_str = json.dumps(to_standard_json_dict(analysis), indent=2, ensure_ascii=False)
        st.download_button(
            "🧾 Download JSON",
            data=json_str.encode("utf-8"),
            file_name=f"RESUMATE_Analysis_{filename.rsplit('.', 1)[0]}.json",
            mime="application/json",
            use_container_width=True,
            key=f"{key_prefix}_json_dl",
        )

    if show_reanalyze_controls:
        with act_cols[2]:
            if st.button("🔁 Reanalyze with Another JD", use_container_width=True, key=f"{key_prefix}_reanalyze"):
                st.session_state["job_description_input"] = ""
                st.session_state["current_analysis"] = None
                navigate_to("Analyze Resume")
                st.rerun()
        with act_cols[3]:
            if st.button("🧹 Clear Analysis", use_container_width=True, key=f"{key_prefix}_clear_res"):
                st.session_state["current_analysis"] = None
                st.session_state["current_filename"] = ""
                st.session_state["current_analyzed_at"] = ""
                st.rerun()

    st.markdown(f'<div class="disclaimer-banner">ℹ️ {esc(ATS_DISCLAIMER)}</div>', unsafe_allow_html=True)

    # A. ATS Compatibility Score & Coverage Overview
    col_gauge, col_metrics = st.columns([1.05, 1.35], gap="large")
    with col_gauge:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🎯 ATS Compatibility Score</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_ats_gauge_chart(breakdown.ats_score, breakdown.match_color),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"{key_prefix}_gauge",
        )
        st.markdown(
            f"""
            <div style="text-align:center; margin-top:-6px;">
              <span class="match-badge" style="background:{badge_bg}; color:{breakdown.match_color}; border:1px solid {breakdown.match_color};">
                {breakdown.ats_score}/100 &middot; {esc(breakdown.match_level)}
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_metrics:
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">Keyword Coverage</div>
                  <div class="kpi-value" style="color:#60A5FA;">{breakdown.keyword_coverage_pct:.1f}%</div>
                  <div class="kpi-sub">{breakdown.matched_keywords_count} matched / {breakdown.missing_keywords_count} missing</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">Skill Alignment</div>
                  <div class="kpi-value" style="color:#34D399;">{breakdown.skill_match_pct:.1f}%</div>
                  <div class="kpi-sub">{breakdown.matching_skills_count} matching / {breakdown.missing_skills_count} missing</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="glass-card" style="margin-top:0.85rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Keyword &amp; Skill Coverage</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_coverage_chart(
                breakdown.matched_keywords_count,
                breakdown.missing_keywords_count,
                breakdown.matching_skills_count,
                breakdown.missing_skills_count,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"{key_prefix}_coverage_bar",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # B. Matching Skills & C. Missing Skills
    col_match, col_miss = st.columns(2, gap="large")

    with col_match:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">✅ Matching Skills</div>', unsafe_allow_html=True)
        if analysis.matching_skills:
            chips = "".join(
                f'<span class="skill-chip-green">✓ {esc(sk)}</span>'
                for sk in analysis.matching_skills
            )
            st.markdown(f'<div class="chip-grid">{chips}</div>', unsafe_allow_html=True)
        else:
            st.info("No direct matching skills were detected between this resume and job description.")

        if analysis.explicit_skills or analysis.inferred_skills:
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            if analysis.explicit_skills:
                exp_chips = "".join(
                    f'<span class="skill-chip-blue">{esc(s)}</span>'
                    for s in analysis.explicit_skills
                )
                st.markdown(
                    f"<div style='font-size:0.82rem; font-weight:600; color:#93C5FD; margin-bottom:4px;'>Explicitly Mentioned in Resume</div>"
                    f'<div class="chip-grid">{exp_chips}</div>',
                    unsafe_allow_html=True,
                )
            if analysis.inferred_skills:
                inf_chips = "".join(
                    f'<span class="skill-chip-purple">{esc(s)}</span>'
                    for s in analysis.inferred_skills
                )
                st.markdown(
                    f"<div style='font-size:0.82rem; font-weight:600; color:#C4B5FD; margin:10px 0 4px 0;'>Inferred from Project/Work Context</div>"
                    f'<div class="chip-grid">{inf_chips}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_miss:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">⚠️ Missing Skills &amp; Job Relevance</div>', unsafe_allow_html=True)
        if analysis.missing_skill_explanations:
            for item in analysis.missing_skill_explanations:
                st.markdown(
                    f"""
                    <div class="missing-skill-card">
                      <div class="missing-skill-title">{esc(item.skill)}</div>
                      <div class="missing-skill-reason">{esc(item.why_important)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        elif analysis.missing_skills:
            for sk in analysis.missing_skills:
                st.markdown(
                    f"""
                    <div class="missing-skill-card">
                      <div class="missing-skill-title">{esc(sk)}</div>
                      <div class="missing-skill-reason">Requested in the job description and not clearly evidenced on the resume.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("No major skill gaps identified for the core job requirements.")
        st.markdown("</div>", unsafe_allow_html=True)

    # D. Keyword Analysis & Technical/Soft Skills Inventory
    col_kw1, col_kw2 = st.columns(2, gap="large")
    with col_kw1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔑 Keyword Analysis</div>', unsafe_allow_html=True)
        matched_kw_html = (
            "".join(f'<span class="skill-chip-green">{esc(k)}</span>' for k in analysis.matched_keywords)
            if analysis.matched_keywords
            else '<span style="color:#94A3B8; font-size:0.86rem;">None detected</span>'
        )
        missing_kw_html = (
            "".join(f'<span class="skill-chip-amber">{esc(k)}</span>' for k in analysis.missing_keywords)
            if analysis.missing_keywords
            else '<span style="color:#94A3B8; font-size:0.86rem;">None missing</span>'
        )
        st.markdown(
            f"""
            <div style="font-size:0.82rem; font-weight:600; color:#6EE7B7; margin-bottom:4px;">Matched Keywords ({len(analysis.matched_keywords)})</div>
            <div class="chip-grid" style="margin-bottom:0.9rem;">{matched_kw_html}</div>
            <div style="font-size:0.82rem; font-weight:600; color:#FCD34D; margin-bottom:4px;">Missing Keywords ({len(analysis.missing_keywords)})</div>
            <div class="chip-grid">{missing_kw_html}</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_kw2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🧩 Technical &amp; Soft Skills Detected</div>', unsafe_allow_html=True)
        tech_html = (
            "".join(f'<span class="skill-chip-blue">{esc(t)}</span>' for t in analysis.technical_skills)
            if analysis.technical_skills
            else '<span style="color:#94A3B8; font-size:0.86rem;">None detected</span>'
        )
        soft_html = (
            "".join(f'<span class="skill-chip-purple">{esc(s)}</span>' for s in analysis.soft_skills)
            if analysis.soft_skills
            else '<span style="color:#94A3B8; font-size:0.86rem;">None detected</span>'
        )
        st.markdown(
            f"""
            <div style="font-size:0.82rem; font-weight:600; color:#93C5FD; margin-bottom:4px;">Technical Skills</div>
            <div class="chip-grid" style="margin-bottom:0.9rem;">{tech_html}</div>
            <div style="font-size:0.82rem; font-weight:600; color:#DDD6FE; margin-bottom:4px;">Soft &amp; Interpersonal Skills</div>
            <div class="chip-grid">{soft_html}</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # E. Resume Strengths & F. Experience / Education Analysis
    col_str, col_exp = st.columns([1.0, 1.3], gap="large")
    with col_str:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🌟 Resume Strengths</div>', unsafe_allow_html=True)
        if analysis.resume_strengths:
            for strength in analysis.resume_strengths:
                st.markdown(f'<div class="strength-item">✓ {esc(strength)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#94A3B8;">No distinct strengths highlighted.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_exp:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">💼 Experience &amp; Education Alignment</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="margin-bottom:0.9rem;">
              <div style="font-size:0.82rem; font-weight:700; color:#93C5FD; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.3rem;">Experience Alignment</div>
              <div style="color:#E2E8F0; font-size:0.92rem; line-height:1.58;">{esc(analysis.experience_analysis)}</div>
            </div>
            <div style="margin-bottom:0.9rem;">
              <div style="font-size:0.82rem; font-weight:700; color:#C4B5FD; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.3rem;">Education Alignment</div>
              <div style="color:#E2E8F0; font-size:0.92rem; line-height:1.58;">{esc(analysis.education_analysis)}</div>
            </div>
            <div>
              <div style="font-size:0.82rem; font-weight:700; color:#6EE7B7; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.3rem;">Final Assessment</div>
              <div style="color:#E2E8F0; font-size:0.92rem; line-height:1.58;">{esc(analysis.final_assessment)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # G. Five Improvement Suggestions
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🚀 5 Personalized Improvement Suggestions</div>', unsafe_allow_html=True)
    for idx, suggestion in enumerate(analysis.improvement_suggestions, start=1):
        st.markdown(
            f"""
            <div class="suggestion-card">
              <div class="suggestion-num">{idx}</div>
              <div class="suggestion-text">{esc(suggestion)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    suggestions_plain = "\n".join(
        f"{idx}. {item}" for idx, item in enumerate(analysis.improvement_suggestions, start=1)
    )
    render_copy_button(suggestions_plain, "📋 Copy Improvement Suggestions", f"{key_prefix}_suggestions")
    st.markdown("</div>", unsafe_allow_html=True)

    # H. AI-Generated Professional Summary
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">✨ AI-Generated Professional Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#94A3B8; font-size:0.84rem; margin-bottom:0.65rem;">'
        "Tailored exclusively from your verified resume experience, education, and skills—no fabricated qualifications."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="summary-box">{esc(analysis.professional_summary)}</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    render_copy_button(analysis.professional_summary, "📋 Copy Summary", f"{key_prefix}_summary")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def render_dashboard_page() -> None:
    stats = get_dashboard_stats()
    history = get_all_history(limit=15)

    st.markdown(
        f"""
        <div class="hero-card">
          <div class="hero-eyebrow">{esc(APP_NAME)} &middot; ATS Intelligence</div>
          <h1 class="hero-title">{esc(APP_TAGLINE)}</h1>
          <p class="hero-desc">
            Benchmark your resume against real job descriptions using Google Gemini AI.
            Uncover matched and missing skills, optimize ATS keyword coverage, and generate
            tailored improvement plans and downloadable PDF reports.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta_col, _ = st.columns([1.2, 2.8])
    with cta_col:
        if st.button("⚡ Quick Analyze Resume", type="primary", use_container_width=True, key="dash_quick_analyze"):
            navigate_to("Analyze Resume")
            st.rerun()

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3, gap="medium")
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Total Resumes Analyzed</div>
              <div class="kpi-value">{stats.total_analyzed}</div>
              <div class="kpi-sub">{"Completed evaluations stored locally" if stats.total_analyzed > 0 else "No analyses recorded yet"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        avg_display = f"{stats.average_score}/100" if stats.average_score is not None else "—"
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Average ATS Score</div>
              <div class="kpi-value" style="color:#60A5FA;">{avg_display}</div>
              <div class="kpi-sub">{"Across all recorded analyses" if stats.average_score is not None else "Run an analysis to calculate"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        high_display = f"{stats.highest_score}/100" if stats.highest_score is not None else "—"
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Highest ATS Score</div>
              <div class="kpi-value" style="color:#34D399;">{high_display}</div>
              <div class="kpi-sub">{"Best compatibility achieved" if stats.highest_score is not None else "Awaiting your first scan"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)

    if not history:
        st.markdown(
            """
            <div class="empty-state-card">
              <div class="empty-state-title">No Resume Analyses Yet</div>
              <p class="empty-state-desc">
                Your dashboard displays real metrics from your evaluations. Upload a PDF resume and paste a target
                job description in <strong>Analyze Resume</strong> to see your ATS score, skill gaps, and tailored recommendations.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        col_chart, col_recent = st.columns([1.2, 1.1], gap="large")
        with col_chart:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📈 Recent ATS Score Trend</div>', unsafe_allow_html=True)
            st.plotly_chart(
                build_history_trend_chart(history),
                use_container_width=True,
                config={"displayModeBar": False},
                key="dash_trend_chart",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_recent:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🕒 Recent Analysis Activity</div>', unsafe_allow_html=True)
            for rec in history[:5]:
                score_col = (
                    "#10B981" if rec.ats_score >= 85
                    else "#3B82F6" if rec.ats_score >= 70
                    else "#F59E0B" if rec.ats_score >= 50
                    else "#EF4444"
                )
                r1, r2 = st.columns([3.2, 1.1])
                with r1:
                    st.markdown(
                        f"""
                        <div style="padding:0.45rem 0; border-bottom:1px solid rgba(148,163,184,0.12);">
                          <div style="font-weight:600; font-size:0.9rem; color:#F8FAFC;">
                            {esc(rec.filename)} &middot;
                            <span style="color:{score_col}; font-family:'JetBrains Mono', monospace;">{rec.ats_score}/100</span>
                          </div>
                          <div style="font-size:0.8rem; color:#94A3B8;">{esc(rec.job_preview)}</div>
                          <div style="font-size:0.74rem; color:#64748B;">{esc(rec.created_at)} &middot; {esc(rec.match_level)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with r2:
                    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                    if st.button("Open", key=f"dash_open_{rec.id}", use_container_width=True):
                        st.session_state["selected_history_id"] = rec.id
                        navigate_to("Analysis History")
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def render_analyze_page() -> None:
    st.markdown(
        """
        <div class="hero-card" style="padding:1.35rem 1.7rem;">
          <div class="hero-eyebrow">AI Evaluation Workspace</div>
          <h1 class="hero-title" style="font-size:1.8rem;">Analyze Resume Against Job Description</h1>
          <p class="hero-desc">
            Upload your resume in PDF format (up to 10 MB) and paste the target job description.
            RESUMATE AI evaluates skill alignment, keyword gaps, and ATS compatibility.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_jd = st.columns([1.0, 1.3], gap="large")

    with col_upload:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">1. Upload Resume (PDF)</div>', unsafe_allow_html=True)
        st.caption(f"Accepted format: Text-based PDF (.pdf) • Maximum size: {MAX_PDF_SIZE_MB} MB • OCR not supported")

        uploaded_file = st.file_uploader(
            "Drag and drop or browse your PDF resume",
            type=["pdf"],
            accept_multiple_files=False,
            key=f"resume_uploader_{st.session_state['uploader_reset_key']}",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            sig = f"{uploaded_file.name}:{len(file_bytes)}"
            if st.session_state["uploaded_file_sig"] != sig:
                try:
                    extracted = extract_resume_from_pdf(uploaded_file.name, file_bytes)
                    st.session_state["extracted_resume"] = extracted
                    st.session_state["uploaded_file_sig"] = sig
                except PDFExtractionError as exc:
                    st.session_state["extracted_resume"] = None
                    st.session_state["uploaded_file_sig"] = None
                    st.error(str(exc))

        resume_data: PDFExtractionResult | None = st.session_state.get("extracted_resume")
        if resume_data is not None:
            size_kb = resume_data.file_size_bytes / 1024
            st.markdown(
                f"""
                <div class="file-meta-pill">
                  📄 {esc(resume_data.filename)} &middot; {resume_data.page_count} page(s) &middot; {resume_data.word_count} words &middot; {size_kb:.1f} KB
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Preview Extracted Resume Text", expanded=False):
                st.text_area(
                    "Extracted Text",
                    value=resume_data.text,
                    height=220,
                    disabled=True,
                    label_visibility="collapsed",
                    key="extracted_text_preview",
                )
        else:
            st.markdown(
                '<div style="color:#94A3B8; font-size:0.86rem; margin-top:0.4rem;">No valid resume PDF loaded yet.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_jd:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">2. Paste Job Description</div>', unsafe_allow_html=True)
        jd_text = st.text_area(
            "Target Job Description",
            key="job_description_input",
            height=245,
            placeholder=(
                "Paste the full job description here (responsibilities, required skills, "
                "preferred qualifications, tools, and experience)..."
            ),
            label_visibility="collapsed",
        )
        jd_words = len(jd_text.split()) if jd_text.strip() else 0
        jd_chars = len(jd_text.strip())
        st.markdown(
            f'<div class="word-counter">Word count: <strong>{jd_words}</strong> words &middot; {jd_chars} characters</div>',
            unsafe_allow_html=True,
        )

        btn_c1, btn_c2 = st.columns([1.6, 1.0])
        with btn_c1:
            analyze_clicked = st.button(
                "🔍 Analyze Resume",
                type="primary",
                use_container_width=True,
                key="btn_run_analysis",
            )
        with btn_c2:
            clear_clicked = st.button(
                "🗑️ Clear",
                use_container_width=True,
                key="btn_clear_inputs",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if clear_clicked:
        st.session_state["job_description_input"] = ""
        st.session_state["current_analysis"] = None
        st.session_state["current_filename"] = ""
        st.session_state["current_analyzed_at"] = ""
        st.rerun()

    if analyze_clicked:
        resume_data = st.session_state.get("extracted_resume")
        if resume_data is None:
            st.error("Please upload a valid text-based PDF resume before running the analysis.")
        else:
            try:
                validate_job_description(jd_text)
                with st.spinner("Analyzing resume compatibility with Gemini AI..."):
                    result = analyze_resume_with_gemini(resume_data.text, jd_text)

                save_analysis(
                    filename=resume_data.filename,
                    job_description=jd_text,
                    analysis=result,
                )
                st.session_state["current_analysis"] = result
                st.session_state["current_filename"] = resume_data.filename
                st.session_state["current_analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("Analysis complete! Review your compatibility dashboard below.")
            except AnalysisError as exc:
                st.error(str(exc))

    active_analysis: ResumeAnalysisSchema | None = st.session_state.get("current_analysis")
    if active_analysis is not None:
        render_analysis_results(
            analysis=active_analysis,
            filename=st.session_state.get("current_filename", "resume.pdf"),
            analyzed_at=st.session_state.get("current_analyzed_at", ""),
            key_prefix="live_result",
            show_reanalyze_controls=True,
        )


def render_history_page() -> None:
    st.markdown(
        """
        <div class="hero-card" style="padding:1.35rem 1.7rem;">
          <div class="hero-eyebrow">Local SQLite Persistence</div>
          <h1 class="hero-title" style="font-size:1.8rem;">Analysis History</h1>
          <p class="hero-desc">
            Revisit past ATS evaluations, compare scores across target roles, and export PDF or JSON reports.
            Uploaded resume PDFs and raw resume text are never stored on disk.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    records = get_all_history(limit=100)
    if not records:
        st.markdown(
            """
            <div class="empty-state-card">
              <div class="empty-state-title">Your Analysis History is Empty</div>
              <p class="empty-state-desc">
                Once you analyze a resume against a job description, your compatibility report will be saved here for quick review and export.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("⚡ Analyze Your First Resume", type="primary", key="hist_empty_cta"):
            navigate_to("Analyze Resume")
            st.rerun()
        return

    top_l, top_r = st.columns([2.4, 1.0])
    with top_l:
        search_q = st.text_input(
            "Filter history",
            placeholder="Search by resume filename or job title snippet...",
            label_visibility="collapsed",
            key="history_search_query",
        ).strip().lower()
    with top_r:
        if not st.session_state.get("confirm_clear_history"):
            if st.button("🗑️ Clear All History", use_container_width=True, key="hist_clear_init"):
                st.session_state["confirm_clear_history"] = True
                st.rerun()
        else:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Confirm", type="primary", use_container_width=True, key="hist_clear_yes"):
                    clear_history()
                    st.session_state["confirm_clear_history"] = False
                    st.session_state["selected_history_id"] = None
                    st.rerun()
            with c2:
                if st.button("Cancel", use_container_width=True, key="hist_clear_no"):
                    st.session_state["confirm_clear_history"] = False
                    st.rerun()

    filtered = [
        r for r in records
        if not search_q or search_q in r.filename.lower() or search_q in r.job_preview.lower()
    ]

    if not filtered:
        st.info("No matching analysis records found for your search filter.")
        return

    selected_id = st.session_state.get("selected_history_id")

    for rec in filtered:
        score_color = (
            "#10B981" if rec.ats_score >= 85
            else "#3B82F6" if rec.ats_score >= 70
            else "#F59E0B" if rec.ats_score >= 50
            else "#EF4444"
        )
        st.markdown('<div class="glass-card" style="padding:1rem 1.2rem;">', unsafe_allow_html=True)
        c_info, c_view, c_pdf, c_del = st.columns([3.0, 0.9, 0.9, 0.7])
        with c_info:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; flex-wrap:wrap; gap:0.65rem;">
                  <span style="font-family:'JetBrains Mono', monospace; font-weight:700; font-size:1.05rem; color:{score_color};">
                    {rec.ats_score}/100
                  </span>
                  <span style="font-weight:700; color:#F8FAFC; font-size:0.98rem;">{esc(rec.filename)}</span>
                  <span style="font-size:0.78rem; padding:2px 8px; border-radius:8px; border:1px solid {score_color}; color:{score_color};">
                    {esc(rec.match_level)}
                  </span>
                </div>
                <div style="color:#CBD5E1; font-size:0.86rem; margin-top:4px;">{esc(rec.job_preview)}</div>
                <div style="color:#64748B; font-size:0.78rem; margin-top:2px;">
                  {esc(rec.created_at)} &middot; Keyword Coverage: {rec.keyword_coverage:.1f}%
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_view:
            btn_label = "Close" if selected_id == rec.id else "View Report"
            if st.button(btn_label, key=f"hist_view_{rec.id}", use_container_width=True):
                st.session_state["selected_history_id"] = None if selected_id == rec.id else rec.id
                st.rerun()
        with c_pdf:
            try:
                pdf_bytes = generate_analysis_pdf_report(rec.analysis, rec.filename, rec.created_at)
                st.download_button(
                    "PDF",
                    data=pdf_bytes,
                    file_name=f"RESUMATE_Report_{rec.id}_{rec.filename.rsplit('.', 1)[0]}.pdf",
                    mime="application/pdf",
                    key=f"hist_pdf_{rec.id}",
                    use_container_width=True,
                )
            except ReportGenerationError:
                st.caption("PDF N/A")
        with c_del:
            if st.button("Delete", key=f"hist_del_{rec.id}", use_container_width=True):
                delete_analysis(rec.id)
                if st.session_state.get("selected_history_id") == rec.id:
                    st.session_state["selected_history_id"] = None
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if selected_id == rec.id:
            render_analysis_results(
                analysis=rec.analysis,
                filename=rec.filename,
                analyzed_at=rec.created_at,
                key_prefix=f"hist_detail_{rec.id}",
                show_reanalyze_controls=False,
            )


def render_tips_page() -> None:
    st.markdown(
        """
        <div class="hero-card" style="padding:1.35rem 1.7rem;">
          <div class="hero-eyebrow">ATS Optimization Playbook</div>
          <h1 class="hero-title" style="font-size:1.8rem;">Resume Improvement Tips</h1>
          <p class="hero-desc">
            Practical, recruiter-backed strategies to make your resume both machine-readable for Applicant Tracking Systems and compelling to hiring managers.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["ATS Formatting", "Keyword Alignment", "Quantifying Impact", "Common Mistakes"]
    )

    with tab1:
        st.markdown(
            """
            <div class="glass-card">
              <div class="section-header">📐 Clean, Parser-Friendly Structure</div>
              <div class="strength-item"><strong>Use Standard Section Headings:</strong> Stick to clear titles like <em>Professional Summary</em>, <em>Experience</em>, <em>Projects</em>, <em>Skills</em>, and <em>Education</em> so ATS parsers map every section accurately.</div>
              <div class="strength-item"><strong>Prefer Single-Column Layouts:</strong> Multi-column tables, text boxes, and floating sidebars can scramble reading order in legacy ATS parsers.</div>
              <div class="strength-item"><strong>Export Selectable Text PDFs:</strong> Always export directly from Word, Google Docs, or LaTeX—never scan printed resumes into image-only PDFs.</div>
              <div class="strength-item"><strong>Avoid Decorative Icons &amp; Skill Bars:</strong> Progress bars (e.g., "Python: 4/5") cannot be parsed reliably by ATS software; list proficiency clearly in text.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown(
            """
            <div class="glass-card">
              <div class="section-header">🎯 Natural Keyword Integration</div>
              <div class="strength-item"><strong>Mirror Exact Role Terminology:</strong> If a posting specifies <em>PostgreSQL</em>, <em>FastAPI</em>, or <em>CI/CD</em>, include those exact terms where you genuinely have experience rather than vague umbrellas.</div>
              <div class="strength-item"><strong>Include Acronyms + Full Names:</strong> Writing <em>Natural Language Processing (NLP)</em> or <em>Amazon Web Services (AWS)</em> covers both search variations.</div>
              <div class="strength-item"><strong>Contextualize Skills in Bullets:</strong> Don't just dump keywords in a skills list—prove how you applied them in real work or capstone projects.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown(
            """
            <div class="glass-card">
              <div class="section-header">📊 Quantify Achievements (XYZ Formula)</div>
              <div class="strength-item"><strong>Formula:</strong> <em>Accomplished [X] as measured by [Y], by doing [Z].</em></div>
              <div class="strength-item"><strong>Weak Bullet:</strong> "Worked on backend APIs and improved database performance."</div>
              <div class="strength-item"><strong>Strong Bullet:</strong> "Reduced API p95 latency by 42% (from 480ms to 278ms) across 120K daily requests by indexing PostgreSQL queries and adding Redis caching."</div>
              <div class="strength-item"><strong>For Students &amp; Fresh Grads:</strong> Quantify dataset sizes, test coverage, latency, user counts, team size, or hackathon/course benchmarks.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab4:
        st.markdown(
            """
            <div class="glass-card">
              <div class="section-header">🚫 Pitfalls to Avoid</div>
              <div class="missing-skill-card">
                <div class="missing-skill-title">Keyword Stuffing or Hidden White Text</div>
                <div class="missing-skill-reason">Modern ATS platforms flag unnatural repetition and recruiters immediately spot unreadable text dumps.</div>
              </div>
              <div class="missing-skill-card">
                <div class="missing-skill-title">Generic One-Size-Fits-All Summaries</div>
                <div class="missing-skill-reason">Sending an un-tailored summary misses the opportunity to spotlight your most relevant qualifications in the first 6 seconds of review.</div>
              </div>
              <div class="missing-skill-card">
                <div class="missing-skill-title">Fabricating Skills You Cannot Defend</div>
                <div class="missing-skill-reason">Only list tools and skills you can confidently discuss in technical interviews. Use gap analysis to guide real upskilling.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about_page() -> None:
    st.markdown(
        f"""
        <div class="hero-card" style="padding:1.35rem 1.7rem;">
          <div class="hero-eyebrow">Architecture &amp; Methodology</div>
          <h1 class="hero-title" style="font-size:1.8rem;">About {esc(APP_NAME)}</h1>
          <p class="hero-desc">
            {esc(APP_NAME)} helps students, graduates, and experienced professionals understand how well their resume aligns with specific job descriptions—before submitting applications.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            """
            <div class="glass-card">
              <div class="section-header">⚙️ How It Works</div>
              <div class="strength-item"><strong>1. In-Memory PDF Parsing:</strong> Validates file headers, enforces size limits (10 MB), detects encrypted or image-only PDFs, and extracts clean text via <code>pypdf</code>.</div>
              <div class="strength-item"><strong>2. Structured Gemini AI Evaluation:</strong> Evaluates resume content against target job requirements inside untrusted-input isolation boundaries, returning strict schema-validated JSON.</div>
              <div class="strength-item"><strong>3. Deterministic Analytics &amp; Export:</strong> Computes coverage breakdowns, renders Plotly visualizations, and generates downloadable PDF &amp; JSON reports.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="glass-card">
              <div class="section-header">🛡️ Security &amp; Privacy Commitments</div>
              <div class="strength-item"><strong>No Permanent Resume Storage:</strong> Uploaded PDF files and extracted full resume texts stay strictly in session memory and are never written to disk or SQLite.</div>
              <div class="strength-item"><strong>Prompt-Injection Defense:</strong> Resume and job description texts are treated strictly as untrusted data blocks so embedded instructions cannot manipulate scoring.</div>
              <div class="strength-item"><strong>Transparent Compatibility Scoring:</strong> Scores are AI-estimated compatibility benchmarks (0–100) designed to guide genuine resume improvements.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="glass-card">
          <div class="section-header">🧱 Technology Stack</div>
          <div class="chip-grid">
            <span class="skill-chip-purple">Streamlit</span>
            <span class="skill-chip-blue">Python 3.11</span>
            <span class="skill-chip-green">Google Gemini API (google-genai)</span>
            <span class="skill-chip-blue">pypdf</span>
            <span class="skill-chip-purple">Plotly</span>
            <span class="skill-chip-green">ReportLab PDF</span>
            <span class="skill-chip-blue">Pydantic Validation</span>
            <span class="skill-chip-purple">SQLite Local History</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    init_db()
    load_custom_css()
    init_session_state()

    active_page = render_sidebar()

    if active_page == "Dashboard":
        render_dashboard_page()
    elif active_page == "Analyze Resume":
        render_analyze_page()
    elif active_page == "Analysis History":
        render_history_page()
    elif active_page == "Resume Improvement Tips":
        render_tips_page()
    elif active_page == "About RESUMATE AI":
        render_about_page()


if __name__ == "__main__":
    main()
