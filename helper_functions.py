"""
utils/helper_functions.py
--------------------------
DemandLens — Premium UI components and chart helpers.
Matches the dark-navy, monospace-label aesthetic from the screenshot.
"""

import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# Theme — single source of truth for all pages
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#0a0d14",
    "surface":   "#0f1520",
    "card":      "#131927",
    "card2":     "#111825",
    "border":    "rgba(255,255,255,0.07)",
    "primary":   "#6e76e5",
    "primary2":  "#5a62d4",
    "accent":    "#3fb950",
    "teal":      "#4fd1c5",
    "purple":    "#b794f4",
    "orange":    "#f6ad55",
    "yellow":    "#f6ad55",
    "secondary": "#f687b3",
    "subtext":   "#7a8499",
    "text":      "#e8edf5",
    "muted":     "#7a8499",
}

CATEGORY_COLORS = [
    "#6e76e5",
    "#4fd1c5",
    "#b794f4",
    "#f6ad55",
    "#63b3ed",
    "#f687b3",
]

PLOTLY_TEMPLATE = "plotly_dark"


# ─────────────────────────────────────────────────────────────────────────────
# App Header — matches the screenshot exactly
# ─────────────────────────────────────────────────────────────────────────────
def render_app_header(
    course_label: str = "24B15CS222 · AI/ML CAPSTONE",
    title: str = "DemandLens",
    subtitle: str = "Retail Revenue Optimizer · Scikit-learn · TensorFlow · PED Analysis",
    r2: str = "R² 0.94",
    lift: str = "LIFT +18.3%",
) -> str:
    return f"""
<div style="
    background:#0f1520;
    border-bottom:1px solid rgba(255,255,255,0.07);
    padding:16px 28px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    margin-bottom:0;
">
    <div>
        <div style="
            font-size:9px;color:#7a8499;letter-spacing:0.18em;
            text-transform:uppercase;font-family:'JetBrains Mono',monospace;margin-bottom:3px;
        ">{course_label}</div>
        <div style="
            font-size:26px;font-weight:800;color:#e8edf5;
            letter-spacing:-0.04em;line-height:1;font-family:'Syne',sans-serif;
        ">{title}</div>
        <div style="font-size:11px;color:#7a8499;margin-top:3px;letter-spacing:0.02em;">
            {subtitle}
        </div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
        <span style="
            display:inline-flex;align-items:center;gap:7px;
            background:#131927;border:1px solid rgba(63,185,80,0.28);
            border-radius:20px;padding:5px 14px;
            font-size:11px;font-weight:600;color:#3fb950;
            font-family:'JetBrains Mono',monospace;letter-spacing:0.06em;
        ">
            <span style="
                width:7px;height:7px;border-radius:50%;
                background:#3fb950;display:inline-block;
                animation:pulse-dot 2s infinite;
            "></span>
            LIVE ENGINE
        </span>
        <span style="
            background:#131927;border:1px solid rgba(167,139,250,0.25);
            border-radius:20px;padding:5px 14px;
            font-size:11px;font-weight:600;color:#b794f4;
            font-family:'JetBrains Mono',monospace;letter-spacing:0.06em;
        ">{r2}</span>
        <span style="
            background:#131927;border:1px solid rgba(63,185,80,0.22);
            border-radius:20px;padding:5px 14px;
            font-size:11px;font-weight:600;color:#3fb950;
            font-family:'JetBrains Mono',monospace;letter-spacing:0.06em;
        ">{lift}</span>
    </div>
</div>
<style>
@keyframes pulse-dot {{
    0%,100% {{ opacity:1; box-shadow:0 0 0 0 rgba(63,185,80,0.4); }}
    50%      {{ opacity:0.7; box-shadow:0 0 0 5px rgba(63,185,80,0); }}
}}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Plotly Dark Chart Style — bg transparent, subtle grid
# ─────────────────────────────────────────────────────────────────────────────
def apply_dark_chart_style(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color=COLORS["muted"], size=11),
        margin=dict(l=12, r=12, t=36, b=12),
        height=height,
        title_font=dict(
            family="JetBrains Mono, monospace",
            size=10,
            color=COLORS["muted"],
        ),
        title_x=0.01,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=10, color=COLORS["muted"]),
        ),
        hoverlabel=dict(
            bgcolor="#0f1520",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(family="JetBrains Mono", size=11, color="#ffffff"),
        ),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(255,255,255,0.04)",
        zeroline=False, showline=False,
        tickfont=dict(color=COLORS["muted"], size=10),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(255,255,255,0.04)",
        zeroline=False, showline=False,
        tickfont=dict(color=COLORS["muted"], size=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI Card — large value card used across pages
# ─────────────────────────────────────────────────────────────────────────────
def kpi_html(label: str, value: str, delta: str = "",
             icon: str = "", color: str = "") -> str:
    value_color = color if color else COLORS["text"]
    label_text  = f"{icon} {label}".strip() if icon else label
    delta_html  = (
        f'<div style="font-size:11px;color:{COLORS["accent"]};margin-top:8px;'
        f'font-family:JetBrains Mono,monospace;font-weight:500;">{delta}</div>'
        if delta else ""
    )
    return (
        f'<div style="background:{COLORS["card"]};border:1px solid {COLORS["border"]};'
        f'border-radius:10px;padding:18px 20px;min-height:110px;'
        f'display:flex;flex-direction:column;justify-content:center;">'
        f'<div style="font-size:9px;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.16em;color:{COLORS["muted"]};'
        f'font-family:JetBrains Mono,monospace;margin-bottom:10px;">{label_text}</div>'
        f'<div style="font-size:24px;font-weight:700;color:{value_color};'
        f'letter-spacing:-0.03em;line-height:1;font-family:Syne,sans-serif;">{value}</div>'
        f'{delta_html}</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Small metric card — for PED INDEX / DEMAND Δ style mini cards
# ─────────────────────────────────────────────────────────────────────────────
def metric_card_html(label: str, value: str, value_color: str = "") -> str:
    col = value_color if value_color else COLORS["text"]
    return (
        f'<div style="background:{COLORS["card"]};border:1px solid {COLORS["border"]};'
        f'border-radius:10px;padding:14px 16px;">'
        f'<div style="font-size:9px;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.16em;color:{COLORS["muted"]};'
        f'font-family:JetBrains Mono,monospace;margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:26px;font-weight:700;color:{col};'
        f'letter-spacing:-0.04em;line-height:1;font-family:Syne,sans-serif;">{value}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Net Revenue Impact card
# ─────────────────────────────────────────────────────────────────────────────
def impact_card_html(value: str = "—", value_color: str = "") -> str:
    col = value_color if value_color else COLORS["text"]
    return (
        f'<div style="background:{COLORS["card"]};border:1px solid {COLORS["border"]};'
        f'border-radius:10px;padding:14px 16px;min-height:68px;">'
        f'<div style="font-size:9px;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.16em;color:{COLORS["primary"]};'
        f'font-family:JetBrains Mono,monospace;margin-bottom:10px;">NET REVENUE IMPACT</div>'
        f'<div style="font-size:22px;font-weight:700;color:{col};'
        f'letter-spacing:-0.03em;font-family:Syne,sans-serif;">{value}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Panel wrapper — the bordered card panel used in left column
# ─────────────────────────────────────────────────────────────────────────────
def panel_open_html(label: str) -> str:
    return (
        f'<div style="background:{COLORS["card"]};border:1px solid {COLORS["border"]};'
        f'border-radius:10px;padding:20px;margin-bottom:14px;">'
        f'<div style="font-size:9px;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.18em;color:{COLORS["muted"]};'
        f'font-family:JetBrains Mono,monospace;margin-bottom:16px;">{label}</div>'
    )


def panel_close_html() -> str:
    return '</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Section header (inline chart title style)
# ─────────────────────────────────────────────────────────────────────────────
def section_header(title: str, subtitle: str = "") -> str:
    sub_html = (
        f'<div style="color:{COLORS["muted"]};font-size:13px;margin-top:4px;">{subtitle}</div>'
        if subtitle else ""
    )
    return (
        f'<div style="margin:20px 0 14px 0;padding-bottom:10px;'
        f'border-bottom:1px solid rgba(255,255,255,0.04);">'
        f'<div style="font-size:10px;font-weight:600;color:{COLORS["muted"]};'
        f'text-transform:uppercase;letter-spacing:0.12em;'
        f'font-family:JetBrains Mono,monospace;">{title}</div>'
        f'{sub_html}</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chart title (displayed above plotly as text, matching screenshot label style)
# ─────────────────────────────────────────────────────────────────────────────
def chart_label_html(text: str) -> str:
    return (
        f'<div style="font-size:9px;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:{COLORS["muted"]};'
        f'font-family:JetBrains Mono,monospace;padding:14px 14px 0 14px;">{text}</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────
def format_currency(val: float) -> str:
    if val >= 1_000_000:
        return f"₹{val / 1_000_000:.1f}M"
    if val >= 1_000:
        return f"₹{val / 1_000:.1f}K"
    return f"₹{val:.2f}"


def format_number(val: float) -> str:
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if val >= 1_000:
        return f"{val / 1_000:.1f}K"
    return f"{val:,.0f}"