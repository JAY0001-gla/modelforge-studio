"""Visual identity for the app: CSS, Plotly template and small HTML components.

Design notes
------------
* One signature moment: the atmospheric hero panel. Its colour glow drifts slowly
  using GPU-composited transforms (no repaint per frame), so it stays smooth.
* Everything else is deliberately quiet: flat panels, hairline borders, no
  per-card entrance animations. Motion is reserved for responding to an action
  (the meter filling after a forecast, button sheen on hover).
* Motion is switched off for people who prefer reduced motion.
"""
from __future__ import annotations

import html as _html
import re

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
INK = "#0C1E3A"       # deep atmosphere navy: headings, tooltips
TEXT = "#1F2A3C"
MUTED = "#5B6B82"
LINE = "#DCE4EE"
PAGE = "#F4F7FB"
COBALT = "#2A5BD7"    # primary
COBALT_DEEP = "#12306F"
TEAL = "#0E9AA7"
SUN = "#F2A03D"
ROSE = "#D6455D"
GREEN = "#139A6B"
VIOLET = "#6D5BD0"

PALETTE = [COBALT, TEAL, SUN, ROSE, VIOLET, GREEN]
SEQUENTIAL = [[0.0, "#EAF1FD"], [0.5, "#6E97E8"], [1.0, COBALT_DEEP]]
DIVERGING = [[0.0, ROSE], [0.5, "#F7F9FC"], [1.0, COBALT]]
FONT = "Manrope, 'Segoe UI', system-ui, -apple-system, sans-serif"

TONES = {"ok": GREEN, "info": COBALT, "caution": "#C47A12", "alert": ROSE}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _st_version() -> tuple[int, int]:
    try:
        major, minor = st.__version__.split(".")[:2]
        return int(major), int("".join(ch for ch in minor if ch.isdigit()) or 0)
    except Exception:
        return (0, 0)


# Streamlit 1.50+ takes width="stretch" and deprecates use_container_width;
# older versions only know use_container_width (and dataframe's `width` is an int there).
_NEW_WIDTH_API = _st_version() >= (1, 50)


def stretch(fn=None) -> dict:
    """Kwarg that makes a widget fill its container, for whichever Streamlit is installed."""
    return {"width": "stretch"} if _NEW_WIDTH_API else {"use_container_width": True}


def content() -> dict:
    """Kwarg that makes a widget keep its own (figure) width instead of stretching."""
    return {"width": "content"} if _NEW_WIDTH_API else {"use_container_width": False}


def esc(value) -> str:
    return _html.escape(str(value), quote=True)


def _collapse(markup: str) -> str:
    """Strip indentation/newlines so Markdown never mistakes HTML for a code block."""
    return "".join(line.strip() for line in markup.strip().splitlines())


def render(markup: str) -> None:
    st.markdown(_collapse(markup), unsafe_allow_html=True)


def lerp_color(stops: list[str], t: float) -> str:
    """Linear interpolation across a list of hex colours; t in [0, 1]."""
    t = min(max(float(t), 0.0), 1.0)
    pos = t * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    f = pos - i
    a = [int(stops[i][k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(stops[i + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * f):02X}" for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
CSS = """
:root{
  --ink:#0C1E3A; --text:#1F2A3C; --muted:#5B6B82; --line:#DCE4EE; --page:#F4F7FB;
  --cobalt:#2A5BD7; --teal:#0E9AA7; --sun:#F2A03D; --rose:#D6455D; --green:#139A6B;
  --ease:cubic-bezier(.22,1,.36,1);
}

/* ---- Typography (system stack: no render-blocking web-font download) ---- */
.stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp input,
.stApp textarea, .stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"] * {
  font-family: 'Manrope', 'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif;
}
.stApp code, .stApp pre, .stApp kbd, .stApp [data-testid="stCode"] * {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace !important;
}
[data-testid="stIconMaterial"], .material-symbols-rounded {
  font-family: 'Material Symbols Rounded' !important;
}
.stApp { color: var(--text); font-variant-numeric: tabular-nums; }
h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.022em; }
h2 { font-size: 1.35rem; font-weight: 800; margin-top: .6rem; }
h3 { font-size: 1.1rem; font-weight: 700; }

/* ---- Page canvas ------------------------------------------------------ */
.stApp {
  background:
    radial-gradient(1100px 520px at 8% -12%, rgba(42,91,215,.08), transparent 70%),
    radial-gradient(900px 480px at 100% 0%, rgba(14,154,167,.07), transparent 70%),
    var(--page);
}
/* Streamlit's header bar is fixed and ~3.75rem tall: keep content clear of it. */
.block-container { padding-top: 4.75rem; padding-bottom: 4rem; max-width: 1240px; }
header[data-testid="stHeader"] { background: rgba(244,247,251,.86); }
#MainMenu, footer { visibility: hidden; }

/* While a rerun is in flight Streamlit dims stale elements for ~1.5s; make it quick and subtle. */
[data-stale="true"] { opacity: .6 !important; transition: opacity .12s linear !important; }

/* ---- Sidebar ---------------------------------------------------------- */
section[data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--line); }
.brand { padding: 2px 0 14px; }
.brand b { display:block; font-size: 1.12rem; font-weight: 800; color: var(--ink); letter-spacing: -.02em; }
.brand span { font-size: .82rem; color: var(--muted); }
.side-note { font-size: .82rem; color: var(--muted); line-height: 1.55; border-top: 1px solid var(--line);
  padding-top: 12px; margin-top: 8px; overflow-wrap: anywhere; }
.side-note b { color: var(--ink); font-weight: 700; }

/* ---- Page header ------------------------------------------------------ */
.ph { margin: 0 0 .8rem; }
.ph h1 { font-size: 2.05rem; font-weight: 800; letter-spacing: -.035em; margin: 0 0 .25rem; padding: 0; line-height: 1.15; overflow-wrap: anywhere; }
.ph p { margin: 0; color: var(--muted); font-size: 1.02rem; max-width: 62ch; line-height: 1.55; }

/* ---- Hero: the one animated moment ------------------------------------
   Glows are radial gradients (no blur filter), moved with transform only, so the
   browser animates them on the GPU without repainting the text above them. */
.hero {
  position: relative; overflow: hidden; border-radius: 22px; color: #fff;
  padding: 32px 40px 28px; margin: 0 0 .6rem; isolation: isolate;
  background: linear-gradient(135deg, var(--b1), var(--b2));
  box-shadow: 0 18px 40px -22px var(--b1); contain: paint;
}
.hero::before, .hero::after {
  content: ""; position: absolute; z-index: 0; border-radius: 50%; pointer-events: none;
  width: 72%; aspect-ratio: 1; opacity: .6; will-change: transform;
}
.hero::before { background: radial-gradient(circle, var(--g1) 0%, transparent 66%); top: -75%; left: -12%; animation: driftA 14s ease-in-out infinite alternate; }
.hero::after  { background: radial-gradient(circle, var(--g2) 0%, transparent 66%); bottom: -100%; right: -12%; animation: driftB 18s ease-in-out infinite alternate; }
@keyframes driftA {
  0%   { transform: translate3d(0,0,0) scale(1); }
  100% { transform: translate3d(38%,22%,0) scale(1.15); }
}
@keyframes driftB {
  0%   { transform: translate3d(0,0,0) scale(1.05); }
  100% { transform: translate3d(-34%,-20%,0) scale(.95); }
}
.hero-in { position: relative; z-index: 1; text-shadow: 0 1px 14px rgba(0,0,0,.22); }
.hero-head { font-size: .98rem; font-weight: 600; opacity: .9; margin-bottom: 6px; }
.hero-title { font-size: clamp(1.4rem, 3vw, 2rem); font-weight: 800; letter-spacing: -.03em; line-height: 1.2; max-width: 26ch; overflow-wrap: anywhere; }
.hero-val { font-size: clamp(2.8rem, 7vw, 4.4rem); font-weight: 800; letter-spacing: -.05em; line-height: 1.05; overflow-wrap: anywhere; }
.hero-val.txt { font-size: clamp(1.7rem, 4.2vw, 2.7rem); letter-spacing: -.03em; line-height: 1.15; max-width: 28ch; }
.hero-val small { font-size: clamp(1.1rem, 2.4vw, 1.7rem); font-weight: 700; margin-left: 8px; opacity: .85; letter-spacing: -.02em; }
.hero-sub { margin-top: 10px; font-size: 1.02rem; opacity: .94; max-width: 60ch; line-height: 1.5; }
.meter { margin-top: 22px; height: 8px; border-radius: 99px; background: rgba(255,255,255,.22); overflow: hidden; }
.meter > span {
  display: block; height: 100%; width: var(--w); border-radius: inherit;
  background: linear-gradient(90deg, rgba(255,255,255,.55), #fff);
  transform-origin: left center; animation: fill .7s var(--ease) both;
}
@keyframes fill { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.meter-scale { display: flex; justify-content: space-between; margin-top: 8px; font-size: .78rem; opacity: .85; }
.meter-scale i { font-style: normal; }

.s-idle    { --b1:#0C1E3A; --b2:#14407A; --g1:#2A5BD7; --g2:#0E9AA7; }
.s-cool    { --b1:#0B2A52; --b2:#0F5A8E; --g1:#38A3F5; --g2:#4FD6C4; }
.s-mild    { --b1:#0B3A40; --b2:#0F6A62; --g1:#19B58C; --g2:#2A8BD7; }
.s-warm    { --b1:#4B2C0E; --b2:#8A5412; --g1:#F2A03D; --g2:#E4683A; }
.s-hot     { --b1:#561C16; --b2:#8E2C1E; --g1:#F26B3A; --g2:#D6455D; }
.s-extreme { --b1:#44102A; --b2:#7A1637; --g1:#E5385F; --g2:#F2A03D; }

/* ---- Stat strip (one panel, divided cells) ---------------------------- */
.strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px;
  background: var(--line); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; margin: 0 0 .4rem; }
.strip > div { background: #fff; padding: 16px 20px; min-width: 0; }
.strip .k { font-size: .84rem; font-weight: 600; color: var(--muted); }
.strip .v { font-size: 1.75rem; font-weight: 800; letter-spacing: -.03em; color: var(--ink); line-height: 1.25; overflow-wrap: anywhere; }
.strip .n { font-size: .8rem; color: var(--muted); overflow-wrap: anywhere; }

/* ---- Guidance rows ---------------------------------------------------- */
.rows { background: #fff; border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
.rows > div { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 4px 16px;
  padding: 14px 20px; border-top: 1px solid var(--line); }
.rows > div:first-child { border-top: 0; }
.rows .d { font-weight: 600; color: var(--ink); }
.rows .s { font-weight: 700; display: inline-flex; align-items: center; gap: 8px; text-align: right; }
.rows .s::before { content: ""; flex: none; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }

/* ---- Callout ---------------------------------------------------------- */
.note { border-left: 3px solid var(--cobalt); background: #fff; border-radius: 0 12px 12px 0;
  padding: 12px 18px; color: var(--text); font-size: .95rem; line-height: 1.55; margin: 0;
  border-top: 1px solid var(--line); border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.note b { color: var(--ink); }

/* ---- Streamlit widgets ------------------------------------------------ */
[data-testid^="stBaseButton"] { height: auto; min-height: 2.75rem; }
[data-testid^="stBaseButton"] p { white-space: normal; overflow-wrap: anywhere; line-height: 1.25; text-align: center; }
[data-testid="stBaseButton-primary"], button[kind="primary"] {
  border: 0; border-radius: 11px; font-weight: 700; padding: .62rem 1.2rem; color: #fff;
  background: linear-gradient(115deg, #2A5BD7 0%, #1F7FD0 48%, #0E9AA7 100%);
  background-size: 190% 100%; background-position: 0% 50%;
  box-shadow: 0 1px 2px rgba(12,30,58,.2);
  transition: background-position .35s var(--ease), box-shadow .2s ease, transform .2s var(--ease);
}
[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover {
  background-position: 100% 50%; color: #fff; transform: translateY(-1px);
  box-shadow: 0 10px 22px -10px rgba(42,91,215,.65);
}
[data-testid="stBaseButton-primary"]:active, button[kind="primary"]:active { transform: translateY(0); }
[data-testid="stBaseButton-primary"] p, button[kind="primary"] p { color: #fff; }

[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
  border-radius: 11px; font-weight: 600; background: #fff; border: 1px solid #C5D1E2; color: var(--ink);
  transition: border-color .18s ease, background-color .18s ease, color .18s ease;
}
[data-testid="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover {
  border-color: var(--cobalt); color: var(--cobalt); background: #F3F7FF;
}
button:focus-visible { outline: 2px solid var(--cobalt) !important; outline-offset: 2px; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); overflow-x: auto; }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 14px; color: var(--muted); font-weight: 600; transition: color .15s ease; white-space: nowrap; }
.stTabs [data-baseweb="tab"]:hover { color: var(--ink); }
.stTabs [aria-selected="true"] { color: var(--cobalt); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--cobalt); height: 2px; transition: left .22s var(--ease), width .22s var(--ease); }

[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px; }
[data-testid="stExpander"] { background: #fff; border: 1px solid var(--line); border-radius: 12px; }
[data-testid="stFileUploaderDropzone"] {
  border: 1.5px dashed #B6C5DB; border-radius: 12px; background: #F8FAFD;
  transition: border-color .18s ease, background-color .18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--cobalt); background: #F1F6FF; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
[data-testid="stPlotlyChart"] { max-width: 100%; overflow-x: auto; overflow-y: hidden; }
[data-baseweb="input"], [data-baseweb="select"] > div { border-radius: 10px; }
hr { border-color: var(--line); }

@media (max-width: 720px) {
  .block-container { padding: 4.25rem 1rem 3rem; }
  .hero { padding: 24px 20px 22px; border-radius: 18px; }
  .ph h1 { font-size: 1.6rem; }
  .strip > div { padding: 12px 14px; }
  .strip .v { font-size: 1.45rem; }
  .rows > div { padding: 12px 14px; }
  .meter-scale { font-size: .7rem; }
}
@media (prefers-reduced-motion: reduce) {
  .hero::before, .hero::after, .meter > span { animation: none !important; }
  [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] { transition: none !important; }
}
"""


def inject_css() -> None:
    css = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
    css = re.sub(r"\s+", " ", css)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def register_plotly() -> None:
    """Make every Plotly figure inherit the app's look and a gentle transition."""
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, color=TEXT, size=13),
        colorway=PALETTE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=16, t=44, b=12),
        xaxis=dict(gridcolor="#E8EEF5", linecolor=LINE, zeroline=False, tickcolor=LINE,
                   automargin=True, title=dict(standoff=10)),
        yaxis=dict(gridcolor="#E8EEF5", linecolor=LINE, zeroline=False, tickcolor=LINE,
                   automargin=True, title=dict(standoff=10)),
        colorscale=dict(sequential=SEQUENTIAL, diverging=DIVERGING),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(color="#fff", family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        transition=dict(duration=250, easing="cubic-in-out"),
    )
    pio.templates["modelforge"] = t
    pio.templates.default = "modelforge"


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def page_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    render(f'<div class="ph"><h1>{esc(title)}</h1>{sub}</div>')


def hero(*, state: str = "idle", head: str = "", title: str = "", value: str | None = None,
         unit: str = "", sub: str = "", meter: tuple[float, list[str]] | None = None) -> None:
    parts = [f'<div class="hero s-{esc(state)}"><div class="hero-in">']
    if head:
        parts.append(f'<div class="hero-head">{esc(head)}</div>')
    if title:
        parts.append(f'<div class="hero-title">{esc(title)}</div>')
    if value is not None:
        # Long text answers (class labels) get a smaller size so they never overflow the panel.
        is_text = re.fullmatch(r"[-+]?[\d.,]+%?", str(value).strip()) is None
        cls = "hero-val txt" if is_text else "hero-val"
        small = f"<small>{esc(unit)}</small>" if unit else ""
        parts.append(f'<div class="{cls}">{esc(value)}{small}</div>')
    if sub:
        parts.append(f'<div class="hero-sub">{esc(sub)}</div>')
    if meter:
        pct, ticks = meter
        pct = min(max(pct, 0.0), 100.0)
        scale = "".join(f"<i>{esc(t)}</i>" for t in ticks)
        parts.append(f'<div class="meter" style="--w:{pct:.1f}%"><span></span></div>'
                     f'<div class="meter-scale">{scale}</div>')
    parts.append("</div></div>")
    render("".join(parts))


def strip(items: list[tuple[str, str, str]]) -> None:
    """A single panel of divided stat cells: (label, value, note)."""
    cells = "".join(
        f'<div><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>'
        f'<div class="n">{esc(n) if n else "&nbsp;"}</div></div>'
        for k, v, n in items
    )
    render(f'<div class="strip">{cells}</div>')


def rows(items: list[tuple[str, str, str]]) -> None:
    """Guidance list: (domain, status text, tone in ok/info/caution/alert)."""
    body = "".join(
        f'<div><span class="d">{esc(d)}</span>'
        f'<span class="s" style="color:{TONES.get(tone, COBALT)}">{esc(s)}</span></div>'
        for d, s, tone in items
    )
    render(f'<div class="rows">{body}</div>')


def note(text: str) -> None:
    """Callout. Supports <b> in `text`; caller must pass trusted/escaped content."""
    render(f'<div class="note">{text}</div>')
