"""Shared dark, cinematic LumaForge theme."""
from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"], [data-testid="stMetricValue"], button, input, textarea,
        select, label, p, h1, h2, h3, h4, li {
            font-family: "Times New Roman", Times, serif !important;
        }
        [data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined,
        [class*="material-symbols"] {
            font-family: "Material Symbols Rounded" !important;
            font-feature-settings: "liga"; font-weight: normal; font-style: normal;
            letter-spacing: normal; text-transform: none; white-space: nowrap; direction: ltr;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(ellipse at 50% -20%, #20272b 0%, #111619 48%, #0b0f11 100%);
            color: #eff2ed;
        }
        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1560px; padding: 1.35rem clamp(1rem, 3vw, 2.6rem) 3rem;
        }
        [data-testid="stSidebar"] {
            background: #111719; border-right: 1px solid #283234;
            min-width: 300px !important; max-width: 340px !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #d5ddd7; }
        h1, h2, h3 { color: #f3f5ef !important; letter-spacing: .01em; }
        h1 { font-size: clamp(1.8rem, 3vw, 2.5rem) !important; line-height: 1.1 !important; }
        h2 { font-size: clamp(1.35rem, 2.2vw, 1.8rem) !important; }
        h3 { font-size: 1.18rem !important; }
        p, label, [data-testid="stCaptionContainer"] { color: #bec9c4; }
        [data-testid="stTabs"] button[role="tab"] { color: #b8c4bf; }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #c3f27e; border-bottom-color: #c3f27e;
        }
        [data-testid="stMetric"] {
            background: #151d20; border: 1px solid #2a3638; border-radius: 12px;
            padding: 11px 14px; box-shadow: 0 5px 20px rgba(0,0,0,.14);
        }
        [data-testid="stMetricLabel"] { color: #aab8b2 !important; }
        [data-testid="stMetricValue"] { color: #f0f4ed !important; }
        .stButton > button, .stDownloadButton > button {
            background: #c3f27e; color: #131911; border: 1px solid #c3f27e;
            border-radius: 9px; font-weight: 700; box-shadow: 0 0 0 rgba(195,242,126,0);
            transition: transform .16s ease, box-shadow .16s ease, background .16s ease;
        }
        .stButton > button p, .stButton > button span,
        .stDownloadButton > button p, .stDownloadButton > button span { color: inherit !important; }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #d2ff91; border-color: #d2ff91; color: #10160e;
            transform: translateY(-1px); box-shadow: 0 0 22px rgba(195,242,126,.18);
        }
        .stButton > button:disabled, .stDownloadButton > button:disabled {
            background: #354039 !important; border-color: #46524a !important;
            color: #b7c1b9 !important; opacity: 1 !important; box-shadow: none !important;
        }
        [data-testid="stFileUploader"] section {
            background: #151d20; border: 1px solid #344143; border-radius: 11px;
        }
        [data-testid="stFileUploader"] section button {
            background: #1d272a; color: #d9e0db; border: 1px solid #3c494a;
            border-radius: 8px; box-shadow: none;
        }
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"],
        [data-testid="stTextArea"] textarea { background: #151d20; }
        [data-testid="stAlert"] { background: #171f21; border: 1px solid #344143; border-radius: 10px; }
        [data-testid="stImage"] img { border-radius: 9px; }
        div[data-testid="stExpander"] { border: 1px solid #2b3739; border-radius: 10px; background: #131a1c; }
        [data-testid="stProgressBar"] > div > div { background: #c3f27e; }
        .lf-brand { display:flex; align-items:center; gap:.75rem; margin:.1rem 0 .3rem; }
        .lf-mark { width:34px; height:34px; display:grid; place-items:center; border-radius:11px;
            color:#10160e; background:linear-gradient(135deg,#d5ff91,#a8d565); font-weight:800; }
        .lf-eyebrow { color:#c3f27e; text-transform:uppercase; letter-spacing:.17em; font-size:.76rem; }
        .lf-card { background:#141b1e; border:1px solid #2c383a; border-radius:14px; padding:14px 16px; }
        .lf-selected { border-color:#c3f27e; box-shadow:0 0 0 1px rgba(195,242,126,.26),0 0 25px rgba(195,242,126,.08); }
        .lf-pipeline { display:flex; align-items:center; gap:8px; overflow-x:auto; padding:12px 2px 16px; }
        .lf-stage { display:flex; align-items:center; gap:7px; white-space:nowrap; color:#e2e9e3;
            background:#172022; border:1px solid #344144; border-radius:999px; padding:8px 12px; font-size:.92rem; }
        .lf-stage.done { border-color:#63824d; background:#1b2920; }
        .lf-dot { width:8px;height:8px;border-radius:50%;background:#c3f27e;box-shadow:0 0 10px rgba(195,242,126,.35); }
        .lf-arrow { color:#82938d; }
        @media(max-width:800px) {
            [data-testid="stSidebar"] { min-width:260px !important; max-width:300px !important; }
            [data-testid="stAppViewContainer"] > .main .block-container { padding:.9rem .8rem 2rem; }
            .lf-pipeline { gap:5px; padding:8px 0 12px; }
            .lf-stage { padding:7px 9px; font-size:.84rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
