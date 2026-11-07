"""Shared LumaForge visual theme for Streamlit entry points."""
from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"], [data-testid="stMetricValue"], button, input, textarea,
        select, label, p, h1, h2, h3, h4, li {
            font-family: "Times New Roman", Times, serif !important;
        }
        /* Keep Streamlit's Material Symbols font intact; overriding every span
           turns icon ligatures into visible strings such as keyboard_double_arrow. */
        [data-testid="stIconMaterial"], .material-symbols-rounded,
        .material-symbols-outlined, [class*="material-symbols"] {
            font-family: "Material Symbols Rounded" !important;
            font-feature-settings: "liga";
            font-weight: normal;
            font-style: normal;
            letter-spacing: normal;
            text-transform: none;
            white-space: nowrap;
            direction: ltr;
        }
        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] { min-width: 340px !important; max-width: 360px !important; }
        @media (max-width: 900px) {
            [data-testid="stSidebar"] { min-width: 280px !important; max-width: 320px !important; }
            [data-testid="stAppViewContainer"] > .main .block-container { padding: 1.25rem 1.25rem 2rem; }
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(ellipse at top left, #fffafb 0%, #f7f1f3 52%, #f1e8eb 100%);
            color: #2b1720;
        }
        [data-testid="stSidebar"] { background: #f4eaed; border-right: 1px solid #e1cbd3; }
        h1, h2, h3 { color: #681832 !important; letter-spacing: .01em; }
        h1 { font-size: clamp(2rem, 3vw, 2.8rem) !important; line-height: 1.15 !important; }
        h2 { font-size: clamp(1.55rem, 2.2vw, 2rem) !important; }
        h3 { font-size: 1.35rem !important; }
        [data-testid="stAlert"] {
            border: 1px solid #ddbdc8; border-left: 4px solid #7b1e3d;
            border-radius: 12px; background: #f7edf1; color: #402431;
        }
        [data-testid="stAlert"] p { color: #402431 !important; }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,.82); border: 1px solid #e4ccd4;
            border-radius: 14px; padding: 12px 16px; box-shadow: 0 5px 18px rgba(83,24,47,.07);
        }
        [data-testid="stMetricLabel"] { color: #6f4b59 !important; }
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, #7b1e3d, #54122b); color: #fff;
            border: 1px solid #6e1a36; border-radius: 10px; font-weight: 700;
            box-shadow: 0 4px 12px rgba(91,20,47,.18); transition: transform .15s ease, box-shadow .15s ease;
        }
        /* Streamlit renders button labels inside a paragraph with its own theme
           colour. Set the nested text explicitly or primary buttons can show a
           dark label on the wine background. */
        .stButton > button p, .stButton > button span,
        .stDownloadButton > button p, .stDownloadButton > button span {
            color: inherit !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            color: #fff; border-color: #a04a66; transform: translateY(-1px);
            box-shadow: 0 7px 18px rgba(91,20,47,.25);
        }
        /* Keep disabled controls readable while making their state obvious. */
        .stButton > button:disabled, .stButton > button[aria-disabled="true"],
        .stDownloadButton > button:disabled, .stDownloadButton > button[aria-disabled="true"] {
            background: #ead7dd !important; color: #4b2332 !important;
            border: 1px solid #cbaab6 !important; opacity: 1 !important;
            box-shadow: none !important; transform: none !important; cursor: not-allowed !important;
        }
        .stButton > button:disabled p, .stButton > button[aria-disabled="true"] p,
        .stDownloadButton > button:disabled p, .stDownloadButton > button[aria-disabled="true"] p {
            color: #4b2332 !important; opacity: 1 !important;
        }
        [data-testid="stFileUploader"] section { background: rgba(255,255,255,.82); border: 1px solid #d8bac5; border-radius: 12px; }
        [data-testid="stFileUploader"] section button {
            background: #fffafb; color: #54122b; border: 1px solid #cfaeba;
            border-radius: 9px; box-shadow: none;
        }
        [data-testid="stFileUploader"] section button:hover { color: #54122b; border-color: #7b1e3d; }
        [data-testid="stTabs"] button[role="tab"] { color: #5a3342; }
        [data-testid="stTabs"] button[aria-selected="true"] { color: #741b39; border-bottom-color: #741b39; }
        [data-testid="stImage"] img { border-radius: 8px; }
        div[data-testid="stExpander"] { border: 1px solid #e0c8d0; border-radius: 12px; background: rgba(255,255,255,.56); }
        </style>
        """,
        unsafe_allow_html=True,
    )
