from __future__ import annotations

import math
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# SCORETRACK AI
# ============================================================
# A student-first marks prediction and performance tracking app.
# The interface intentionally stays focused on three experiences:
#   1. Predict Marks
#   2. Performance Graph
#   3. History
# ============================================================


APP_TITLE = "ScoreTrack AI"
APP_SUBTITLE = "Your personal AI-powered marks predictor & performance tracker"
DATA_FILE = Path("student_marks_data.csv")
HISTORY_FILE = Path("marks_history.csv")
RANDOM_STATE = 42

FEATURES = [
    "attendance_pct",
    "study_hours_per_day",
    "previous_test_avg",
    "assignment_completion_pct",
    "sleep_hours",
    "class_participation_pct",
    "practice_questions_per_week",
    "screen_time_hours",
    "stress_level",
]

FEATURE_LABELS = {
    "attendance_pct": "Attendance",
    "study_hours_per_day": "Study hours / day",
    "previous_test_avg": "Previous test average",
    "assignment_completion_pct": "Assignment completion",
    "sleep_hours": "Sleep hours",
    "class_participation_pct": "Class participation",
    "practice_questions_per_week": "Practice questions / week",
    "screen_time_hours": "Screen time / day",
    "stress_level": "Stress level",
}

FEATURE_HELP = {
    "attendance_pct": "How regularly you attend classes.",
    "study_hours_per_day": "Average focused study time each day.",
    "previous_test_avg": "Your recent average score out of 100.",
    "assignment_completion_pct": "Percentage of assignments completed.",
    "sleep_hours": "Average sleep you get on a school night.",
    "class_participation_pct": "How actively you participate in class.",
    "practice_questions_per_week": "Questions or problems you solve each week.",
    "screen_time_hours": "Approximate non-study screen time per day.",
    "stress_level": "Self-rated stress from 1 (low) to 10 (high).",
}

DEFAULT_INPUTS = {
    "attendance_pct": 90.0,
    "study_hours_per_day": 3.0,
    "previous_test_avg": 72.0,
    "assignment_completion_pct": 90.0,
    "sleep_hours": 7.0,
    "class_participation_pct": 80.0,
    "practice_questions_per_week": 80.0,
    "screen_time_hours": 3.0,
    "stress_level": 4.0,
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS / VISUAL SYSTEM
# ============================================================


def inject_global_styles() -> None:
    """Inject the complete glassmorphism visual system."""

    st.markdown(
        """
        <style>
        :root {
            --bg-1: #08111f;
            --bg-2: #0d1728;
            --surface: rgba(255,255,255,0.075);
            --surface-strong: rgba(255,255,255,0.11);
            --surface-light: rgba(255,255,255,0.05);
            --border: rgba(255,255,255,0.13);
            --border-strong: rgba(255,255,255,0.19);
            --text: #f8fafc;
            --muted: #9aa7bb;
            --accent: #7c8cff;
            --accent-2: #50e3c2;
            --accent-3: #b37aff;
            --danger: #ff758f;
            --warning: #ffbf69;
            --success: #58d68d;
            --shadow: 0 20px 70px rgba(0, 0, 0, 0.35);
            --radius-lg: 28px;
            --radius-md: 20px;
            --radius-sm: 14px;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;
        }

        .stApp {
            min-height: 100vh;
            background:
                radial-gradient(circle at 15% 10%, rgba(124,140,255,0.18), transparent 28%),
                radial-gradient(circle at 85% 15%, rgba(80,227,194,0.13), transparent 24%),
                radial-gradient(circle at 72% 82%, rgba(179,122,255,0.12), transparent 25%),
                linear-gradient(145deg, var(--bg-1), var(--bg-2));
            color: var(--text);
        }

        .stApp::before,
        .stApp::after {
            content: "";
            position: fixed;
            z-index: -1;
            border-radius: 999px;
            filter: blur(80px);
            opacity: 0.55;
            pointer-events: none;
        }

        .stApp::before {
            width: 320px;
            height: 320px;
            left: -90px;
            top: 120px;
            background: rgba(124, 140, 255, 0.16);
        }

        .stApp::after {
            width: 360px;
            height: 360px;
            right: -100px;
            bottom: 50px;
            background: rgba(80, 227, 194, 0.11);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            visibility: hidden;
            height: 0;
        }

        [data-testid="stDecoration"] {
            display: none;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        .main .block-container {
            padding-top: 2.3rem;
            padding-bottom: 4rem;
            max-width: 1280px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6,15,28,0.87), rgba(10,17,30,0.92));
            border-right: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(22px);
            -webkit-backdrop-filter: blur(22px);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.6rem;
            padding-bottom: 1.5rem;
        }

        /* Keep the Streamlit sidebar toggle available even if the browser
           remembers that the sidebar was previously collapsed. */
        [data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 999999 !important;
        }

        [data-testid="stSidebarCollapsedControl"] button {
            width: 42px !important;
            height: 42px !important;
            border-radius: 12px !important;
            background: rgba(18, 29, 50, 0.92) !important;
            border: 1px solid rgba(255,255,255,0.16) !important;
            box-shadow: 0 12px 30px rgba(0,0,0,0.28) !important;
        }

        /* Permanent app-level navigation: independent of Streamlit sidebar state. */
        [data-testid="stSidebar"] {
            display: none !important;
        }

        [data-testid="stAppViewContainer"] > .main {
            margin-left: 0 !important;
        }

        /* The navigation lives in the first column of the top-level layout.
           Style the column itself so Streamlit's separate elements remain
           inside the visible glass panel. */
        .main .block-container > div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="stColumn"]:first-child {
            position: sticky;
            top: 1.2rem;
            align-self: flex-start;
            min-height: calc(100vh - 2.4rem);
            padding: 1.15rem !important;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(8,17,31,0.94), rgba(10,18,32,0.78));
            border: 1px solid rgba(255,255,255,0.09);
            box-shadow: 0 18px 55px rgba(0,0,0,0.25);
            backdrop-filter: blur(22px);
            -webkit-backdrop-filter: blur(22px);
            box-sizing: border-box;
        }

        .main .block-container > div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="stColumn"]:first-child > div {
            width: 100%;
        }

        @media (max-width: 900px) {
            .main .block-container > div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="stColumn"]:first-child {
                position: relative;
                min-height: auto;
                margin-bottom: 1rem;
            }
        }

        .brand-wrap {
            padding: 0.3rem 0.25rem 1.2rem 0.25rem;
        }

        .brand-pill {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, rgba(124,140,255,0.95), rgba(179,122,255,0.9));
            box-shadow: 0 10px 30px rgba(124,140,255,0.35);
            font-size: 1.3rem;
            margin-bottom: 0.8rem;
        }

        .brand-title {
            font-size: 1.16rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #ffffff;
        }

        .brand-caption {
            color: #8996ab;
            font-size: 0.78rem;
            line-height: 1.4;
            margin-top: 0.28rem;
        }

        .side-note {
            margin-top: 1.2rem;
            padding: 0.95rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            color: #a7b1c4;
            font-size: 0.76rem;
            line-height: 1.55;
        }

        .section-label {
            color: #657189;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.68rem;
            font-weight: 800;
            margin: 0.9rem 0 0.55rem 0;
        }

        div[data-testid="stRadio"] > label {
            display: none;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 0.35rem;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            width: 100%;
            padding: 0.72rem 0.85rem;
            border-radius: 14px;
            border: 1px solid transparent;
            background: rgba(255,255,255,0.018);
            color: #98a4b7;
            transition: 0.2s ease;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
            background: rgba(255,255,255,0.055);
            border-color: rgba(255,255,255,0.07);
            color: #eef2ff;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
            background: linear-gradient(135deg, rgba(124,140,255,0.20), rgba(179,122,255,0.12));
            border-color: rgba(124,140,255,0.30);
            color: #ffffff;
            box-shadow: 0 12px 28px rgba(48,54,120,0.22);
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
            display: none;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label p {
            margin: 0;
            font-size: 0.88rem;
            font-weight: 700;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.3rem 2.35rem;
            border-radius: 30px;
            border: 1px solid rgba(255,255,255,0.12);
            background:
                linear-gradient(135deg, rgba(124,140,255,0.13), rgba(255,255,255,0.055) 47%, rgba(80,227,194,0.06));
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
        }

        .hero::before {
            content: "";
            position: absolute;
            right: -40px;
            top: -90px;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(124,140,255,0.26), transparent 70%);
            filter: blur(8px);
        }

        .hero-kicker {
            color: #9ca8c6;
            font-size: 0.8rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 800;
            margin-bottom: 0.65rem;
        }

        .hero-title {
            font-size: clamp(2rem, 4vw, 3.25rem);
            line-height: 1.02;
            letter-spacing: -0.055em;
            font-weight: 850;
            margin: 0;
            color: #ffffff;
        }

        .hero-subtitle {
            margin-top: 0.8rem;
            color: #aab4c7;
            max-width: 680px;
            font-size: 0.98rem;
            line-height: 1.65;
        }

        .glass-card {
            border: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(255,255,255,0.070), rgba(255,255,255,0.035));
            backdrop-filter: blur(22px);
            -webkit-backdrop-filter: blur(22px);
            border-radius: var(--radius-lg);
            box-shadow: 0 18px 60px rgba(0,0,0,0.24);
            padding: 1.2rem;
        }

        .glass-card-tight {
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.055);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: var(--radius-md);
            box-shadow: 0 14px 45px rgba(0,0,0,0.20);
            padding: 1rem;
        }

        .card-title {
            font-weight: 800;
            color: #f7f9ff;
            font-size: 1rem;
            letter-spacing: -0.015em;
        }

        .card-subtitle {
            color: #8f9bb0;
            font-size: 0.77rem;
            margin-top: 0.28rem;
            line-height: 1.5;
        }

        .metric-card {
            min-height: 118px;
            border: 1px solid rgba(255,255,255,0.11);
            background: linear-gradient(145deg, rgba(255,255,255,0.085), rgba(255,255,255,0.035));
            border-radius: 20px;
            padding: 1.1rem 1.15rem;
            box-shadow: 0 12px 42px rgba(0,0,0,0.20);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
        }

        .metric-label {
            color: #8995a9;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .metric-value {
            color: #ffffff;
            font-size: 1.9rem;
            font-weight: 850;
            letter-spacing: -0.045em;
            margin-top: 0.25rem;
        }

        .metric-caption {
            color: #8290a7;
            font-size: 0.72rem;
            margin-top: 0.15rem;
        }

        .prediction-shell {
            padding: 1.1rem;
            border-radius: 22px;
            border: 1px solid rgba(124,140,255,0.23);
            background: linear-gradient(135deg, rgba(124,140,255,0.14), rgba(179,122,255,0.07));
            box-shadow: 0 18px 55px rgba(72,72,160,0.16);
        }

        .prediction-label {
            color: #9ca8c6;
            text-transform: uppercase;
            letter-spacing: 0.13em;
            font-size: 0.7rem;
            font-weight: 850;
        }

        .prediction-score {
            color: #ffffff;
            font-size: 3.85rem;
            line-height: 1;
            font-weight: 900;
            letter-spacing: -0.065em;
            margin-top: 0.4rem;
        }

        .prediction-caption {
            color: #a3afc6;
            font-size: 0.8rem;
            margin-top: 0.42rem;
        }

        .score-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
            background: rgba(88,214,141,0.12);
            color: #8af0b0;
            border: 1px solid rgba(88,214,141,0.18);
        }

        .empty-state {
            padding: 2.6rem 1.4rem;
            text-align: center;
            border: 1px dashed rgba(255,255,255,0.14);
            border-radius: 24px;
            background: rgba(255,255,255,0.025);
        }

        .empty-icon {
            width: 62px;
            height: 62px;
            margin: 0 auto 0.8rem auto;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.6rem;
            background: rgba(124,140,255,0.12);
            border: 1px solid rgba(124,140,255,0.18);
        }

        .empty-title {
            color: #eef2ff;
            font-size: 1rem;
            font-weight: 800;
        }

        .empty-copy {
            color: #8490a5;
            font-size: 0.8rem;
            max-width: 520px;
            margin: 0.45rem auto 0 auto;
            line-height: 1.55;
        }

        .micro-note {
            color: #748198;
            font-size: 0.7rem;
            line-height: 1.5;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 0.35rem;
            background: #58d68d;
            box-shadow: 0 0 14px rgba(88,214,141,0.55);
        }

        .trend-up { color: #79e9a4; }
        .trend-down { color: #ff899d; }
        .trend-flat { color: #b4bfd1; }

        /* Streamlit inputs */
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stDateInput"] input,
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stMultiSelect"] > div,
        div[data-baseweb="select"] > div {
            background: rgba(255,255,255,0.052) !important;
            border: 1px solid rgba(255,255,255,0.11) !important;
            color: #f6f8ff !important;
            border-radius: 14px !important;
            backdrop-filter: blur(15px);
        }

        div[data-testid="stNumberInput"] button {
            background: rgba(255,255,255,0.04) !important;
            color: #b8c2d3 !important;
            border: none !important;
        }

        div[data-testid="stSlider"] div[data-baseweb="slider"] {
            padding-top: 0.45rem;
        }

        div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            box-shadow: 0 0 0 4px rgba(124,140,255,0.12);
        }

        label,
        .stMarkdown label,
        div[data-testid="stWidgetLabel"] p {
            color: #c6cfde !important;
            font-weight: 700 !important;
            font-size: 0.77rem !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            width: 100%;
            border-radius: 14px !important;
            min-height: 46px;
            border: 1px solid rgba(255,255,255,0.10) !important;
            background: linear-gradient(135deg, rgba(124,140,255,0.95), rgba(179,122,255,0.92)) !important;
            color: white !important;
            font-weight: 800 !important;
            letter-spacing: -0.01em;
            box-shadow: 0 12px 30px rgba(89,94,185,0.28);
            transition: 0.18s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 36px rgba(89,94,185,0.34);
            border-color: rgba(255,255,255,0.20) !important;
        }

        .secondary-action .stButton > button {
            background: rgba(255,255,255,0.05) !important;
            box-shadow: none;
        }

        .danger-action .stButton > button {
            background: rgba(255,117,143,0.12) !important;
            border-color: rgba(255,117,143,0.22) !important;
            color: #ff9db0 !important;
            box-shadow: none;
        }

        .divider {
            height: 1px;
            width: 100%;
            background: rgba(255,255,255,0.08);
            margin: 0.9rem 0 1.1rem 0;
        }

        .table-wrap {
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.09);
            overflow: hidden;
            background: rgba(255,255,255,0.025);
        }

        .footer-copy {
            color: #6f7b90;
            text-align: center;
            font-size: 0.68rem;
            padding-top: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA GENERATION / MODEL
# ============================================================


def clip_score(value: float) -> float:
    """Keep a predicted or real score in the expected 0-100 range."""

    return float(np.clip(value, 0.0, 100.0))


def generate_training_data(rows: int = 800) -> pd.DataFrame:
    """Generate reproducible synthetic training data for the local model."""

    rng = np.random.default_rng(RANDOM_STATE)

    attendance = rng.normal(86, 8, rows).clip(45, 100)
    study = rng.normal(3.1, 1.25, rows).clip(0.2, 8.0)
    previous = rng.normal(67, 15, rows).clip(20, 98)
    assignments = rng.normal(82, 15, rows).clip(25, 100)
    sleep = rng.normal(6.9, 1.1, rows).clip(3.5, 9.5)
    participation = rng.normal(72, 17, rows).clip(10, 100)
    practice = rng.normal(82, 45, rows).clip(5, 260)
    screen = rng.normal(3.4, 1.45, rows).clip(0.4, 8.0)
    stress = rng.normal(4.9, 1.9, rows).clip(1, 10)

    noise = rng.normal(0, 4.7, rows)

    score = (
        17.0
        + 0.14 * attendance
        + 3.75 * study
        + 0.39 * previous
        + 0.105 * assignments
        + 2.05 * sleep
        + 0.075 * participation
        + 0.075 * practice
        - 1.00 * screen
        - 1.55 * stress
        + noise
    )

    score = np.clip(score, 0, 100)

    return pd.DataFrame(
        {
            "attendance_pct": attendance,
            "study_hours_per_day": study,
            "previous_test_avg": previous,
            "assignment_completion_pct": assignments,
            "sleep_hours": sleep,
            "class_participation_pct": participation,
            "practice_questions_per_week": practice,
            "screen_time_hours": screen,
            "stress_level": stress,
            "final_score": score,
        }
    ).round(2)


def ensure_training_data() -> pd.DataFrame:
    """Load training data, creating a local synthetic set if needed."""

    if DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE)
            required = set(FEATURES + ["final_score"])
            if required.issubset(df.columns) and len(df) >= 100:
                return df.dropna(subset=FEATURES + ["final_score"]).copy()
        except Exception:
            pass

    df = generate_training_data()
    df.to_csv(DATA_FILE, index=False)
    return df


@st.cache_resource(show_spinner=False)
def train_model() -> Tuple[RandomForestRegressor, float]:
    """Train the prediction model and return the model + reference test MAE."""

    df = ensure_training_data()
    X = df[FEATURES]
    y = df["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=320,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = float(np.mean(np.abs(predictions - y_test.to_numpy())))

    return model, mae


# ============================================================
# HISTORY STORAGE
# ============================================================


HISTORY_COLUMNS = [
    "id",
    "saved_at",
    "test_name",
    "subject",
    "expected_marks",
    "actual_marks",
]


def empty_history() -> pd.DataFrame:
    """Return a correctly shaped empty history frame."""

    return pd.DataFrame(columns=HISTORY_COLUMNS)


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Make older or malformed history files safe to use."""

    if df is None or df.empty:
        return empty_history()

    result = df.copy()

    for col in HISTORY_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan

    result = result[HISTORY_COLUMNS]
    result["expected_marks"] = pd.to_numeric(result["expected_marks"], errors="coerce")
    result["actual_marks"] = pd.to_numeric(result["actual_marks"], errors="coerce")
    result["saved_at"] = result["saved_at"].fillna("").astype(str)
    result["test_name"] = result["test_name"].fillna("Test").astype(str)
    result["subject"] = result["subject"].fillna("General").astype(str)
    result["id"] = result["id"].fillna("").astype(str)

    result["expected_marks"] = result["expected_marks"].clip(0, 100)
    result["actual_marks"] = result["actual_marks"].clip(0, 100)

    return result


def load_history() -> pd.DataFrame:
    """Load persistent prediction history from the local CSV file."""

    if not HISTORY_FILE.exists():
        return empty_history()

    try:
        return normalize_history(pd.read_csv(HISTORY_FILE))
    except Exception:
        return empty_history()


def save_history(df: pd.DataFrame) -> None:
    """Persist history to disk."""

    clean = normalize_history(df)
    clean.to_csv(HISTORY_FILE, index=False)


def add_history_entry(
    df: pd.DataFrame,
    test_name: str,
    subject: str,
    expected_marks: float,
    actual_marks: Optional[float],
) -> pd.DataFrame:
    """Append one prediction to the history."""

    row = {
        "id": uuid.uuid4().hex[:12],
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "test_name": test_name.strip() or "Test",
        "subject": subject.strip() or "General",
        "expected_marks": clip_score(expected_marks),
        "actual_marks": np.nan if actual_marks is None else clip_score(actual_marks),
    }

    result = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_history(result)
    return normalize_history(result)


def update_actual_result(
    df: pd.DataFrame,
    record_id: str,
    actual_marks: float,
) -> pd.DataFrame:
    """Update the real score for an existing prediction."""

    result = df.copy()
    mask = result["id"].astype(str) == str(record_id)
    if mask.any():
        result.loc[mask, "actual_marks"] = clip_score(actual_marks)
        save_history(result)
    return normalize_history(result)


def delete_history_entry(df: pd.DataFrame, record_id: str) -> pd.DataFrame:
    """Delete a single history record."""

    result = df[df["id"].astype(str) != str(record_id)].copy()
    save_history(result)
    return normalize_history(result)


def clear_history() -> pd.DataFrame:
    """Delete all locally stored tracking history."""

    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
    return empty_history()


# ============================================================
# SMALL UTILITY HELPERS
# ============================================================


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float without raising UI errors."""

    try:
        return float(value)
    except Exception:
        return float(default)


def format_score(value: Any) -> str:
    """Format score values for the interface."""

    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "—"
    return f"{float(number):.1f}"


def format_difference(expected: Any, actual: Any) -> str:
    """Format the prediction difference."""

    expected_num = pd.to_numeric(expected, errors="coerce")
    actual_num = pd.to_numeric(actual, errors="coerce")
    if pd.isna(expected_num) or pd.isna(actual_num):
        return "—"
    difference = float(actual_num) - float(expected_num)
    sign = "+" if difference >= 0 else ""
    return f"{sign}{difference:.1f}"


def trend_text(history: pd.DataFrame) -> Tuple[str, str]:
    """Calculate a simple recent trend from actual results."""

    if history.empty:
        return "No results", "trend-flat"

    completed = history.dropna(subset=["actual_marks"]).copy()
    if len(completed) < 2:
        return "Start tracking", "trend-flat"

    completed = completed.sort_values("saved_at")
    recent = completed.tail(3)["actual_marks"].to_numpy(dtype=float)

    if len(recent) < 2:
        return "Start tracking", "trend-flat"

    delta = recent[-1] - recent[-2]
    if delta > 1:
        return f"↑ {delta:.1f} from last", "trend-up"
    if delta < -1:
        return f"↓ {abs(delta):.1f} from last", "trend-down"
    return "→ Holding steady", "trend-flat"


def performance_summary(history: pd.DataFrame) -> Dict[str, float]:
    """Return headline metrics for the completed tests."""

    completed = history.dropna(subset=["actual_marks"]).copy()
    if completed.empty:
        return {
            "tests": 0,
            "average": math.nan,
            "best": math.nan,
            "accuracy": math.nan,
        }

    error = (completed["actual_marks"] - completed["expected_marks"]).abs()

    return {
        "tests": float(len(completed)),
        "average": float(completed["actual_marks"].mean()),
        "best": float(completed["actual_marks"].max()),
        "accuracy": float((100 - error.mean()).clip(0, 100)),
    }


def page_header(kicker: str, title: str, subtitle: str) -> None:
    """Render a premium page header."""

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, caption: str = "") -> None:
    """Render a glass metric card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, copy: str) -> None:
    """Render a calm empty state."""

    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-icon">{icon}</div>
            <div class="empty-title">{title}</div>
            <div class="empty-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(current_page: str) -> str:
    """Render a reliable in-app sidebar navigation.

    This intentionally does not use Streamlit's collapsible native sidebar.
    A browser can persist a collapsed native-sidebar state, making navigation
    appear to disappear. Rendering the tracker as a dedicated left column keeps
    it visible and stable on every load.
    """

    st.markdown(
        """
        <div class="brand-wrap">
            <div class="brand-pill">📈</div>
            <div class="brand-title">ScoreTrack AI</div>
            <div class="brand-caption">Marks prediction made simple.</div>
        </div>
        <div class="section-label">Your tracker</div>
        """,
        unsafe_allow_html=True,
    )

    options = [
        "🎯  Predict Marks",
        "📊  Performance Graph",
        "🕘  History",
    ]

    default_index = {
        "Predict Marks": 0,
        "Performance Graph": 1,
        "History": 2,
    }.get(current_page, 0)

    selected = st.radio(
        "Navigation",
        options,
        index=default_index,
        label_visibility="collapsed",
        key="sidebar_nav",
    )

    st.markdown(
        """
        <div class="side-note">
            <span class="status-dot"></span>
            Local tracking is enabled. Your history stays in this project folder.
        </div>
        <div class="side-note" style="margin-top:0.7rem;">
            AI estimates help you compare expected and actual performance over time.
        </div>
        <div class="side-footer">
            ScoreTrack AI<br>
            <span>Student performance tracker</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return selected.split("  ", 1)[-1].strip()


# ============================================================
# INPUT FORM HELPERS
# ============================================================


def render_numeric_input(
    label: str,
    key: str,
    minimum: float,
    maximum: float,
    step: float,
    default: float,
    help_text: str,
) -> float:
    """Render a number input using a consistent design."""

    return st.number_input(
        label,
        min_value=float(minimum),
        max_value=float(maximum),
        value=float(default),
        step=float(step),
        key=key,
        help=help_text,
    )


def collect_student_inputs() -> Dict[str, float]:
    """Collect all model features in two neat glass panels."""

    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">Student profile</div>
            <div class="card-subtitle">Use your recent habits and scores. You can change these every time you take a new test.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            "<div class='card-title'>📚 Study & consistency</div>",
            unsafe_allow_html=True,
        )
        attendance = render_numeric_input(
            "Attendance (%)",
            "input_attendance",
            40,
            100,
            1,
            DEFAULT_INPUTS["attendance_pct"],
            FEATURE_HELP["attendance_pct"],
        )
        study_hours = render_numeric_input(
            "Study hours per day",
            "input_study",
            0.0,
            12.0,
            0.25,
            DEFAULT_INPUTS["study_hours_per_day"],
            FEATURE_HELP["study_hours_per_day"],
        )
        previous_avg = render_numeric_input(
            "Previous test average",
            "input_previous",
            0,
            100,
            1,
            DEFAULT_INPUTS["previous_test_avg"],
            FEATURE_HELP["previous_test_avg"],
        )
        assignments = render_numeric_input(
            "Assignment completion (%)",
            "input_assignments",
            0,
            100,
            1,
            DEFAULT_INPUTS["assignment_completion_pct"],
            FEATURE_HELP["assignment_completion_pct"],
        )
        practice = render_numeric_input(
            "Practice questions per week",
            "input_practice",
            0,
            400,
            5,
            DEFAULT_INPUTS["practice_questions_per_week"],
            FEATURE_HELP["practice_questions_per_week"],
        )

    with right:
        st.markdown(
            "<div class='card-title'>🧠 Wellbeing & engagement</div>",
            unsafe_allow_html=True,
        )
        sleep = render_numeric_input(
            "Sleep hours per night",
            "input_sleep",
            2.0,
            12.0,
            0.25,
            DEFAULT_INPUTS["sleep_hours"],
            FEATURE_HELP["sleep_hours"],
        )
        participation = render_numeric_input(
            "Class participation (%)",
            "input_participation",
            0,
            100,
            1,
            DEFAULT_INPUTS["class_participation_pct"],
            FEATURE_HELP["class_participation_pct"],
        )
        screen = render_numeric_input(
            "Screen time per day",
            "input_screen",
            0.0,
            14.0,
            0.25,
            DEFAULT_INPUTS["screen_time_hours"],
            FEATURE_HELP["screen_time_hours"],
        )
        stress = render_numeric_input(
            "Stress level (1–10)",
            "input_stress",
            1,
            10,
            1,
            DEFAULT_INPUTS["stress_level"],
            FEATURE_HELP["stress_level"],
        )

    return {
        "attendance_pct": attendance,
        "study_hours_per_day": study_hours,
        "previous_test_avg": previous_avg,
        "assignment_completion_pct": assignments,
        "sleep_hours": sleep,
        "class_participation_pct": participation,
        "practice_questions_per_week": practice,
        "screen_time_hours": screen,
        "stress_level": stress,
    }


# ============================================================
# PREDICT PAGE
# ============================================================


def render_prediction_result(prediction: float, mae: float) -> None:
    """Show the prediction as the visual centre of the page."""

    safe_prediction = clip_score(prediction)
    typical_error = max(2.5, mae)
    low = clip_score(safe_prediction - typical_error)
    high = clip_score(safe_prediction + typical_error)

    st.markdown(
        """
        <div class="prediction-shell">
            <div class="prediction-label">AI expected score</div>
            <div class="prediction-score">{score}<span style="font-size:1.5rem;color:#a8b1c4;font-weight:700;"> / 100</span></div>
            <div class="prediction-caption">Estimated range: {low}–{high} · based on the model's recent validation error.</div>
        </div>
        """.format(
            score=f"{safe_prediction:.1f}",
            low=f"{low:.1f}",
            high=f"{high:.1f}",
        ),
        unsafe_allow_html=True,
    )


def save_prediction_panel(prediction: float) -> None:
    """Collect test details and save the prediction to local history."""

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">Save this test</div>
            <div class="card-subtitle">Keep your prediction now, then add the real score later from History.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        test_name = st.text_input(
            "Test name",
            value="Class Test",
            key="save_test_name",
            placeholder="e.g. Physics Unit Test 2",
        )

    with right:
        subject = st.text_input(
            "Subject",
            value="Physics",
            key="save_subject",
            placeholder="e.g. Physics",
        )

    known_actual = st.checkbox(
        "I already know the real marks",
        value=False,
        key="known_actual",
    )

    actual_marks: Optional[float] = None
    if known_actual:
        actual_marks = st.number_input(
            "Real marks",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key="actual_marks_now",
        )

    if st.button("💾 Save prediction", key="save_prediction_button"):
        history = load_history()
        new_history = add_history_entry(
            history,
            test_name,
            subject,
            prediction,
            actual_marks,
        )
        st.session_state.history = new_history
        st.session_state.last_prediction_saved = True
        st.success("Saved to your history.")

    if st.session_state.get("last_prediction_saved", False):
        st.markdown(
            "<div class='micro-note'>Your prediction is now visible in History and will appear on the Performance Graph once a real score is added.</div>",
            unsafe_allow_html=True,
        )


def render_predict_page(model: RandomForestRegressor, mae: float) -> None:
    """Render the main marks prediction experience."""

    history = load_history()
    st.session_state.history = history

    page_header(
        "PREDICT MARKS",
        "Know your expected score before the test.",
        "Enter a few study and routine details. ScoreTrack AI turns them into an expected score, then lets you compare the prediction with what you actually achieved.",
    )

    summary = performance_summary(history)
    trend, trend_class = trend_text(history)

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        metric_card(
            "Completed tests",
            str(int(summary["tests"])),
            "real results recorded",
        )
    with m2:
        average = "—" if math.isnan(summary["average"]) else f"{summary['average']:.1f}"
        metric_card("Average score", average, "across completed tests")
    with m3:
        best = "—" if math.isnan(summary["best"]) else f"{summary['best']:.1f}"
        metric_card("Best score", best, "your highest recorded mark")
    with m4:
        metric_card("Recent trend", trend, "last completed test")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.08, 0.92], gap="large")

    with left:
        inputs = collect_student_inputs()

        if st.button("✨ Predict my marks", key="predict_button"):
            feature_frame = pd.DataFrame([inputs])[FEATURES]
            prediction = float(model.predict(feature_frame)[0])
            st.session_state.prediction = clip_score(prediction)
            st.session_state.prediction_inputs = inputs
            st.session_state.last_prediction_saved = False

    with right:
        prediction = st.session_state.get("prediction")

        if prediction is None:
            empty_state(
                "✨",
                "Your prediction will appear here",
                "Complete the profile on the left and select “Predict my marks”. Your expected score will stay here while you save the test.",
            )
        else:
            render_prediction_result(prediction, mae)
            save_prediction_panel(prediction)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # A small personal context panel keeps the page focused on tracking.
    st.markdown(
        """
        <div class="glass-card-tight">
            <div class="card-title">How to use your score</div>
            <div class="card-subtitle">
                Save the prediction before the exam. After you receive the actual mark, open History and add it to the same test. The Performance Graph will then show the gap between expectation and reality.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='footer-copy'>Predictions are estimates generated from the local machine-learning model and are not official grades.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# GRAPH PAGE
# ============================================================


def build_expected_vs_real_graph(history: pd.DataFrame) -> Optional[plt.Figure]:
    """Build a clean expected-vs-reality chart for completed tests."""

    completed = history.dropna(subset=["actual_marks", "expected_marks"]).copy()
    if completed.empty:
        return None

    completed = completed.sort_values("saved_at").reset_index(drop=True)

    x = completed["expected_marks"].astype(float).to_numpy()
    y = completed["actual_marks"].astype(float).to_numpy()

    figure, axis = plt.subplots(figsize=(10.6, 5.7), dpi=120)
    figure.patch.set_alpha(0)
    axis.set_facecolor("#0e1828")

    lower = min(float(np.min(x)), float(np.min(y)), 0)
    upper = max(float(np.max(x)), float(np.max(y)), 100)

    axis.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.8,
        color="#7f8aa2",
        alpha=0.65,
        label="Expectation = reality",
    )

    axis.scatter(
        x,
        y,
        s=88,
        c="#7c8cff",
        edgecolors="#ffffff",
        linewidths=1.3,
        alpha=0.92,
        zorder=3,
    )

    for index, row in completed.iterrows():
        label = str(row["test_name"])
        if len(label) > 16:
            label = label[:13] + "…"
        axis.annotate(
            label,
            (row["expected_marks"], row["actual_marks"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color="#b9c2d3",
        )

    axis.set_xlim(max(0, lower - 5), min(100, upper + 5))
    axis.set_ylim(max(0, lower - 5), min(100, upper + 5))
    axis.set_xlabel("Expected marks", color="#aeb8ca", labelpad=8)
    axis.set_ylabel("Real marks", color="#aeb8ca", labelpad=8)
    axis.set_title("Expected vs Reality", loc="left", color="#f7f9ff", fontsize=15, fontweight="bold", pad=14)
    axis.grid(alpha=0.10, color="#ffffff")
    axis.tick_params(colors="#8f9bb0")

    for spine in axis.spines.values():
        spine.set_color("#243047")

    legend = axis.legend(frameon=False, loc="lower right")
    for text_item in legend.get_texts():
        text_item.set_color("#9da8ba")

    figure.tight_layout()
    return figure


def render_graph_page() -> None:
    """Render the expected-vs-reality performance graph."""

    history = load_history()
    completed = history.dropna(subset=["actual_marks", "expected_marks"]).copy()

    page_header(
        "PERFORMANCE GRAPH",
        "See how close your predictions are.",
        "Every completed test becomes a point on the graph. The closer a point is to the diagonal line, the closer your real mark was to the AI expectation.",
    )

    if completed.empty:
        empty_state(
            "📊",
            "No completed tests yet",
            "Save a prediction first. Once you know the real mark, add it from History and this page will turn your results into a personal expected-vs-reality graph.",
        )
        return

    completed = completed.sort_values("saved_at").reset_index(drop=True)
    errors = (completed["actual_marks"] - completed["expected_marks"]).abs()

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric_card("Tests compared", str(len(completed)), "expected vs real")
    with c2:
        metric_card("Average real", f"{completed['actual_marks'].mean():.1f}", "marks")
    with c3:
        metric_card("Average gap", f"{errors.mean():.1f}", "marks between expected and real")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    figure = build_expected_vs_real_graph(history)
    if figure is not None:
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

    # Recent comparison cards.
    recent = completed.tail(5).iloc[::-1]
    columns = st.columns(min(5, len(recent)), gap="small")

    for column, (_, row) in zip(columns, recent.iterrows()):
        actual = safe_float(row["actual_marks"])
        expected = safe_float(row["expected_marks"])
        gap = actual - expected
        trend_color = "trend-up" if gap >= 0 else "trend-down"
        gap_text = f"+{gap:.1f}" if gap >= 0 else f"{gap:.1f}"
        test_name = str(row["test_name"])
        if len(test_name) > 18:
            test_name = test_name[:15] + "…"

        with column:
            st.markdown(
                f"""
                <div class="glass-card-tight">
                    <div class="metric-label">{row['subject']}</div>
                    <div class="card-title" style="margin-top:0.2rem;">{test_name}</div>
                    <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-top:0.8rem;">
                        <div>
                            <div class="micro-note">Expected</div>
                            <div style="font-size:1.35rem;font-weight:850;color:#fff;">{expected:.1f}</div>
                        </div>
                        <div>
                            <div class="micro-note">Real</div>
                            <div style="font-size:1.35rem;font-weight:850;color:#fff;">{actual:.1f}</div>
                        </div>
                    </div>
                    <div class="{trend_color}" style="font-size:0.75rem;font-weight:800;margin-top:0.6rem;">{gap_text} marks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# HISTORY PAGE
# ============================================================


def history_status(expected: Any, actual: Any) -> Tuple[str, str]:
    """Return human-readable history status."""

    if pd.isna(actual):
        return "Awaiting result", "trend-flat"

    diff = safe_float(actual) - safe_float(expected)
    if diff >= 0:
        return f"+{diff:.1f} vs AI", "trend-up"
    return f"{diff:.1f} vs AI", "trend-down"


def render_history_table(history: pd.DataFrame) -> None:
    """Render a compact tracker table."""

    if history.empty:
        empty_state(
            "🕘",
            "Your history is empty",
            "Every prediction you save will appear here. Once the test is over, you can add the real mark and build a long-term performance trail.",
        )
        return

    view = history.copy().sort_values("saved_at", ascending=False)
    view["Expected"] = view["expected_marks"].map(format_score)
    view["Real"] = view["actual_marks"].map(format_score)
    view["Difference"] = [
        format_difference(e, a)
        for e, a in zip(view["expected_marks"], view["actual_marks"])
    ]
    view["Test"] = view["test_name"].astype(str)
    view["Subject"] = view["subject"].astype(str)
    view["Date"] = view["saved_at"].astype(str)

    display = view[["Date", "Test", "Subject", "Expected", "Real", "Difference"]].reset_index(drop=True)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Test": st.column_config.TextColumn("Test", width="medium"),
            "Subject": st.column_config.TextColumn("Subject", width="small"),
            "Expected": st.column_config.TextColumn("Expected", width="small"),
            "Real": st.column_config.TextColumn("Real", width="small"),
            "Difference": st.column_config.TextColumn("Gap", width="small"),
        },
    )


def render_update_panel(history: pd.DataFrame) -> pd.DataFrame:
    """Allow users to add actual marks after a test."""

    pending = history[history["actual_marks"].isna()].copy()

    if pending.empty:
        return history

    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">Add a real result</div>
            <div class="card-subtitle">Pick a saved prediction, enter the real mark, and ScoreTrack AI will update your graph automatically.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    labels: List[str] = []
    id_by_label: Dict[str, str] = {}

    for _, row in pending.sort_values("saved_at", ascending=False).iterrows():
        label = f"{row['test_name']} · {row['subject']} · expected {format_score(row['expected_marks'])}"
        labels.append(label)
        id_by_label[label] = str(row["id"])

    selected_label = st.selectbox(
        "Prediction to update",
        labels,
        key="history_update_select",
    )

    actual = st.number_input(
        "Real marks",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key="history_update_actual",
    )

    if st.button("✅ Update result", key="update_result_button"):
        updated = update_actual_result(
            history,
            id_by_label[selected_label],
            actual,
        )
        st.session_state.history = updated
        st.success("Real marks added. Your graph is updated.")
        return updated

    return history


def render_manage_history_panel(history: pd.DataFrame) -> pd.DataFrame:
    """Offer delete controls without adding another navigation page."""

    if history.empty:
        return history

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card-tight">
            <div class="card-title">Manage history</div>
            <div class="card-subtitle">You can remove one record or clear the local tracker.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    labels: List[str] = []
    ids: Dict[str, str] = {}

    for _, row in history.sort_values("saved_at", ascending=False).iterrows():
        label = f"{row['test_name']} · {row['subject']} · {row['saved_at']}"
        labels.append(label)
        ids[label] = str(row["id"])

    selected = st.selectbox(
        "Record",
        labels,
        key="history_delete_select",
    )

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        if st.button("🗑 Delete record", key="delete_one_button"):
            updated = delete_history_entry(history, ids[selected])
            st.session_state.history = updated
            st.success("Record deleted.")
            return updated

    with c2:
        confirm = st.checkbox(
            "Confirm clear all",
            key="confirm_clear_all",
        )
        if confirm and st.button("Clear all history", key="clear_all_button"):
            updated = clear_history()
            st.session_state.history = updated
            st.success("History cleared.")
            return updated

    return history


def render_history_page() -> None:
    """Render the student's complete prediction history."""

    history = st.session_state.get("history", load_history())
    history = normalize_history(history)

    page_header(
        "HISTORY",
        "Your marks, one test at a time.",
        "See every prediction you have saved, add real results after your tests, and keep a clean record of how your expectations compare with reality.",
    )

    summary = performance_summary(history)
    trend, _ = trend_text(history)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Saved predictions", str(len(history)), "all tests in your tracker")
    with c2:
        metric_card(
            "Completed",
            str(int(summary["tests"])),
            "tests with real marks",
        )
    with c3:
        average = "—" if math.isnan(summary["average"]) else f"{summary['average']:.1f}"
        metric_card("Average real", average, "completed results")
    with c4:
        metric_card("Recent trend", trend, "based on last two results")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    history = render_update_panel(history)
    st.session_state.history = history

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    render_history_table(history)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    history = render_manage_history_panel(history)
    st.session_state.history = history

    csv_bytes = normalize_history(history).to_csv(index=False).encode("utf-8")
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Export history as CSV",
        data=csv_bytes,
        file_name="marks_history.csv",
        mime="text/csv",
        key="download_history",
    )

    st.markdown(
        "<div class='footer-copy'>Your history file is stored locally next to the app.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# APP BOOTSTRAP
# ============================================================


def initialize_session() -> None:
    """Initialize the small amount of session state used by the app."""

    if "history" not in st.session_state:
        st.session_state.history = load_history()

    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    if "prediction_inputs" not in st.session_state:
        st.session_state.prediction_inputs = None

    if "last_prediction_saved" not in st.session_state:
        st.session_state.last_prediction_saved = False


def main() -> None:
    """Main Streamlit entrypoint."""

    inject_global_styles()
    initialize_session()

    model, mae = train_model()

    current_page_key = st.session_state.get("active_page", "Predict Marks")

    # Permanent left navigation so the tracker cannot disappear because of a
    # browser-persisted Streamlit sidebar state.
    nav_col, content_col = st.columns([1.08, 4.92], gap="large")

    with nav_col:
        selected_page = render_sidebar(current_page_key)

    with content_col:
        st.session_state.active_page = selected_page

        if selected_page == "Performance Graph":
            render_graph_page()
        elif selected_page == "History":
            render_history_page()
        else:
            render_predict_page(model, mae)


if __name__ == "__main__":
    main()
