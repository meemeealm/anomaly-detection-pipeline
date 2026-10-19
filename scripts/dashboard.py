"""Live dashboard over the SQLite results.

    streamlit run scripts/dashboard.py
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get("IOT_STREAM_DB", ROOT / "results" / "stream.db"))

st.set_page_config(page_title="IoT anomaly stream", layout="wide")
st.title("IoT anomaly stream")
window_min = st.sidebar.slider("Window (minutes)", 1, 60, 5)
refresh_s = st.sidebar.slider("Refresh every (s)", 1, 10, 2)
st.sidebar.caption(f"DB: {DB}")

QUERY = """
SELECT r.device_id, d.site, r.event_ts, r.score, r.is_anomaly, r.label,
       r.e2e_ms, r.infer_ms, r.batch_size
FROM readings r LEFT JOIN devices d ON d.device_id = r.device_id
WHERE r.event_ts >= ? ORDER BY r.event_ts
"""


def load(minutes: int):
    if not DB.exists():
        return None
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
    try:
        return pd.read_sql_query(QUERY, conn, params=(time.time() - minutes * 60,))
    finally:
        conn.close()


@st.fragment(run_every=refresh_s)
def live() -> None:
    df = load(window_min)
    if df is None or df.empty:
        st.info("Waiting for data. Start the consumer and the publisher.")
        return

    span = max(float(df.event_ts.max() - df.event_ts.min()), 1.0)
    flagged = df.is_anomaly == 1
    truth = df.label == 1
    tp = int((flagged & truth).sum())
    fp = int((flagged & ~truth).sum())
    fn = int((~flagged & truth).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")

    cols = st.columns(6)
    cols[0].metric("Readings", f"{len(df):,}")
    cols[1].metric("Throughput", f"{len(df) / span:.0f}/s")
    cols[2].metric("Flagged", f"{flagged.mean():.1%}")
    cols[3].metric("e2e p50 / p95 (ms)", f"{df.e2e_ms.quantile(.5):.1f} / {df.e2e_ms.quantile(.95):.1f}")
    cols[4].metric("Precision", f"{precision:.2f}")
    cols[5].metric("Recall", f"{recall:.2f}")

    left, right = st.columns([3, 2])
    with left:
        device = st.selectbox("Device", sorted(df.device_id.unique()))
        d = df[df.device_id == device].copy()
        d["time"] = pd.to_datetime(d.event_ts, unit="s")
        st.line_chart(d.set_index("time")[["score"]])
    with right:
        st.caption("Flagged readings per device")
        st.bar_chart(df[flagged].groupby("device_id").size().rename("flagged"))

    st.subheader("Most recent flagged readings")
    recent = df[flagged].tail(20).iloc[::-1]
    st.dataframe(recent[["device_id", "site", "score", "label", "e2e_ms"]])


live()
