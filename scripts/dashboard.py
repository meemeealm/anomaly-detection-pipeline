"""window-aware, flatline-aware, config-aware: live dashboard over the SQLite results.

    streamlit run scripts/dashboard.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from iot_anomaly.runtime_config import resolve_one  # noqa: E402

# Same database the consumer writes: IOT_STREAM_DB (legacy override) > env/config/default from runtime_config
DB = Path(os.environ.get("IOT_STREAM_DB") or resolve_one("db")[0])

st.set_page_config(page_title="IoT anomaly stream", layout="wide")
st.title("IoT anomaly stream")
window_min = st.sidebar.slider("Window (minutes)", 1, 60, 5)
refresh_s = st.sidebar.slider("Refresh every (s)", 1, 10, 2)
st.sidebar.caption(f"DB: {DB}")

READINGS = """
SELECT r.device_id, d.site, r.event_ts, r.score, r.is_anomaly, r.label,
       r.e2e_ms, r.infer_ms, r.batch_size, r.model, r.run_id, r.alert_reason
FROM readings r LEFT JOIN devices d ON d.device_id = r.device_id
WHERE r.event_ts >= ? ORDER BY r.event_ts
"""
FAILURES = ("SELECT reason, COUNT(*) AS n FROM failed_events WHERE ts >= ? "
            "GROUP BY reason ORDER BY n DESC")
RUNS = ("SELECT run_id, component, status, datetime(started_at,'unixepoch') AS started, "
        "datetime(heartbeat_at,'unixepoch') AS last_seen FROM runs ORDER BY started_at DESC LIMIT 8")


def load(minutes: int):
    if not DB.exists():
        return None
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
    since = time.time() - minutes * 60
    try:
        return (pd.read_sql_query(READINGS, conn, params=(since,)),
                pd.read_sql_query(FAILURES, conn, params=(since,)),
                pd.read_sql_query(RUNS, conn))
    finally:
        conn.close()


@st.fragment(run_every=refresh_s)
def live() -> None:
    try:
        loaded = load(window_min)
    except (sqlite3.OperationalError, pd.errors.DatabaseError) as exc:
        st.warning(f"Database not ready or old schema ({exc}). Restart the consumer once to migrate it.")
        return
    if loaded is None:
        st.info("Waiting for data. Start the consumer and the publisher.")
        return
    df, failures, runs = loaded
    if df.empty:
        st.info("No scored readings in this window yet (sliding-window models need a warm-up period).")
    else:
        models = sorted(df.model.dropna().unique())
        if len(models) > 1:
            choice = st.sidebar.selectbox("Model", ["all"] + models)
            if choice != "all":
                df = df[df.model == choice]

        span = max(float(df.event_ts.max() - df.event_ts.min()), 1.0)
        flagged = df.is_anomaly == 1
        truth = df.label == 1
        tp = int((flagged & truth).sum())
        fp = int((flagged & ~truth).sum())
        fn = int((~flagged & truth).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")

        cols = st.columns(7)
        cols[0].metric("Readings", f"{len(df):,}")
        cols[1].metric("Throughput", f"{len(df) / span:.0f}/s")
        cols[2].metric("Flagged", f"{flagged.mean():.1%}")
        cols[3].metric("e2e p50 / p95 (ms)", f"{df.e2e_ms.quantile(.5):.1f} / {df.e2e_ms.quantile(.95):.1f}")
        cols[4].metric("Precision", f"{precision:.2f}")
        cols[5].metric("Recall", f"{recall:.2f}")
        cols[6].metric("Failed events", f"{int(failures.n.sum()) if not failures.empty else 0:,}")

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
        st.dataframe(recent[["device_id", "site", "score", "alert_reason", "label", "model", "e2e_ms"]])

    st.subheader("Failed events (window)")
    if failures.empty:
        st.caption("None recorded.")
    else:
        st.dataframe(failures)
    st.subheader("Recent runs")
    st.dataframe(runs)


live()
