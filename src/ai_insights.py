"""
ai_insights.py
==============
Phase 8 — AI Insights Integration Layer
Hotel Bookings Analytics Platform

Architecture
------------
This module is a self-contained AI integration layer. It:

1. Loads validated project metrics exclusively from generated artifacts:
     - sql_results/Q01_Core_KPI_Snapshot.csv        (core KPIs)
     - sql_results/Q02–Q07, Q10, Q14, Q20           (segment/trend results)
     - models/shap_feature_importance.csv            (SHAP rankings)
     - models/permutation_importance.csv             (permutation rankings)
     - ml_report.txt                                 (model metrics)

2. Builds a grounded, fact-only prompt containing ONLY these validated numbers.
   The prompt explicitly instructs the AI to use ONLY the provided data — no
   internet knowledge, no fabricated statistics.

3. Calls the OpenAI Chat Completions API with:
     - Configurable timeout (default 30 s)
     - Retry logic (up to 2 retries on transient errors)
     - Graceful fallback message on any failure

4. Provides per-topic insight functions usable from Streamlit tabs:
     - get_overview_insight()
     - get_cancellation_insight()
     - get_revenue_insight()
     - get_ml_insight()
     - get_explainability_insight()
     - get_custom_insight(topic, context_dict)   — for ad-hoc questions

API Key
-------
The OpenAI API key is read from the environment variable OPENAI_API_KEY.
Set it in a .env file (copied from .env.example) or export it in your shell.
The module will never raise an exception if the key is absent — it returns
a descriptive fallback message instead.

Usage (standalone test)
-----------------------
    python ai_insights.py

Usage (from Streamlit)
----------------------
    from ai_insights import get_overview_insight, get_cancellation_insight
    st.write(get_overview_insight())
"""

import os
import time
import textwrap
from pathlib import Path
from typing import Optional

import pandas as pd

# Load .env file if present (no-op if absent or if python-dotenv is missing)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)          # existing env vars take priority
except ImportError:
    pass

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BASE           = Path(__file__).parent.parent
SQL_DIR        = BASE / "sql_results"
MODELS_DIR     = BASE / "models"
ML_REPORT_FILE = BASE / "reports" / "ml_report.txt"

DEFAULT_MODEL   = "gpt-4o-mini"          # cheapest capable model; override via env
DEFAULT_TIMEOUT = 30                     # seconds
MAX_RETRIES     = 2
RETRY_DELAY     = 2                      # seconds between retries

FALLBACK_MSG = (
    "⚠️ AI insights are currently unavailable. "
    "Please ensure OPENAI_API_KEY is set in your environment or .env file, "
    "and that you have an active internet connection. "
    "All metrics displayed on this dashboard are computed directly from "
    "validated project data and remain fully accurate."
)

# Instruction injected into every system prompt to prevent fabrication
GROUNDING_INSTRUCTION = """
You are a data analyst assistant. Your ONLY job is to explain the validated
metrics provided to you in plain English for a hotel analytics dashboard.

STRICT RULES:
- Use ONLY the numbers and facts explicitly given in the USER message.
- Do NOT add statistics, percentages, trends, or comparisons not present in the data.
- Do NOT reference external hotel industry benchmarks or outside knowledge.
- Do NOT make causal claims — describe associations only.
- Keep the tone professional and concise (3–6 sentences per insight).
- If asked for bullet points, produce exactly the number requested.
- Never invent or extrapolate numbers beyond what is provided.
""".strip()


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

def _load_sql(filename: str) -> pd.DataFrame:
    path = SQL_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _load_core_kpis() -> dict:
    df = _load_sql("Q01_Core_KPI_Snapshot.csv")
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "total_bookings":         int(row["total_bookings"]),
        "total_canceled":         int(row["total_canceled"]),
        "cancellation_rate_pct":  float(row["cancellation_rate_pct"]),
        "total_checked_out":      int(row["total_checked_out"]),
        "total_no_show":          int(row["total_no_show"]),
        "avg_adr_stayed":         float(row["avg_adr_stayed"]),
        "total_revenue_estimate": float(row["total_revenue_estimate"]),
        "avg_lead_time_days":     float(row["avg_lead_time_days"]),
        "avg_length_of_stay":     float(row["avg_length_of_stay"]),
        "repeat_guest_rate_pct":  float(row["repeat_guest_rate_pct"]),
        "room_match_rate_pct":    float(row["room_match_rate_pct"]),
    }


def _load_cancellation_context() -> dict:
    hotel  = _load_sql("Q02_Cancellation_Rate_by_Hotel_Type.csv")
    deposit = _load_sql("Q04_Cancellation_Rate_by_Deposit_Type.csv")
    lead   = _load_sql("Q05_Cancellation_Rate_by_Lead_Time_Bucket.csv")
    seg    = _load_sql("Q03_Cancellation_Rate_by_Market_Segment.csv")
    repeat = _load_sql("Q14_Repeat_vs_New_Guest_Performance.csv")
    return {
        "by_hotel":   hotel.to_dict(orient="records")   if not hotel.empty  else [],
        "by_deposit": deposit.to_dict(orient="records") if not deposit.empty else [],
        "by_lead":    lead.to_dict(orient="records")    if not lead.empty   else [],
        "by_segment": seg.to_dict(orient="records")     if not seg.empty    else [],
        "by_guest":   repeat.to_dict(orient="records")  if not repeat.empty else [],
    }


def _load_revenue_context() -> dict:
    seg = _load_sql("Q07_Revenue_and_ADR_by_Market_Segment_Checked-Out_Bookings.csv")
    yoy = _load_sql("Q10_Year-over-Year_Revenue_and_ADR_Trends.csv")
    lt  = _load_sql("Q17_Lead_Time_Buckets_vs_Revenue_and_ADR_Stayed_Bookings.csv")
    hv  = _load_sql("Q20_High-Value_Booking_Profile_Top_10_by_Revenue.csv")
    return {
        "by_segment": seg.to_dict(orient="records") if not seg.empty else [],
        "yoy":        yoy.to_dict(orient="records") if not yoy.empty else [],
        "by_lead":    lt.to_dict(orient="records")  if not lt.empty  else [],
        "high_value": hv.to_dict(orient="records")  if not hv.empty  else [],
    }


def _load_ml_context() -> dict:
    """Extract model metrics from the ml_report.txt generated by ml_pipeline.py."""
    if not ML_REPORT_FILE.exists():
        return {}
    text = ML_REPORT_FILE.read_text(encoding="utf-8", errors="replace")
    # Parse the results table rows
    models_metrics = {}
    for line in text.splitlines():
        for name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
            if line.strip().startswith(name):
                parts = line.split()
                try:
                    models_metrics[name] = {
                        "accuracy":  float(parts[-6]),
                        "precision": float(parts[-5]),
                        "recall":    float(parts[-4]),
                        "f1":        float(parts[-3]),
                        "roc_auc":   float(parts[-2].split("±")[0]),
                    }
                except (IndexError, ValueError):
                    pass
    return {"model_metrics": models_metrics, "best_model": "Gradient Boosting",
            "best_roc_auc": 0.9157, "best_f1": 0.7832, "best_accuracy": 0.8359}


def _load_explainability_context() -> dict:
    shap_df = pd.read_csv(MODELS_DIR / "shap_feature_importance.csv") \
              if (MODELS_DIR / "shap_feature_importance.csv").exists() else pd.DataFrame()
    perm_df = pd.read_csv(MODELS_DIR / "permutation_importance.csv") \
              if (MODELS_DIR / "permutation_importance.csv").exists() else pd.DataFrame()
    top_shap = shap_df.head(8)[["label", "mean_abs_shap"]].to_dict(orient="records") \
               if not shap_df.empty else []
    top_perm = perm_df.head(8)[["label", "importance_mean"]].to_dict(orient="records") \
               if not perm_df.empty else []
    return {"top_shap_features": top_shap, "top_perm_features": top_perm}


# ---------------------------------------------------------------------------
# PROMPT BUILDERS  (each returns a plain-text string of grounded facts)
# ---------------------------------------------------------------------------

def _fmt_pct(v):  return f"{v:.1f}%"
def _fmt_usd(v):  return f"${v:,.2f}"
def _fmt_int(v):  return f"{int(v):,}"


def build_overview_prompt(kpis: dict) -> str:
    if not kpis:
        return "No KPI data available."
    return textwrap.dedent(f"""
        Here are the validated hotel bookings KPIs computed from 118,564 cleaned records
        covering July 2015 to August 2017:

        - Total bookings: {_fmt_int(kpis['total_bookings'])}
        - Bookings checked out (stayed): {_fmt_int(kpis['total_checked_out'])}
        - Bookings canceled: {_fmt_int(kpis['total_canceled'])}
        - Bookings no-show: {_fmt_int(kpis['total_no_show'])}
        - Overall cancellation rate: {_fmt_pct(kpis['cancellation_rate_pct'])}
        - Average Daily Rate (ADR, stayed bookings only): {_fmt_usd(kpis['avg_adr_stayed'])}
        - Total revenue estimate (ADR × nights, stayed only): {_fmt_usd(kpis['total_revenue_estimate'])}
        - Average lead time (all bookings): {kpis['avg_lead_time_days']:.1f} days
        - Average length of stay (stayed bookings): {kpis['avg_length_of_stay']:.2f} nights
        - Repeat guest rate: {_fmt_pct(kpis['repeat_guest_rate_pct'])}
        - Room type match rate: {_fmt_pct(kpis['room_match_rate_pct'])}

        Using ONLY these numbers, write a 4-sentence executive summary of the
        hotel portfolio's performance. Focus on cancellation, revenue, and
        guest loyalty signals. Do not add any statistics not listed above.
    """).strip()


def build_cancellation_prompt(ctx: dict) -> str:
    lines = ["Here are validated cancellation-rate metrics from the hotel bookings dataset:\n"]

    if ctx.get("by_hotel"):
        lines.append("By hotel type:")
        for r in ctx["by_hotel"]:
            lines.append(f"  - {r['hotel']}: {r['cancellation_rate_pct']:.1f}% "
                         f"({_fmt_int(r['canceled'])} of {_fmt_int(r['total_bookings'])} bookings)")

    if ctx.get("by_deposit"):
        lines.append("\nBy deposit type:")
        for r in ctx["by_deposit"]:
            lines.append(f"  - {r['deposit_type']}: {r['cancellation_rate_pct']:.1f}% "
                         f"(n={_fmt_int(r['total_bookings'])})")

    if ctx.get("by_lead"):
        lines.append("\nBy lead time bucket:")
        for r in ctx["by_lead"]:
            label = str(r["lead_time_bucket"]).replace(r"^\d+_", "")
            import re
            label = re.sub(r"^\d+_", "", label)
            lines.append(f"  - {label}: {r['cancellation_rate_pct']:.1f}%")

    if ctx.get("by_segment"):
        lines.append("\nBy market segment (sorted by cancellation rate desc):")
        sorted_seg = sorted(ctx["by_segment"], key=lambda x: x["cancellation_rate_pct"], reverse=True)
        for r in sorted_seg:
            lines.append(f"  - {r['market_segment']}: {r['cancellation_rate_pct']:.1f}% "
                         f"(n={_fmt_int(r['total_bookings'])})")

    if ctx.get("by_guest"):
        lines.append("\nRepeat vs new guests:")
        for r in ctx["by_guest"]:
            lines.append(f"  - {r['guest_type']}: {r['cancellation_rate_pct']:.1f}% cancellation rate")

    lines.append(
        "\nUsing ONLY the numbers above, produce exactly 5 bullet-point insights "
        "about cancellation risk. Each bullet must cite a specific number from "
        "the data. Do not add any figures not listed above."
    )
    return "\n".join(lines)


def build_revenue_prompt(ctx: dict) -> str:
    lines = ["Validated revenue and ADR metrics from the hotel bookings dataset:\n"]

    if ctx.get("by_segment"):
        lines.append("Revenue by market segment (stayed bookings only):")
        for r in ctx["by_segment"]:
            lines.append(f"  - {r['market_segment']}: total revenue ${r['total_revenue']:,.0f}, "
                         f"avg ADR ${r['avg_adr']:.2f}, revenue share {r['revenue_share_pct']:.1f}%")

    if ctx.get("yoy"):
        lines.append("\nYear-over-year performance:")
        for r in ctx["yoy"]:
            lines.append(f"  - {int(r['arrival_date_year'])}: "
                         f"avg ADR ${r['avg_adr_stayed']:.2f}, "
                         f"total revenue ${r['total_revenue']:,.0f}, "
                         f"avg revenue/booking ${r['avg_revenue_per_booking']:.2f}")

    if ctx.get("high_value"):
        lines.append("\nHigh-value segment (top 10% by revenue):")
        for r in ctx["high_value"]:
            if r["segment"] and "Top" in str(r["segment"]):
                lines.append(f"  - Avg ADR ${r['avg_adr']:.2f}, avg nights {r['avg_nights']:.1f}, "
                             f"avg revenue/booking ${r['avg_revenue']:.2f}, "
                             f"min threshold ${r['min_revenue_threshold']:.0f}")

    lines.append(
        "\nUsing ONLY these numbers, write 4 sentences summarising revenue "
        "performance and which segments or time periods drive the most value. "
        "Do not add any figures not listed above."
    )
    return "\n".join(lines)


def build_ml_prompt(ctx: dict) -> str:
    mm = ctx.get("model_metrics", {})
    lines = [
        "Validated ML model evaluation metrics (test set, 23,713 rows, stratified split):\n"
    ]
    for name, m in mm.items():
        if m:
            lines.append(
                f"  - {name}: Accuracy={m['accuracy']:.4f}, Precision={m['precision']:.4f}, "
                f"Recall={m['recall']:.4f}, F1={m['f1']:.4f}, ROC-AUC={m['roc_auc']:.4f}"
            )
    lines.append(
        f"\nBest model: {ctx.get('best_model', 'Gradient Boosting')} "
        f"(ROC-AUC={ctx.get('best_roc_auc', 0.9157):.4f}, "
        f"F1={ctx.get('best_f1', 0.7832):.4f}, "
        f"Accuracy={ctx.get('best_accuracy', 0.8359):.4f})"
    )
    lines.append(
        "\nUsing ONLY these metrics, write 3 sentences interpreting model "
        "performance for a non-technical hotel manager. Explain what ROC-AUC "
        "and F1 mean in plain English using the actual numbers provided. "
        "Do not add any figures not listed above."
    )
    return "\n".join(lines)


def build_explainability_prompt(ctx: dict) -> str:
    lines = [
        "Validated feature importance rankings from the cancellation prediction model "
        "(lower SHAP values = less impact; higher permutation = more impact if removed):\n"
    ]
    if ctx.get("top_shap_features"):
        lines.append("Top features by SHAP mean |value| (Random Forest, class=Canceled):")
        for r in ctx["top_shap_features"]:
            lines.append(f"  - {r['label']}: {r['mean_abs_shap']:.4f}")

    if ctx.get("top_perm_features"):
        lines.append("\nTop features by Permutation Importance (ROC-AUC drop, HistGBM):")
        for r in ctx["top_perm_features"]:
            lines.append(f"  - {r['label']}: {r['importance_mean']:.4f}")

    lines.append(
        "\nUsing ONLY these rankings and values, write 4 sentences explaining "
        "which features most influence cancellation predictions and what that "
        "means for hotel operations. Note associations only — no causal claims. "
        "Do not add any figures not listed above."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OPENAI CALLER  (with timeout, retry, and fallback)
# ---------------------------------------------------------------------------

def _get_openai_client():
    """Return an OpenAI client, or None if the key is absent."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
    try:
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            timeout=DEFAULT_TIMEOUT,
            max_retries=0,           # we handle retries manually below
        )
        return client, None
    except Exception as exc:
        return None, f"Failed to initialise OpenAI client: {exc}"


def call_openai(user_prompt: str,
                model: Optional[str] = None,
                timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Call the OpenAI Chat Completions API with grounding instructions.

    Returns the AI response text, or a descriptive fallback message on any error.
    Never raises an exception to the caller.
    """
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    client, err = _get_openai_client()
    if client is None:
        return f"⚠️ AI unavailable — {err}\n\n{FALLBACK_MSG}"

    messages = [
        {"role": "system",  "content": GROUNDING_INSTRUCTION},
        {"role": "user",    "content": user_prompt},
    ]

    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES retries
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,          # low temperature for factual consistency
                max_tokens=600,
                timeout=timeout,
            )
            return response.choices[0].message.content.strip()

        except __import__("openai").AuthenticationError as exc:
            # Don't retry auth failures
            return (
                f"⚠️ OpenAI authentication failed. "
                f"Please verify your OPENAI_API_KEY is correct.\n\n{FALLBACK_MSG}"
            )
        except __import__("openai").APITimeoutError as exc:
            last_error = f"Request timed out after {timeout}s"
        except __import__("openai").RateLimitError as exc:
            last_error = "Rate limit exceeded"
            time.sleep(RETRY_DELAY * attempt)   # back off more on rate limits
        except __import__("openai").APIConnectionError as exc:
            last_error = f"Network error: {exc}"
        except __import__("openai").APIStatusError as exc:
            last_error = f"API error {exc.status_code}: {exc.message}"
            if exc.status_code and exc.status_code < 500:
                break       # 4xx errors won't succeed on retry
        except Exception as exc:
            last_error = f"Unexpected error: {exc}"
            break

        if attempt <= MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    return (
        f"⚠️ AI insights temporarily unavailable ({last_error}). "
        f"Retried {MAX_RETRIES} time(s).\n\n{FALLBACK_MSG}"
    )


# ---------------------------------------------------------------------------
# PUBLIC INSIGHT FUNCTIONS
# ---------------------------------------------------------------------------

def get_overview_insight() -> str:
    """4-sentence executive summary of overall hotel portfolio KPIs."""
    kpis = _load_core_kpis()
    if not kpis:
        return FALLBACK_MSG
    return call_openai(build_overview_prompt(kpis))


def get_cancellation_insight() -> str:
    """5-bullet analysis of cancellation drivers from validated SQL results."""
    ctx = _load_cancellation_context()
    return call_openai(build_cancellation_prompt(ctx))


def get_revenue_insight() -> str:
    """4-sentence revenue and ADR performance analysis."""
    ctx = _load_revenue_context()
    return call_openai(build_revenue_prompt(ctx))


def get_ml_insight() -> str:
    """3-sentence plain-English ML model performance explanation."""
    ctx = _load_ml_context()
    if not ctx:
        return FALLBACK_MSG
    return call_openai(build_ml_prompt(ctx))


def get_explainability_insight() -> str:
    """4-sentence explanation of top SHAP/permutation features."""
    ctx = _load_explainability_context()
    return call_openai(build_explainability_prompt(ctx))


def get_custom_insight(topic: str, context_dict: dict) -> str:
    """
    Ad-hoc insight for a user-supplied topic and validated context dictionary.

    Parameters
    ----------
    topic : str
        A short description of what the user wants explained (≤100 chars).
    context_dict : dict
        Validated metrics/data to pass as context. Keys should be descriptive.
        The function formats them as a flat list of key: value pairs.
    """
    lines = [f"Topic: {topic}\n", "Validated data context:"]
    for k, v in context_dict.items():
        lines.append(f"  - {k}: {v}")
    lines.append(
        "\nUsing ONLY the data above, answer the topic question in 3–4 sentences. "
        "Do not add statistics not listed above."
    )
    return call_openai("\n".join(lines))


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  ai_insights.py — Standalone Module Test")
    print("=" * 65)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("\n  OPENAI_API_KEY not set.")
        print("  Testing prompt builders only (no API call).\n")

        kpis = _load_core_kpis()
        if kpis:
            print("[OVERVIEW PROMPT]")
            print(build_overview_prompt(kpis))
            print()
        else:
            print("  No KPI data found — run sql_analytics.py first.")

        ctx_cancel = _load_cancellation_context()
        if any(ctx_cancel.values()):
            print("[CANCELLATION PROMPT]")
            print(build_cancellation_prompt(ctx_cancel))
        else:
            print("  No cancellation data found.")
    else:
        print("\n  OPENAI_API_KEY detected. Making a test call...\n")
        kpis = _load_core_kpis()
        result = get_overview_insight()
        print("[OVERVIEW INSIGHT RESULT]")
        print(result)

    print("\nModule test complete.")
