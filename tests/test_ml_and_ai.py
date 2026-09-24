# tests/test_ml_and_ai.py
"""
Tests for the ML pipeline artifacts and ai_insights.py module.

ML tests operate on the pre-trained models already saved to models/ —
they do not re-train. AI tests exercise prompt builders and fallback
logic without making real API calls.
"""

import os
import pathlib
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODELS_DIR = pathlib.Path("models")
SQL_DIR    = pathlib.Path("sql_results")

NUMERIC_FEATURES = [
    "lead_time", "arrival_date_year", "arrival_month_num",
    "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "total_nights", "adults", "children", "babies", "total_guests",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes",
    "days_in_waiting_list", "adr", "required_car_parking_spaces",
    "total_of_special_requests", "room_type_match", "is_high_season",
]
CATEGORICAL_FEATURES = [
    "hotel", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "reserved_room_type", "assigned_room_type",
    "deposit_type", "customer_type",
]


def _encode_and_build_X(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals and build feature matrix exactly as ml_pipeline.py does."""
    from sklearn.preprocessing import LabelEncoder
    full_df = pd.read_csv("hotel_bookings_cleaned.csv")  # for fitting encoders
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        le.fit(full_df[col].astype(str))
        df[col + "_enc"] = le.transform(df[col].astype(str))
    feature_names = (MODELS_DIR / "feature_names.txt").read_text().splitlines()
    df["children"] = df["children"].fillna(0)
    return df[feature_names]


# ---------------------------------------------------------------------------
# ML Artifact tests
# ---------------------------------------------------------------------------

class TestModelArtifacts:
    def test_model_files_exist(self):
        for fname in ["best_model.pkl", "random_forest.pkl",
                      "logistic_regression.pkl", "gradient_boosting.pkl"]:
            assert (MODELS_DIR / fname).exists(), f"Missing: {fname}"

    def test_feature_names_file_exists(self):
        assert (MODELS_DIR / "feature_names.txt").exists()

    def test_feature_names_count(self):
        names = (MODELS_DIR / "feature_names.txt").read_text().splitlines()
        assert len(names) == 31, f"Expected 31 features, got {len(names)}"

    def test_shap_importance_csv_exists(self):
        assert (MODELS_DIR / "shap_feature_importance.csv").exists()

    def test_permutation_importance_csv_exists(self):
        assert (MODELS_DIR / "permutation_importance.csv").exists()

    def test_feature_ranking_comparison_exists(self):
        assert (MODELS_DIR / "feature_ranking_comparison.csv").exists()


class TestBestModelPrediction:
    @pytest.fixture(autouse=True)
    def load_model(self):
        import joblib
        self.model = joblib.load(MODELS_DIR / "best_model.pkl")

    def test_model_loads_without_error(self):
        assert self.model is not None

    def test_predict_proba_shape(self, clean_df):
        X = _encode_and_build_X(clean_df.copy())
        proba = self.model.predict_proba(X)
        assert proba.shape == (len(clean_df), 2), \
            f"Expected shape ({len(clean_df)}, 2), got {proba.shape}"

    def test_probabilities_in_range(self, clean_df):
        X = _encode_and_build_X(clean_df.copy())
        proba = self.model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all(), \
            "All probabilities must be in [0, 1]"

    def test_probabilities_sum_to_one(self, clean_df):
        X = _encode_and_build_X(clean_df.copy())
        proba = self.model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6,
                                   err_msg="Probabilities must sum to 1 per row")

    def test_canceled_booking_higher_prob(self, clean_df):
        """A booking with Non Refund deposit and long lead time should
        have a higher cancel probability than a same-week Direct booking."""
        full_df = pd.read_csv("hotel_bookings_cleaned.csv")
        # High-risk row: Non Refund, 400-day lead time, Groups
        high_risk = full_df[
            (full_df["deposit_type"] == "Non Refund") &
            (full_df["lead_time"] > 200)
        ].head(1).copy()
        # Low-risk row: No Deposit, 0–7 day lead time, Direct
        low_risk = full_df[
            (full_df["deposit_type"] == "No Deposit") &
            (full_df["lead_time"] <= 7) &
            (full_df["market_segment"] == "Direct") &
            (full_df["is_canceled"] == 0)
        ].head(1).copy()

        if high_risk.empty or low_risk.empty:
            pytest.skip("Insufficient rows for risk comparison")

        prob_high = self.model.predict_proba(_encode_and_build_X(high_risk))[0, 1]
        prob_low  = self.model.predict_proba(_encode_and_build_X(low_risk))[0, 1]
        assert prob_high > prob_low, \
            f"High-risk booking ({prob_high:.3f}) should have higher cancel prob " \
            f"than low-risk ({prob_low:.3f})"

    def test_benchmark_roc_auc(self):
        """Verify the saved model achieves ROC-AUC ≥ 0.90 on the real test set."""
        import joblib
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        df = pd.read_csv("hotel_bookings_cleaned.csv")
        X = _encode_and_build_X(df.copy())
        y = df["is_canceled"]
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        model = joblib.load(MODELS_DIR / "best_model.pkl")
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        assert auc >= 0.90, f"Best model ROC-AUC {auc:.4f} below 0.90 threshold"


# ---------------------------------------------------------------------------
# SHAP / Permutation importance sanity
# ---------------------------------------------------------------------------

class TestExplainabilityArtifacts:
    def test_top_shap_feature_is_deposit_type(self):
        df = pd.read_csv(MODELS_DIR / "shap_feature_importance.csv")
        top = df.sort_values("mean_abs_shap", ascending=False).iloc[0]
        assert "deposit" in top["feature"].lower(), \
            f"Expected deposit_type as top SHAP feature, got {top['feature']}"

    def test_shap_values_all_non_negative(self):
        df = pd.read_csv(MODELS_DIR / "shap_feature_importance.csv")
        assert (df["mean_abs_shap"] >= 0).all(), \
            "Mean |SHAP| values must be non-negative"

    def test_permutation_top_feature_is_deposit_type(self):
        df = pd.read_csv(MODELS_DIR / "permutation_importance.csv")
        top = df.sort_values("importance_mean", ascending=False).iloc[0]
        assert "deposit" in top["label"].lower() or "deposit" in top["feature"].lower(), \
            f"Expected Deposit Type as top permutation feature, got {top.to_dict()}"

    def test_ranking_comparison_has_31_features(self):
        df = pd.read_csv(MODELS_DIR / "feature_ranking_comparison.csv")
        assert len(df) == 31, f"Expected 31 features in ranking comparison, got {len(df)}"

    def test_plot_files_exist(self):
        plots = pathlib.Path("plots")
        for fname in [
            "EX_shap_summary_bar.png",
            "EX_shap_summary_beeswarm.png",
            "EX_permutation_importance.png",
            "EX_partial_dependence.png",
            "EX_shap_waterfall_cancel.png",
            "EX_shap_waterfall_nocancel.png",
            "EX_method_comparison_heatmap.png",
        ]:
            assert (plots / fname).exists(), f"Missing explainability plot: {fname}"


# ---------------------------------------------------------------------------
# AI Insights module — prompt builders and fallback
# ---------------------------------------------------------------------------

class TestAiInsightsPromptBuilders:
    def test_overview_prompt_contains_total_bookings(self):
        from ai_insights import build_overview_prompt, _load_core_kpis
        kpis = _load_core_kpis()
        prompt = build_overview_prompt(kpis)
        assert "118,564" in prompt, "Overview prompt must contain total bookings 118,564"

    def test_overview_prompt_contains_cancellation_rate(self):
        from ai_insights import build_overview_prompt, _load_core_kpis
        kpis = _load_core_kpis()
        prompt = build_overview_prompt(kpis)
        assert "37." in prompt, "Overview prompt must contain cancellation rate ~37%"

    def test_cancellation_prompt_contains_all_hotel_types(self):
        from ai_insights import build_cancellation_prompt, _load_cancellation_context
        ctx = _load_cancellation_context()
        prompt = build_cancellation_prompt(ctx)
        assert "City Hotel" in prompt
        assert "Resort Hotel" in prompt

    def test_cancellation_prompt_contains_deposit_types(self):
        from ai_insights import build_cancellation_prompt, _load_cancellation_context
        ctx = _load_cancellation_context()
        prompt = build_cancellation_prompt(ctx)
        assert "Non Refund" in prompt
        assert "No Deposit" in prompt

    def test_ml_prompt_contains_best_model_name(self):
        from ai_insights import build_ml_prompt, _load_ml_context
        ctx = _load_ml_context()
        prompt = build_ml_prompt(ctx)
        assert "Gradient Boosting" in prompt

    def test_ml_prompt_contains_roc_auc(self):
        from ai_insights import build_ml_prompt, _load_ml_context
        ctx = _load_ml_context()
        prompt = build_ml_prompt(ctx)
        assert "0.9157" in prompt

    def test_explainability_prompt_contains_shap_features(self):
        from ai_insights import build_explainability_prompt, _load_explainability_context
        ctx = _load_explainability_context()
        prompt = build_explainability_prompt(ctx)
        assert "Deposit Type" in prompt or "deposit" in prompt.lower()

    def test_revenue_prompt_contains_segment_data(self):
        from ai_insights import build_revenue_prompt, _load_revenue_context
        ctx = _load_revenue_context()
        prompt = build_revenue_prompt(ctx)
        assert "Online TA" in prompt or "revenue" in prompt.lower()


class TestAiInsightsFallback:
    def test_fallback_when_no_api_key(self, monkeypatch):
        """With no API key, all insight functions must return a warning string."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from ai_insights import get_overview_insight
        result = get_overview_insight()
        assert result.startswith("\u26a0"), \
            "Without API key, result should start with ⚠️"

    def test_fallback_does_not_raise(self, monkeypatch):
        """The fallback path must never raise an exception."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from ai_insights import (get_overview_insight, get_cancellation_insight,
                                  get_revenue_insight, get_ml_insight,
                                  get_explainability_insight)
        for fn in [get_overview_insight, get_cancellation_insight,
                   get_revenue_insight, get_ml_insight, get_explainability_insight]:
            result = fn()   # must not raise
            assert isinstance(result, str)

    def test_custom_insight_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from ai_insights import get_custom_insight
        result = get_custom_insight("test topic", {"key": "value"})
        assert isinstance(result, str)
        assert result.startswith("\u26a0")

    def test_kpi_data_matches_validated_values(self):
        """KPIs loaded from SQL results must match Phase 3 validated figures."""
        from ai_insights import _load_core_kpis
        kpis = _load_core_kpis()
        assert kpis["total_bookings"]   == 118564
        assert kpis["total_canceled"]   == 44176
        assert abs(kpis["cancellation_rate_pct"] - 37.26) < 0.01
        assert abs(kpis["avg_adr_stayed"] - 101.01) < 0.01
        assert abs(kpis["total_revenue_estimate"] - 25986976.03) < 1.0

    def test_grounding_instruction_present(self):
        """The GROUNDING_INSTRUCTION must forbid fabrication."""
        from ai_insights import GROUNDING_INSTRUCTION
        text = GROUNDING_INSTRUCTION.lower()
        assert "only" in text
        assert "not" in text or "do not" in text
