# Hotel Bookings Analytics Platform
**End-to-End Data Analytics & Machine Learning Project Report**

**Author:** chirag2005dv  
**GitHub:** [github.com/chirag2005dv/hotel-bookings-analytics-platform](https://github.com/chirag2005dv/hotel-bookings-analytics-platform)  
**License:** MIT  
**Tech Stack:** Python 3.10+ | Streamlit 1.63 | Scikit-Learn 1.7.2 | OpenAI GPT-4o-mini | SQLite | SHAP 0.49.1 | pytest 8.4.2  

---

## 1. Executive Summary
This report documents the Hotel Bookings Analytics Platform — a complete, nine-phase end-to-end data analytics and machine learning project built on the publicly available Hotel Booking Demand dataset (Antonio, Almeida & Nunes, 2019). The project transforms raw booking records into production-ready insights: data cleaning, exploratory analysis, SQL analytics, machine learning, model explainability, an interactive Streamlit dashboard, and AI-powered natural language insights via GPT-4o-mini.

**Key outcomes achieved:**
* **Dataset:** 118,564 clean bookings (from 119,390 raw) across July 2015 – August 2017
* **Core KPI:** 37.26% overall cancellation rate; City Hotel 41.9% vs Resort Hotel 28.0%
* **Best ML Model:** HistGradientBoosting — ROC-AUC 0.9157, F1 0.7832, Accuracy 83.6%
* **Top Features:** Deposit Type (#1 SHAP/Perm/MDI), Lead Time (#2), Special Requests (#3)
* **Revenue:** $25.99M total estimated revenue; Online TA drives 52.8% of revenue
* **Dashboard:** 7-tab Streamlit dashboard with real-time ML prediction and GPT-4o-mini insights
* **Testing:** 73 automated pytest tests across 3 test files

---

## 2. Project Overview

### 2.1 Objectives
The primary business objective is to predict hotel booking cancellations (`is_canceled`) at booking time — before check-in — using only features available at booking creation, enabling hotel operators to take proactive action. Secondary objectives include:
* Understand revenue trends and ADR patterns across hotel types, years, and segments
* Identify high-risk customer segments for targeted retention strategies
* Surface actionable operational insights through 20 annotated SQL queries
* Provide hotel managers with an interactive self-service analytics dashboard
* Explain model predictions in plain language using SHAP and AI narration

### 2.2 Pipeline Architecture
| Phase | Name | Description |
| :--- | :--- | :--- |
| **Phase 1** | Project Setup | Repository initialisation, dependency pinning, `.env` configuration |
| **Phase 2** | Data Cleaning | NULL handling, dtype casting, invalid row removal, outlier flagging, feature engineering |
| **Phase 3** | EDA & KPIs | 33 charts across 8 analytical sections, structured KPI report |
| **Phase 4** | SQL Analytics | 20 annotated SQL queries via SQLite, cross-validated against Phase 3 |
| **Phase 5** | Machine Learning | Binary classification pipeline: 3 models, cross-validation, evaluation |
| **Phase 6** | Explainability | SHAP TreeExplainer, Permutation Importance, Partial Dependence Plots |
| **Phase 7** | Dashboard | 7-tab Streamlit dashboard with global sidebar filters and Plotly charts |
| **Phase 8** | AI Insights | GPT-4o-mini integration with data-grounded natural language analysis |
| **Phase 9** | Testing | 73 automated pytest tests; synthetic in-memory data, no network calls |

---

## 3. Dataset Description
The Hotel Booking Demand dataset contains one row per hotel booking, covering a City Hotel and a Resort Hotel in Portugal.

| Attribute | Value |
| :--- | :--- |
| **Source File** | `hotel_bookings.csv` |
| **Publication** | Antonio, Almeida & Nunes (2019) — ScienceDirect / Kaggle |
| **Raw Rows / Cols** | 119,390 rows × 32 columns |
| **Cleaned Rows / Cols** | 118,564 rows × 42 columns (including 7 engineered features) |
| **Target Variable** | `is_canceled` (0 = Not Canceled, 1 = Canceled) |
| **Class Balance** | 74,388 Not Canceled (62.7%) \| 44,176 Canceled (37.3%) |

### 3.1 Engineered Features (Phase 2 Additions)
* `total_nights`: `stays_in_weekend_nights + stays_in_week_nights`
* `arrival_date`: Parsed from year + month + day_of_month
* `room_type_match`: 1 if reserved_room_type == assigned_room_type
* `is_high_season`: 1 if arrival month is June, July, or August
* `revenue_estimate`: `adr * total_nights`
* `total_guests`: `adults + children + babies`
* `arrival_month_num`: Numeric month 1-12 for time-series ordering

---

## 4. File Structure
The project is organized into a professional folder hierarchy:
```text
datasets/
├── src/                # Pipeline scripts (data_cleaning, eda, sql, ml, dashboard)
├── audits/             # Phase-specific audit and validation scripts
├── data/               # Raw and processed datasets (gitignored)
├── sql/                # Annotated business queries (.sql)
├── reports/            # Output logs, KPI summaries, markdown decisions
├── models/             # Serialised ML models (.pkl)
├── plots/              # 33 EDA and ML evaluation chart images (PNG)
├── screenshots/        # Dashboard UI screenshots
├── sql_results/        # CSV outputs from all 20 SQL queries
├── tests/              # pytest suite (73 tests)
├── pytest.ini          # Test configuration
├── requirements.txt    # Pinned dependencies
└── README.md           # Main project documentation
```

---

## 5. Phase 2 — Data Cleaning & Preprocessing
*Script: `src/data_cleaning.py` | Docs: `reports/PREPROCESSING_DECISIONS.md`*

The cleaning pipeline is a 10-step, sequential, idempotent process:
1. **NULL Strings:** 129,421 occurrences of literal `'NULL'` strings converted to true NaN.
2. **Missing Values:** `country` (Unknown), `agent/company` (0), `children` (0).
3. **Invalid Rows:** Removed zero-night (715), zero-guest (110), and negative-ADR (1) bookings.
4. **Leakage Columns:** `reservation_status`, `reservation_status_date`, `revenue_estimate`, and engineered flags were carefully excluded from ML training sets.

---

## 6. Phase 3 — Exploratory Data Analysis & KPIs
*Script: `src/eda_analysis.py` | Output: `reports/kpi_report.txt` | Charts: `plots/`*

### 6.1 Core KPIs
* **Overall Cancellation Rate:** 37.26%
* **Average Daily Rate (ADR):** $101.01 (City Hotel: $106.62 | Resort Hotel: $92.01)
* **Total Revenue Estimate:** $25,986,976 (Checked-out bookings only)
* **Average Lead Time:** 104.5 days (Canceled: 145.0d | Not Canceled: 80.5d)

### 6.2 Cancellation Profiles
* **By Deposit Type:** Non Refund (99.4%), No Deposit (28.6%), Refundable (22.2%)
* **By Lead Time:** Monotonic increase from 11.0% (0-7 days) to 67.7% (365+ days)
* **By Segment:** Groups (61.2%), Online TA (36.9%), Direct (15.5%)

---

## 7. Phase 4 — SQL Analytics
*Query Library: `sql/hotel_queries.sql` | Runner: `src/sql_analytics.py`*

Executed 20 annotated business queries via SQLite (in-memory) yielding key insights:
* **Q04 (Deposits):** Non-Refund deposits have a 99.4% cancel rate — the strongest single-feature predictor in the dataset.
* **Q07 (Segments):** Online TA drives 47.3% of volume and **52.8% of total revenue**.
* **Q10 (Trends):** ADR grew 24.7% from 2015 ($89.84) to 2017 ($112.02).
* **Q14 (Loyalty):** Repeat guests cancel 2.4x less (15.7% vs 37.9%) and make more special requests.

---

## 8. Phase 5 — Machine Learning Pipeline
*Script: `src/ml_pipeline.py` | Report: `reports/ml_report.txt`*

* **Task:** Binary classification to predict `is_canceled` at booking time.
* **Features:** 31 features (22 numeric scaled, 9 categorical encoded) after removing 7 leakage variables.
* **Best Model:** HistGradientBoostingClassifier (ROC-AUC **0.9157**, Accuracy **83.6%**)
* **Top 5 Features (MDI):** Deposit Type (20.9%), Lead Time (12.9%), Special Requests (10.0%), Market Segment (7.9%), Room Type Match (7.2%).

---

## 9. Phase 6 — Model Explainability
*Script: `src/explainability.py`*

Applied three methods (SHAP TreeExplainer, Permutation Importance, MDI) to establish a consensus ranking of feature importance:
1. **Deposit Type:** Ranks #1 across all methods. (Note: Non-Refundable rates reflect pricing policy, not direct causation).
2. **Lead Time:** Ranks #2 across all methods. Longer lead time = higher probability of cancellation.
3. **Total Special Requests:** Ranks #3 across all methods. Acts as a strong negative indicator (more requests = less likely to cancel).

---

## 10. Phase 7 & 8 — Dashboard & AI Insights
*Script: `src/dashboard.py` & `src/ai_insights.py`*

A comprehensive 7-tab Streamlit dashboard:
1. **Overview & KPI Panels:** Interactive metrics and charts filtered globally by Hotel, Year, and Segment.
2. **Cancellation & Revenue Tabs:** Deep-dives into booking profiles and ADR trends.
3. **ML Prediction:** Live form generating a real-time cancellation probability using the trained HistGBM model.
4. **Explainability:** Interactive SHAP visualizations (Beeswarm, Summary, Dependence).
5. **AI Insights (GPT-4o-mini):** Natural language narration dynamically grounded ONLY to validated SQL and ML reports (zero hallucination design).

---

## 11. Phase 9 — Testing
*Framework: `pytest 8.4.2` | Tests: `tests/`*

* **73 automated tests** covering data cleaning, SQL logic, ML evaluation integrity, and AI context assembly.
* Uses entirely in-memory synthetic data mockups to prevent side-effects on production artifacts.

---

## 12. Limitations & Caveats
1. **Deposit Artefact:** 99.4% cancel rate on non-refundable rates likely reflects platform (OTA) behaviors/policies, requiring business validation before relying heavily on this feature for intervention.
2. **Partial Date Range:** The dataset lacks full calendar coverage for 2015 and 2017, introducing slight seasonal bias into YoY comparisons.
3. **Identity Linkage:** True guest-level tracking is unavailable; repeat guest statistics rely purely on the PMS flag.

---

## 13. Setup & Usage Guide

```bash
# Clone and Setup
git clone https://github.com/chirag2005dv/hotel-bookings-analytics-platform.git
pip install -r requirements.txt

# (Optional) Add OpenAI key to .env for AI insights
copy .env.example .env

# Run Pipeline Phases
python src/data_cleaning.py
python src/eda_analysis.py
python src/sql_analytics.py
python src/ml_pipeline.py
python src/explainability.py

# Launch Dashboard
streamlit run src/dashboard.py

# Run Tests
python -m pytest tests/ -v
```

---

## 14. Conclusion
The Hotel Bookings Analytics Platform demonstrates a production-quality data science workflow on a real-world hospitality dataset. Through 118,564 clean records, 33 charts, 20 queries, and robust ML (ROC-AUC 0.9157), the project highlights *Deposit Type*, *Lead Time*, and *Special Requests* as the strongest predictors of cancellation.

The modular architecture (clean → explore → query → model → explain → dashboard) provides hotel operators with an intuitive, interactive tool to assess booking risks dynamically, backed by a fully tested and extensible codebase.

---

**References:**
* Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. Data in Brief.
* Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.
* Scikit-Learn, Streamlit, and OpenAI documentation.
