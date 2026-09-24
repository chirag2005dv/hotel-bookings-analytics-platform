from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

ROOT = Path(__file__).parent

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)

def set_col_width(table, col_idx, width_cm):
    for row in table.rows:
        row.cells[col_idx].width = Cm(width_cm)

def add_rule(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bot = OxmlElement("w:bottom")
    bot.set(qn("w:val"), "single"); bot.set(qn("w:sz"), "6")
    bot.set(qn("w:space"), "1"); bot.set(qn("w:color"), "2563EB")
    pBdr.append(bot); pPr.append(pBdr)

def h(doc, text, level=1, color=(0x1D,0x35,0x57)):
    hd = doc.add_heading(text, level=level)
    hd.paragraph_format.space_before = Pt(14 if level==1 else 8)
    hd.paragraph_format.space_after = Pt(6)
    for run in hd.runs:
        run.font.color.rgb = RGBColor(*color)
    return hd

def p(doc, text, bold=False, italic=False, size=11, color=None, align=None, sa=6):
    par = doc.add_paragraph()
    par.paragraph_format.space_after = Pt(sa)
    run = par.add_run(text)
    run.bold = bold; run.italic = italic; run.font.size = Pt(size)
    if color: run.font.color.rgb = RGBColor(*color)
    if align: par.alignment = align
    return par

def tbl(doc, headers, rows, hbg="1D3557", alt="EBF2FB", widths=None):
    t = doc.add_table(rows=1+len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hcells = t.rows[0].cells
    for i, hdr in enumerate(headers):
        hcells[i].text = hdr
        set_cell_bg(hcells[i], hbg)
        for r in hcells[i].paragraphs[0].runs:
            r.bold = True; r.font.color.rgb = RGBColor(0xFF,0xFF,0xFF); r.font.size = Pt(9.5)
        hcells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for ri, row_data in enumerate(rows):
        cells = t.rows[ri+1].cells
        bg = alt if ri%2==1 else "FFFFFF"
        for ci, val in enumerate(row_data):
            cells[ci].text = str(val)
            set_cell_bg(cells[ci], bg)
            for r in cells[ci].paragraphs[0].runs:
                r.font.size = Pt(9.5)
    if widths:
        for ci, w in enumerate(widths):
            set_col_width(t, ci, w)
    doc.add_paragraph()
    return t

def bl(doc, text, prefix=None):
    par = doc.add_paragraph(style="List Bullet")
    par.paragraph_format.space_after = Pt(3)
    if prefix:
        r = par.add_run(prefix); r.bold = True; r.font.size = Pt(10.5)
        par.add_run(text).font.size = Pt(10.5)
    else:
        par.add_run(text).font.size = Pt(10.5)
    return par

# ─── BUILD DOCUMENT ──────────────────────────────────────────────────────
doc = Document()
sec = doc.sections[0]
sec.top_margin=Cm(2.2); sec.bottom_margin=Cm(2.2)
sec.left_margin=Cm(2.5); sec.right_margin=Cm(2.5)
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)

# ── TITLE PAGE ──────────────────────────────────────────────────────────
for _ in range(3): doc.add_paragraph()
tp = doc.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tr = tp.add_run("Hotel Bookings Analytics Platform")
tr.bold=True; tr.font.size=Pt(28); tr.font.color.rgb=RGBColor(0x1D,0x35,0x57)
doc.add_paragraph()
sp = doc.add_paragraph(); sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sr = sp.add_run("End-to-End Data Analytics & Machine Learning Project Report")
sr.font.size=Pt(16); sr.font.color.rgb=RGBColor(0x25,0x63,0xEB); sr.italic=True
for _ in range(2): doc.add_paragraph()
for line, sz in [
    ("Author: chirag2005dv", 13),
    ("GitHub: github.com/chirag2005dv/hotel-bookings-analytics-platform", 11),
    (f"Date: {datetime.date.today().strftime('%B %d, %Y')}", 11),
    ("License: MIT", 11),
]:
    lp = doc.add_paragraph(); lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lr = lp.add_run(line); lr.font.size=Pt(sz); lr.font.color.rgb=RGBColor(0x44,0x44,0x44)
for _ in range(2): doc.add_paragraph()
bp = doc.add_paragraph(); bp.alignment = WD_ALIGN_PARAGRAPH.CENTER
br = bp.add_run("Tech Stack: Python 3.10+  |  Streamlit 1.63  |  Scikit-Learn 1.7.2  |  OpenAI GPT-4o-mini  |  SQLite  |  SHAP 0.49.1  |  pytest 8.4.2")
br.font.size=Pt(10); br.font.color.rgb=RGBColor(0x55,0x55,0x55)
doc.add_page_break()

# ── 1. EXECUTIVE SUMMARY ─────────────────────────────────────────────────
h(doc,"1. Executive Summary"); add_rule(doc)
p(doc,"This report documents the Hotel Bookings Analytics Platform — a complete, nine-phase end-to-end data analytics and machine learning project built on the publicly available Hotel Booking Demand dataset (Antonio, Almeida & Nunes, 2019). The project transforms raw booking records into production-ready insights: data cleaning, exploratory analysis, SQL analytics, machine learning, model explainability, an interactive Streamlit dashboard, and AI-powered natural language insights via GPT-4o-mini.")
p(doc,"Key outcomes achieved:", bold=True)
bl(doc,"118,564 clean bookings (from 119,390 raw) across July 2015 – August 2017", prefix="Dataset: ")
bl(doc,"37.26% overall cancellation rate; City Hotel 41.9% vs Resort Hotel 28.0%", prefix="Core KPI: ")
bl(doc,"HistGradientBoosting — ROC-AUC 0.9157, F1 0.7832, Accuracy 83.6%", prefix="Best ML Model: ")
bl(doc,"Deposit Type (#1 SHAP/Perm/MDI), Lead Time (#2), Special Requests (#3)", prefix="Top Features: ")
bl(doc,"$25.99M total estimated revenue; Online TA drives 52.8% of revenue", prefix="Revenue: ")
bl(doc,"7-tab Streamlit dashboard with real-time ML prediction and GPT-4o-mini insights", prefix="Dashboard: ")
bl(doc,"73 automated pytest tests across 3 test files", prefix="Testing: ")
doc.add_paragraph()

# ── 2. PROJECT OVERVIEW ──────────────────────────────────────────────────
h(doc,"2. Project Overview"); add_rule(doc)
h(doc,"2.1 Objectives", 2)
p(doc,"The primary business objective is to predict hotel booking cancellations (is_canceled) at booking time — before check-in — using only features available at booking creation, enabling hotel operators to take proactive action. Secondary objectives include:")
for obj in [
    "Understand revenue trends and ADR patterns across hotel types, years, and segments",
    "Identify high-risk customer segments for targeted retention strategies",
    "Surface actionable operational insights through 20 annotated SQL queries",
    "Provide hotel managers with an interactive self-service analytics dashboard",
    "Explain model predictions in plain language using SHAP and AI narration",
]: bl(doc, obj)

h(doc,"2.2 Pipeline Architecture", 2)
tbl(doc,["Phase","Name","Description"],[
    ("Phase 1","Project Setup","Repository initialisation, dependency pinning, .env configuration"),
    ("Phase 2","Data Cleaning","NULL handling, dtype casting, invalid row removal, outlier flagging, feature engineering"),
    ("Phase 3","EDA & KPIs","33 charts across 8 analytical sections, structured KPI report"),
    ("Phase 4","SQL Analytics","20 annotated SQL queries via SQLite, cross-validated against Phase 3"),
    ("Phase 5","Machine Learning","Binary classification pipeline: 3 models, cross-validation, evaluation"),
    ("Phase 6","Explainability","SHAP TreeExplainer, Permutation Importance, Partial Dependence Plots"),
    ("Phase 7","Dashboard","7-tab Streamlit dashboard with global sidebar filters and Plotly charts"),
    ("Phase 8","AI Insights","GPT-4o-mini integration with data-grounded natural language analysis"),
    ("Phase 9","Testing","73 automated pytest tests; synthetic in-memory data, no network calls"),
], widths=[2.0,3.5,10.5])

h(doc,"2.3 Technology Stack", 2)
tbl(doc,["Library / Tool","Purpose"],[
    ("Python 3.10+","Core language"),
    ("pandas 2.3.3 / numpy 2.2.6","Data manipulation and numerical computing"),
    ("matplotlib / seaborn","Static chart generation (EDA & ML evaluation)"),
    ("SQLite (sqlite3)","In-memory SQL analytics engine"),
    ("scikit-learn 1.7.2","ML pipeline: preprocessing, models, evaluation metrics"),
    ("SHAP 0.49.1","Model explainability: TreeExplainer, waterfall, beeswarm plots"),
    ("Streamlit 1.63","Interactive web dashboard framework"),
    ("Plotly","Interactive charts within the Streamlit dashboard"),
    ("OpenAI API (gpt-4o-mini)","AI-powered natural language insights generation"),
    ("joblib","Model serialisation/deserialisation"),
    ("pytest 8.4.2","Automated test framework"),
    ("python-docx","This report generation"),
], widths=[5.5,10.5])
doc.add_page_break()

# ── 3. DATASET ───────────────────────────────────────────────────────────
h(doc,"3. Dataset Description"); add_rule(doc)
p(doc,"The Hotel Booking Demand dataset (Antonio, Almeida & Nunes, 2019) contains one row per hotel booking, covering a City Hotel and a Resort Hotel in Portugal, published on ScienceDirect and Kaggle.")
tbl(doc,["Attribute","Value"],[
    ("Source File","hotel_bookings.csv"),
    ("Publication","Antonio, Almeida & Nunes (2019) — ScienceDirect / Kaggle"),
    ("Raw Rows","119,390"),("Raw Columns","32"),
    ("Cleaned Rows","118,564"),("Cleaned Columns","42 (including 7 engineered features)"),
    ("Date Range","July 1, 2015 – August 31, 2017"),
    ("City Hotel bookings","78,899 (66.5%)"),("Resort Hotel bookings","39,665 (33.5%)"),
    ("Target Variable","is_canceled  (0 = Not Canceled,  1 = Canceled)"),
    ("Class Balance","74,388 Not Canceled (62.7%)  |  44,176 Canceled (37.3%)"),
], widths=[5.5,10.5])

h(doc,"3.1 Original Feature Set (32 columns)", 2)
tbl(doc,["Feature","Type","Description"],[
    ("hotel","category","Hotel type: City Hotel or Resort Hotel"),
    ("is_canceled","Int64","Target variable: 1 = booking was canceled"),
    ("lead_time","Int64","Days between booking creation and arrival"),
    ("arrival_date_year / month / week / day","Int64/category","Arrival date components"),
    ("stays_in_weekend_nights","Int64","Weekend nights booked (Sat/Sun)"),
    ("stays_in_week_nights","Int64","Weekday nights booked (Mon–Fri)"),
    ("adults / children / babies","Int64/float64","Guest composition counts"),
    ("meal","category","Meal plan (BB, HB, FB, SC)"),
    ("country","category","Guest country of origin (ISO 3166-1 alpha-3)"),
    ("market_segment","category","Booking channel group (Online TA, Direct, Groups, etc.)"),
    ("distribution_channel","category","Distribution method (TA/TO, Direct, Corporate, etc.)"),
    ("is_repeated_guest","Int64","1 = returning guest"),
    ("previous_cancellations","Int64","Number of prior cancellations by the guest"),
    ("previous_bookings_not_canceled","Int64","Prior successful bookings"),
    ("reserved_room_type","category","Room type requested at booking time"),
    ("assigned_room_type","category","Room type actually assigned at check-in"),
    ("booking_changes","Int64","Number of changes made to the booking"),
    ("deposit_type","category","Deposit policy (No Deposit, Non Refund, Refundable)"),
    ("agent","category","Travel agent ID (NULL = no agent)"),
    ("company","category","Company booking ID (NULL = individual)"),
    ("days_in_waiting_list","Int64","Days on waiting list before confirmation"),
    ("customer_type","category","Booking type (Transient, Contract, Group, Transient-Party)"),
    ("adr","float64","Average Daily Rate in euros"),
    ("required_car_parking_spaces","Int64","Parking spaces requested"),
    ("total_of_special_requests","Int64","Count of special requests submitted"),
    ("reservation_status","category","Final status (Check-Out, Canceled, No-Show)"),
    ("reservation_status_date","datetime64","Date of last status change"),
], widths=[4.5,3.0,8.5])

h(doc,"3.2 Engineered Features (Phase 2 additions)", 2)
tbl(doc,["Feature","Type","Formula / Definition"],[
    ("total_nights","Int64","stays_in_weekend_nights + stays_in_week_nights"),
    ("arrival_date","datetime64","Parsed from year + month + day_of_month (0 parse failures)"),
    ("room_type_match","Int64","1 if reserved_room_type == assigned_room_type"),
    ("is_high_season","Int64","1 if arrival month is June, July, or August"),
    ("revenue_estimate","float64","adr x total_nights (proxy for room revenue in euros)"),
    ("total_guests","Int64","adults + children + babies"),
    ("arrival_month_num","Int64","Numeric month 1-12 for time-series ordering"),
], widths=[4.5,3.0,8.5])
doc.add_page_break()

# ── 4. FILE STRUCTURE ────────────────────────────────────────────────────
h(doc,"4. Project File Structure"); add_rule(doc)
p(doc,"The project is organized into a professional folder hierarchy separating source code, data, SQL queries, reports, models, charts, and tests:")
tbl(doc,["File / Folder","Location","Purpose"],[
    ("README.md","root","Full project documentation with embedded screenshots"),
    ("requirements.txt","root","Pinned Python dependencies (pip install -r requirements.txt)"),
    (".env.example","root","Template for OpenAI API key configuration"),
    ("pytest.ini","root","pytest config — sets pythonpath = src for all tests"),
    ("LICENSE","root","MIT License"),
    ("src/data_cleaning.py","src/","Phase 2: 10-step data cleaning pipeline"),
    ("src/eda_analysis.py","src/","Phase 3: EDA charts and KPI computation (33 plots)"),
    ("src/sql_analytics.py","src/","Phase 4: SQL query runner and cross-validator"),
    ("src/ml_pipeline.py","src/","Phase 5: ML training, evaluation, model serialisation"),
    ("src/explainability.py","src/","Phase 6: SHAP, permutation importance, PDP"),
    ("src/dashboard.py","src/","Phase 7+8: 7-tab Streamlit interactive dashboard"),
    ("src/ai_insights.py","src/","Phase 8: GPT-4o-mini grounded insights module"),
    ("src/validate_cleaned.py","src/","Quick validation script for the cleaned dataset"),
    ("audits/audit2-7_*.py","audits/","Six per-phase automated audit/validation scripts"),
    ("data/raw/hotel_bookings.csv","data/raw/","Original dataset (~16 MB, gitignored)"),
    ("data/processed/hotel_bookings_cleaned.csv","data/processed/","Cleaned dataset (~20 MB, gitignored)"),
    ("sql/hotel_queries.sql","sql/","20 annotated SQL analytical queries"),
    ("reports/cleaning_report.txt","reports/","Audit log from Phase 2 cleaning pipeline"),
    ("reports/kpi_report.txt","reports/","KPI output from Phase 3 EDA analysis"),
    ("reports/sql_results_report.txt","reports/","Full SQL results report from Phase 4"),
    ("reports/ml_report.txt","reports/","ML model evaluation report from Phase 5"),
    ("reports/explainability_report.txt","reports/","Explainability findings from Phase 6"),
    ("reports/PREPROCESSING_DECISIONS.md","reports/","Documented rationale for all cleaning decisions"),
    ("models/","models/","Serialised model .pkl files and feature_names.txt (gitignored)"),
    ("plots/","plots/","33 EDA and ML evaluation chart images (PNG, 150 DPI)"),
    ("screenshots/","screenshots/","14 dashboard UI screenshot images (PNG)"),
    ("sql_results/","sql_results/","20 SQL query result CSV files"),
    ("tests/","tests/","pytest suite: 73 tests across 3 files + conftest.py"),
], widths=[5.0,2.8,8.2])
doc.add_page_break()

# ── 5. DATA CLEANING ─────────────────────────────────────────────────────
h(doc,"5. Phase 2 — Data Cleaning & Preprocessing"); add_rule(doc)
p(doc,"Script: src/data_cleaning.py  |  Report: reports/cleaning_report.txt  |  Docs: reports/PREPROCESSING_DECISIONS.md", italic=True, size=10, color=(0x55,0x55,0x55))
p(doc,"The cleaning pipeline is a 10-step, sequential, idempotent process. Every design decision is justified in PREPROCESSING_DECISIONS.md. The pipeline reads the raw CSV, applies all transformations in-memory, and writes the cleaned output.")
tbl(doc,["Step","Action","Outcome"],[
    ("Step 1","Load Raw Data","119,390 rows x 32 columns loaded from hotel_bookings.csv"),
    ("Step 2","Schema Validation","All 32 expected columns verified; pipeline aborts on schema mismatch"),
    ("Step 3","Replace NULL Strings","129,421 literal 'NULL' strings -> real NaN; 0 empty strings found"),
    ("Step 4","Type Casting","32 columns cast: Int64, float64, category, datetime64[ns]"),
    ("Step 5","Handle Missing Values","country 488->Unknown; agent/company 128,933->0; children 4->0"),
    ("Step 6","Invalid Row Removal","826 rows removed: 715 zero-night, 110 zero-guest, 1 negative-ADR"),
    ("Step 7","Outlier Flagging","3 binary flags added non-destructively: adr_outlier, long_lead, zero_adr"),
    ("Step 8","Feature Engineering","7 new columns: total_nights, arrival_date, room_type_match, is_high_season, revenue_estimate, total_guests, arrival_month_num"),
    ("Step 9","Final Validation","Zero nulls; shape=118,564x42; ADR/lead_time/nights all non-negative"),
    ("Step 10","Save Output","hotel_bookings_cleaned.csv: 19.83 MB, 118,564 rows x 42 columns"),
], widths=[1.8,4.0,10.2])

h(doc,"5.1 Key Cleaning Decisions", 2)
bl(doc,"Source uses literal 'NULL' strings (not empty cells). All 129,421 occurrences converted to NaN before type casting to prevent silent dtype errors.", prefix="NULL Strings: ")
bl(doc,"Encoded as '0' (no agent/no company) rather than NaN — preserves numeric encoding compatibility for ML pipeline.", prefix="Agent / Company: ")
bl(doc,"Zero-night (715) and zero-guest (110) bookings are physically impossible hotel stays. One negative-ADR row removed as data error.", prefix="Invalid Rows: ")
bl(doc,"ADR > 5,400 threshold flagged but retained — zero rows exceeded cap. The ADR=5,400 outlier (53x mean) is a valid canceled booking.", prefix="ADR Outliers: ")
bl(doc,"reservation_status, reservation_status_date, revenue_estimate, and three flag columns excluded from ML as post-booking leakage.", prefix="Leakage Columns: ")
bl(doc,"room_type_match may reflect post-booking assignment. Flagged in PREPROCESSING_DECISIONS.md as a production risk.", prefix="Room Type Match: ")
doc.add_page_break()

# ── 6. EDA & KPIs ────────────────────────────────────────────────────────
h(doc,"6. Phase 3 — Exploratory Data Analysis & KPIs"); add_rule(doc)
p(doc,"Script: src/eda_analysis.py  |  Report: reports/kpi_report.txt  |  33 charts saved to plots/", italic=True, size=10, color=(0x55,0x55,0x55))
p(doc,"Phase 3 produces 33 publication-quality charts across 8 analytical sections and generates a structured KPI report. All charts are saved at 150 DPI as PNG files and later embedded in the Streamlit dashboard.")

h(doc,"6.1 Core KPIs", 2)
tbl(doc,["KPI","Value","Notes"],[
    ("Overall Cancellation Rate","37.26%","44,176 of 118,564 bookings canceled"),
    ("Average Daily Rate (ADR)","$101.01","City Hotel: $106.62  |  Resort Hotel: $92.01  |  Median: $93.39"),
    ("Total Revenue Estimate","$25,986,976","ADR x total_nights for checked-out bookings only"),
    ("Avg Revenue per Booking","$349.34","Across all non-canceled bookings"),
    ("Average Lead Time","104.5 days","Canceled: 145.0d  |  Not Canceled: 80.5d"),
    ("Average Length of Stay","3.42 nights","0.94 weekend nights + 2.48 week nights"),
    ("Repeat Guest Rate","2.95%","3,494 of 118,564 bookings from returning guests"),
    ("Room Type Match Rate","87.81%","104,058 bookings received their requested room type"),
    ("Avg Special Requests / Booking","0.57","Range: 0-5 per booking"),
    ("Peak Month (Volume)","August","Lowest volume month: January"),
], widths=[5.0,3.5,7.5])

h(doc,"6.2 Cancellation Analysis", 2)
p(doc,"By Hotel Type:", bold=True)
tbl(doc,["Hotel Type","Bookings","Canceled","Cancellation Rate"],[
    ("City Hotel","78,899","33,066","41.9%"),
    ("Resort Hotel","39,665","11,110","28.0%"),
], widths=[4.5,3.5,3.5,4.5])

p(doc,"By Deposit Type:", bold=True)
tbl(doc,["Deposit Type","Bookings","Cancellation Rate"],[
    ("Non Refund","14,587","99.4%"),
    ("No Deposit","103,815","28.6%"),
    ("Refundable","162","22.2%"),
], widths=[5.5,4.5,6.0])

p(doc,"By Lead Time Bucket:", bold=True)
tbl(doc,["Lead Time Bucket","Bookings","Cancellation Rate"],[
    ("0-7 days","13,239","11.0%"),
    ("8-30 days","18,848","28.0%"),
    ("31-90 days","29,437","37.8%"),
    ("91-180 days","26,384","44.8%"),
    ("181-365 days","21,505","55.5%"),
    ("365+ days","3,144","67.7%"),
], widths=[5.5,4.5,6.0])

p(doc,"By Market Segment:", bold=True)
tbl(doc,["Market Segment","Bookings","Cancellation Rate"],[
    ("Groups","19,758","61.2%"),
    ("Online TA","56,129","36.9%"),
    ("Offline TA/TO","24,049","34.5%"),
    ("Aviation","231","22.1%"),
    ("Corporate","5,231","18.9%"),
    ("Direct","12,449","15.5%"),
    ("Complementary","715","12.2%"),
], widths=[5.5,4.5,6.0])

h(doc,"6.3 EDA Chart Groups (33 total)", 2)
tbl(doc,["Charts","Section","Content"],[
    ("C1-C4","Cancellation Analysis","By hotel type, market segment, deposit type, lead time bucket"),
    ("D1-D4","Revenue & ADR","Monthly ADR by hotel, ADR distribution, ADR by segment, monthly revenue estimate"),
    ("E1-E3","Booking Channels","Volume by segment, distribution channel mix, segment cancel-rate vs ADR scatter"),
    ("F1-F3","Booking Behaviour","Lead time distribution, cancellation by booking changes, by special requests"),
    ("G1-G3","Seasonality & Time","Monthly bookings + cancel rate, yearly trends, hotel x month cancel heatmap"),
    ("H1-H4","Guest Profile","Top 15 countries, length of stay distribution, meal plan mix, new vs repeat guests"),
    ("ML1-ML7","ML Evaluation","ROC curves, confusion matrices, feature importance bars, calibration curve, PR curve"),
], widths=[2.5,4.0,9.5])
doc.add_page_break()

# ── 7. SQL ANALYTICS ─────────────────────────────────────────────────────
h(doc,"7. Phase 4 — SQL Analytics"); add_rule(doc)
p(doc,"Query Library: sql/hotel_queries.sql  |  Runner: src/sql_analytics.py  |  Engine: SQLite (in-memory)", italic=True, size=10, color=(0x55,0x55,0x55))
p(doc,"Phase 4 loads the cleaned dataset into an in-memory SQLite database, executes 20 annotated business queries, saves each result to sql_results/ as a CSV, generates a comprehensive text report, and cross-validates Q01 KPI values against Phase 3 (zero difference confirmed).")

tbl(doc,["Query","Title","Business Question"],[
    ("Q01","Core KPI Snapshot","Single-row KPI summary: cancel rate, ADR, revenue, lead time, repeat rate"),
    ("Q02","Cancellation by Hotel Type","Cancel rate and volume for City vs Resort Hotel"),
    ("Q03","Cancellation by Market Segment","Cancel rate, ADR, revenue contribution per segment"),
    ("Q04","Cancellation by Deposit Type","Non Refund vs No Deposit vs Refundable cancel profiles"),
    ("Q05","Cancellation by Lead Time Bucket","Monotonic lead time -> cancel rate (6 buckets: 0-7d to 365d+)"),
    ("Q06","Cancellation by Customer Type","Transient, Contract, Group, Transient-Party comparison"),
    ("Q07","Revenue & ADR by Segment","Revenue share %, ADR, and volume per market segment"),
    ("Q08","Monthly ADR by Hotel","Month-by-month ADR for City Hotel and Resort Hotel"),
    ("Q09","Monthly Booking Volume","Total bookings per calendar month (all years aggregated)"),
    ("Q10","Year-over-Year Trends","Annual revenue, ADR, bookings, cancel rate: 2015-2017"),
    ("Q11","Top 15 Countries","Countries ranked by booking volume with cancel rate"),
    ("Q12","Distribution Channel Performance","Channel ADR, revenue share, and cancellation rate"),
    ("Q13","Length of Stay Analysis","Average total nights by hotel type and market segment"),
    ("Q14","Repeat vs New Guest","Cancel rate, ADR, special requests: returning vs new guests"),
    ("Q15","Room Assignment Analysis","Match rate and ADR gap between matched and upgraded rooms"),
    ("Q16","Deposit Revenue Impact","Revenue by deposit type for checked-out bookings only"),
    ("Q17","Lead Time vs Revenue","Revenue and ADR by lead time bucket (checked-out only)"),
    ("Q18","Special Requests Impact","Cancel rate and ADR by special request count (0-5+)"),
    ("Q19","Year x Month Heatmap Grid","Booking volume grid: month columns x year rows"),
    ("Q20","High-Value Booking Profile","Top 10% by revenue: avg nights, ADR, lead time, cancel rate"),
], widths=[1.5,5.0,9.5])

h(doc,"7.1 Key SQL Findings", 2)
bl(doc,"Non-Refund deposits: 99.4% cancel rate — strongest single-feature predictor in the entire dataset", prefix="Q04: ")
bl(doc,"Cancellation rate rises monotonically from 11.0% (0-7 days) to 67.7% (365+ days)", prefix="Q05: ")
bl(doc,"Groups: 61.2% cancellation, $77 ADR — highest risk, lowest revenue per booking of all segments", prefix="Q03: ")
bl(doc,"Online TA: 47.3% of all bookings, 52.8% of total revenue at a 36.9% cancel rate", prefix="Q07: ")
bl(doc,"ADR grew 24.7%: $89.84 (2015) -> $101.00 (2016) -> $112.02 (2017)", prefix="Q10: ")
bl(doc,"Repeat guests cancel 2.4x less (15.7% vs 37.9%), make 2.1 avg special requests vs 0.55", prefix="Q14: ")
bl(doc,"Top 10% bookings (>=$711 revenue): avg 7.6 nights, $159 ADR", prefix="Q20: ")
bl(doc,"Q01 KPIs match Phase 3 EDA values exactly within floating-point tolerance (cross-validation passed)", prefix="Validation: ")
doc.add_page_break()

# ── 8. MACHINE LEARNING ──────────────────────────────────────────────────
h(doc,"8. Phase 5 — Machine Learning Pipeline"); add_rule(doc)
p(doc,"Script: src/ml_pipeline.py  |  Report: reports/ml_report.txt", italic=True, size=10, color=(0x55,0x55,0x55))
p(doc,"Phase 5 builds a binary classification pipeline to predict is_canceled at booking time. The pipeline performs feature selection, leakage removal, label encoding, standard scaling, model training with 5-fold cross-validation, full evaluation, and model serialisation for dashboard use.")

h(doc,"8.1 Experimental Setup", 2)
tbl(doc,["Parameter","Value"],[
    ("Task","Binary classification: predict is_canceled (0 or 1)"),
    ("Dataset","118,564 rows x 31 features (after leakage removal)"),
    ("Train / Test Split","80% / 20%  (stratified by target, random_state=42)"),
    ("Training Set","94,851 rows"),
    ("Test Set","23,713 rows"),
    ("Cross-Validation","StratifiedKFold(n_splits=5) on training set"),
    ("Primary Metric","ROC-AUC"),
    ("Secondary Metric","F1-Score (class=Canceled)"),
    ("Numeric Features","22 features (scaled with StandardScaler)"),
    ("Categorical Features","9 features (label-encoded with LabelEncoder)"),
], widths=[5.0,11.0])

h(doc,"8.2 Leakage Removal", 2)
p(doc,"The following 7 columns were excluded from the feature set as they contain post-booking information not available at booking creation time:")
for col in ["reservation_status","reservation_status_date","revenue_estimate","flag_adr_outlier","flag_long_lead_time","flag_zero_adr","arrival_date"]:
    bl(doc, col)

h(doc,"8.3 Model Comparison (Test Set)", 2)
tbl(doc,["Model","Accuracy","Precision","Recall","F1","ROC-AUC","CV ROC-AUC"],[
    ("Logistic Regression","0.7764","0.6884","0.7305","0.7088","0.8550","0.8534 +/- 0.0018"),
    ("Random Forest","0.8414","0.7941","0.7753","0.7846","0.9149","0.9126 +/- 0.0021"),
    ("Gradient Boosting (BEST)","0.8359","0.7712","0.7957","0.7832","0.9157","0.9136 +/- 0.0021"),
], widths=[4.5,2.2,2.2,2.2,2.0,2.2,3.5])
p(doc,"HistGradientBoostingClassifier selected as best model (highest ROC-AUC 0.9157). Saved to models/best_model.pkl.", italic=True, size=10)

h(doc,"8.4 Classification Report — Best Model (HistGBM)", 2)
tbl(doc,["Class","Precision","Recall","F1-Score","Support"],[
    ("Not Canceled (0)","0.88","0.86","0.87","14,878"),
    ("Canceled (1)","0.77","0.80","0.78","8,835"),
    ("Macro Average","0.82","0.83","0.83","23,713"),
    ("Weighted Average","0.84","0.84","0.84","23,713"),
], widths=[4.5,3.0,3.0,3.0,3.0])

h(doc,"8.5 Top 10 Feature Importances (Random Forest MDI)", 2)
tbl(doc,["Rank","Feature","Importance"],[
    ("1","Deposit Type","20.94%"),
    ("2","Lead Time","12.94%"),
    ("3","Total of Special Requests","10.05%"),
    ("4","Market Segment","7.91%"),
    ("5","Room Type Match","7.17%"),
    ("6","Previous Cancellations","5.85%"),
    ("7","Required Car Parking Spaces","5.34%"),
    ("8","Customer Type","4.08%"),
    ("9","Booking Changes","3.84%"),
    ("10","ADR (Average Daily Rate)","3.53%"),
], widths=[1.5,8.0,4.0])
doc.add_page_break()

# ── 9. EXPLAINABILITY ────────────────────────────────────────────────────
h(doc,"9. Phase 6 — Model Explainability"); add_rule(doc)
p(doc,"Script: src/explainability.py  |  Report: reports/explainability_report.txt  |  8 charts in plots/", italic=True, size=10, color=(0x55,0x55,0x55))
p(doc,"Phase 6 applies three complementary explainability techniques. The consensus ranking across all methods provides a robust, method-agnostic view of which features drive cancellation predictions.")

h(doc,"9.1 Explainability Methods", 2)
tbl(doc,["Method","Model Used","Output"],[
    ("SHAP TreeExplainer","Random Forest","Global beeswarm, bar, waterfall, and dependence plots; mean |SHAP| ranking"),
    ("Permutation Importance","HistGBM (best)","ROC-AUC drop per feature when randomly shuffled; robust to feature correlation"),
    ("Partial Dependence Plots","HistGBM (best)","Marginal effect of individual features on predicted cancellation probability"),
    ("MDI (Mean Decrease Impurity)","Random Forest","Built-in sklearn feature importance; fast but can be biased for high-cardinality features"),
], widths=[4.0,3.5,8.5])

h(doc,"9.2 Consensus Feature Ranking (all 3 methods)", 2)
tbl(doc,["Avg Rank","Feature","MDI","SHAP","Perm.","SHAP Mean |Value|","Perm. Importance"],[
    ("1.00","Deposit Type","1","1","1","0.0891","0.1088"),
    ("2.33","Lead Time","2","3","2","0.0593","0.0532"),
    ("2.67","Total of Special Requests","3","2","3","0.0806","0.0413"),
    ("5.33","Market Segment","4","4","8","0.0502","0.0290"),
    ("5.33","Room Type Match","5","5","6","0.0406","0.0331"),
    ("5.67","Required Car Parking Spaces","7","6","4","0.0300","0.0373"),
    ("6.33","Previous Cancellations","6","8","5","0.0287","0.0370"),
    ("7.33","Customer Type","8","7","7","0.0296","0.0326"),
    ("9.33","Booking Changes","9","9","10","0.0277","0.0099"),
    ("10.00","ADR","10","11","9","0.0151","0.0219"),
], widths=[2.0,4.5,1.5,1.5,1.5,3.0,3.0])

h(doc,"9.3 Key Feature Interpretations", 2)
p(doc,"IMPORTANT: All interpretations describe statistical associations, NOT causal relationships. Changing a feature value would not necessarily change cancellation behaviour.", bold=True, size=10.5, color=(0xCC,0x44,0x00))
doc.add_paragraph()
interps = [
    ("Deposit Type (#1 all methods)","Non-Refundable bookings associate with 99.4% cancellation. This likely reflects OTA pricing policy conventions (non-refundable rate codes) rather than guest intent. The model uses this as the strongest signal, but changing a booking to 'No Deposit' would NOT necessarily prevent cancellation."),
    ("Lead Time (#2)","Monotonically positive relationship confirmed by PDP: longer gap between booking and arrival -> higher predicted cancellation. Guests booking far in advance have more uncertain plans, but shortening the booking window cannot be used as a direct intervention."),
    ("Total Special Requests (#3)","Negatively associated with cancellation: guests with 1+ requests cancel at 22% vs 48% with zero requests (EDA). SHAP PDP shows steep drop at 1+ requests. This reflects committed guest behaviour, not a lever to manipulate."),
    ("Market Segment (#4)","Groups (61.2%) and Online TA (36.9%) push predictions toward cancellation; Direct (15.5%) and Corporate (18.9%) push away. SHAP strongly discriminates between segments."),
    ("Previous Cancellations (#7)","Strongest behavioural signal: even 1 prior cancellation sharply increases predicted probability. Reflects genuine guest reliability history."),
    ("Required Car Parking (#6)","Guests who request parking cancel less — proxy for committed, specific-need travelers. Not a direct causal mechanism."),
]
for feat, interp in interps:
    bl(doc, interp, prefix=feat+": ")

h(doc,"9.4 Charts Produced", 2)
tbl(doc,["File","Type","Description"],[
    ("EX_shap_summary_bar.png","SHAP Bar","Mean absolute SHAP values, top 20 features"),
    ("EX_shap_summary_beeswarm.png","SHAP Beeswarm","Feature impact distribution across 2,000 test samples"),
    ("EX_shap_dependence_top4.png","SHAP Dependence (x4)","SHAP value vs feature value for top 4 features"),
    ("EX_shap_waterfall_cancel.png","SHAP Waterfall","Individual prediction for a representative canceled booking"),
    ("EX_shap_waterfall_nocancel.png","SHAP Waterfall","Individual prediction for a representative non-canceled booking"),
    ("EX_permutation_importance.png","Permutation Bar","Top 20 features by ROC-AUC drop (HistGBM)"),
    ("EX_partial_dependence.png","PDP (4 panels)","Marginal effect: deposit type, lead time, special requests, ADR"),
    ("EX_method_comparison_heatmap.png","Heatmap","Rank comparison across MDI vs SHAP vs Permutation"),
], widths=[5.0,3.0,8.0])
doc.add_page_break()

# ── 10. DASHBOARD & AI ───────────────────────────────────────────────────
h(doc,"10. Phase 7 & 8 — Dashboard & AI Insights"); add_rule(doc)
p(doc,"Script: src/dashboard.py  |  Launch: streamlit run src/dashboard.py  |  AI Module: src/ai_insights.py", italic=True, size=10, color=(0x55,0x55,0x55))
p(doc,"The Streamlit dashboard is a complete self-service analytics platform delivered in a web browser. It loads all pre-computed artefacts at startup (SQL CSVs, model pkl files, plot PNGs) and renders 7 specialised tabs with interactive Plotly charts, SHAP visualisations, a real-time ML prediction form, and GPT-powered natural language explanations.")

h(doc,"10.1 Dashboard Tabs", 2)
tbl(doc,["Tab","Content"],[
    ("1 — Overview","8 KPI metric tiles; hotel-type donut chart; year-over-year bar chart with breakdown table; active filter summary panel"),
    ("2 — Cancellation","Cancel rate by hotel, deposit type, lead time bucket, segment; monthly cancel heatmap; customer type scatter"),
    ("3 — Revenue & ADR","Monthly ADR trend by hotel; revenue by segment bar; ADR distribution histogram; lead time vs revenue scatter"),
    ("4 — Segments","Segment selector with live KPI panel; monthly volume by segment; country chart; full comparison table with color formatting"),
    ("5 — ML Prediction","Live cancellation probability input form (31 fields); HistGBM prediction result; gauge chart; explanation cards"),
    ("6 — Explainability","SHAP bar + beeswarm plots; dependence charts; waterfall plots; interpretation cards with prediction-vs-causation caveats"),
    ("7 — AI Insights","GPT-4o-mini analysis grounded to validated SQL/SHAP data; 6 insight types with toggle; 'View data being sent' transparency expander"),
], widths=[3.5,12.5])

h(doc,"10.2 Sidebar Global Filters", 2)
p(doc,"Three filters in the left sidebar propagate across Tabs 1-4 simultaneously:")
bl(doc,"Hotel Type: City Hotel / Resort Hotel / All Hotels")
bl(doc,"Arrival Year: 2015 / 2016 / 2017 / All Years")
bl(doc,"Market Segment: individual segment selection or All Segments")

h(doc,"10.3 AI Insights Module", 2)
p(doc,"The AI Insights module integrates OpenAI GPT-4o-mini to generate natural language analysis grounded strictly to validated data from SQL results, SHAP rankings, and the ML evaluation report — preventing hallucinated statistics.")
tbl(doc,["Insight Type","Grounded Data Sources","Focus"],[
    ("Overview Insight","Q01, Q10, Q02","KPIs, YoY revenue growth, hotel type comparison"),
    ("Cancellation Insight","Q02, Q04, Q05, Q03, Q14","Deposit type, lead time, segment, repeat guest patterns"),
    ("Revenue Insight","Q07, Q10, Q17, Q20","Segment revenue share, ADR trends, high-value profiles"),
    ("ML Insight","reports/ml_report.txt","Model performance, feature importances, classification metrics"),
    ("Explainability Insight","SHAP CSV + permutation CSV","Feature consensus ranking, SHAP interpretation"),
    ("Custom Insight","User-selected dataset","Any user question with injected data context"),
], widths=[4.0,4.5,7.5])

h(doc,"10.4 AI Safety & Reliability", 2)
bl(doc,"System prompt: 'Use ONLY the numbers and facts explicitly given in the context below'", prefix="Grounding: ")
bl(doc,"Explicitly forbids external benchmarks, fabricated statistics, and causal claims", prefix="Guardrails: ")
bl(doc,"temperature=0.2 (factual consistency mode); max_tokens=800", prefix="Model Settings: ")
bl(doc,"30s timeout, 2 retries with 2s delay, graceful fallback message on failure or missing key", prefix="Reliability: ")
bl(doc,"All data sent to the API displayed in 'View data being sent' expander for full user transparency", prefix="Transparency: ")
doc.add_page_break()

# ── 11. TESTING ──────────────────────────────────────────────────────────
h(doc,"11. Phase 9 — Testing"); add_rule(doc)
p(doc,"Framework: pytest 8.4.2  |  Run: python -m pytest tests/ -v  |  Expected: 73 passed", italic=True, size=10, color=(0x55,0x55,0x55))
p(doc,"All 73 tests use synthetic in-memory data — no file I/O, no network calls, no modification of production artefacts. The suite verifies functional correctness of every pipeline module independently.")
tbl(doc,["Test File","Tests","Coverage Areas"],[
    ("test_data_cleaning.py","27","Schema validation; NULL string replacement; dtype casting; missing value handling; invalid row removal (zero-night, zero-guest, negative ADR); outlier flagging; all 7 engineered features; final validation checks"),
    ("test_sql_analytics.py","13","SQL file parsing; query block extraction and naming; query execution against synthetic SQLite; result shape validation; cross-validation function; revenue share summation; Q01 KPI cross-validation"),
    ("test_ml_and_ai.py","33","Model artifact existence (pkl files, feature_names.txt); prediction shape and probability range; SHAP + permutation feature ranking integrity; AI prompt construction; grounding instruction presence; fallback on missing key/data; KPI cross-validation"),
], widths=[4.5,1.5,10.0])

h(doc,"11.1 Test Design Principles", 2)
bl(doc,"All fixtures in tests/conftest.py; zero disk reads or network calls during test execution", prefix="Isolation: ")
bl(doc,"12-row synthetic DataFrame mirroring the exact 32-column schema of hotel_bookings.csv", prefix="Synthetic Data: ")
bl(doc,"conftest.py inserts src/ into sys.path; pytest.ini sets pythonpath=src globally", prefix="Import Hygiene: ")
bl(doc,"All 73 tests pass in a clean virtualenv after: pip install -r requirements.txt", prefix="CI Readiness: ")
doc.add_page_break()

# ── 12. KEY RESULTS ──────────────────────────────────────────────────────
h(doc,"12. Key Results & Findings Summary"); add_rule(doc)
tbl(doc,["Finding","Value","Source / Evidence"],[
    ("Overall cancellation rate","37.26%","Q01 KPI snapshot, kpi_report.txt"),
    ("Non-Refund deposit cancel rate","99.4%","Q04; SHAP/Perm/MDI rank #1"),
    ("Lead time cancellation range","11.0% to 67.7%","Q05 (0-7 days to 365+ days)"),
    ("City Hotel vs Resort cancel delta","41.9% vs 28.0%","Q02, kpi_report.txt"),
    ("Groups segment profile","61.2% cancel, $77 ADR","Q03, Q07"),
    ("Online TA revenue dominance","52.8% of total revenue","Q07"),
    ("ADR growth 2015-2017","+24.7% ($89.84 to $112.02)","Q10"),
    ("Repeat guest advantage","15.7% vs 37.9% cancel rate","Q14"),
    ("High-value booking profile (top 10%)",">=$ 711 revenue, 7.6 nights avg","Q20"),
    ("Best ML model","HistGBM: ROC-AUC 0.9157, F1 0.7832","ml_report.txt"),
    ("Feature consensus #1","Deposit Type — #1 in MDI, SHAP, Permutation","explainability_report.txt"),
    ("Special requests protective effect","22% cancel (>=1 req) vs 48% (0 req)","EDA, SHAP PDP"),
    ("Room type match rate","87.81% got requested room type","kpi_report.txt"),
    ("Total revenue estimate","$25,986,976","Q07, kpi_report.txt"),
    ("SQL vs EDA cross-validation","Zero difference — all KPIs match","sql_analytics.py validator"),
    ("Test suite result","73 / 73 tests passed","pytest tests/"),
], widths=[5.5,4.0,6.5])
doc.add_page_break()

# ── 13. LIMITATIONS ──────────────────────────────────────────────────────
h(doc,"13. Limitations & Caveats"); add_rule(doc)
limitations = [
    ("Partial Date Range","2015 covers July-December only; 2017 covers January-August only. Year-over-year comparisons may carry seasonal bias and should be interpreted with care."),
    ("Non-Refund Deposit Artefact","The 99.4% cancel rate for Non-Refund deposits likely reflects OTA pricing conventions rather than true guest intent. Validate before production deployment."),
    ("Static 2015-2017 Training Data","Post-COVID travel behaviour, OTA platform evolution, and demand patterns may have shifted significantly, reducing generalisability to current bookings."),
    ("No Guest-Level Identity","Repeat guest detection relies on the is_repeated_guest flag provided by the source PMS. No cross-booking identity linkage is possible without a unique guest ID."),
    ("Revenue is an Estimate","revenue_estimate = ADR x total_nights covers room revenue only. F&B, spa, parking, and other ancillary revenues are excluded."),
    ("AI Hallucination Risk","Despite temperature=0.2 and strict grounding prompts, LLM hallucinations remain possible. Always verify AI output against the 'View data being sent' expander."),
    ("Room Type Match Post-Booking","assigned_room_type is set at check-in, making room_type_match partially post-booking. Review before production ML deployment."),
    ("Class Imbalance Not Addressed","The 62.7% / 37.3% split is moderate. Cost-sensitive learning and probability threshold tuning were not explored in this phase."),
]
for i, (title, desc) in enumerate(limitations, 1):
    bl(doc, desc, prefix=f"{i}. {title}: ")
doc.add_paragraph()

# ── 14. SETUP ────────────────────────────────────────────────────────────
h(doc,"14. Setup & Usage Guide"); add_rule(doc)
h(doc,"14.1 Prerequisites", 2)
bl(doc,"Python 3.10 or higher"); bl(doc,"pip (bundled with Python)")
bl(doc,"OpenAI API key (optional — only required for AI Insights tab in the dashboard)")
h(doc,"14.2 Install & Run", 2)
tbl(doc,["Command","What It Does"],[
    ("git clone https://github.com/chirag2005dv/hotel-bookings-analytics-platform.git","Clone the repository"),
    ("pip install -r requirements.txt","Install all pinned dependencies"),
    ("copy .env.example .env  # then add your key","Configure optional OpenAI API key"),
    ("python src/data_cleaning.py","Phase 2: clean raw data -> data/processed/"),
    ("python src/eda_analysis.py","Phase 3: generate 33 charts + KPI report"),
    ("python src/sql_analytics.py","Phase 4: run 20 SQL queries + cross-validate"),
    ("python src/ml_pipeline.py","Phase 5: train models + save to models/"),
    ("python src/explainability.py","Phase 6: SHAP + permutation + PDP charts"),
    ("streamlit run src/dashboard.py","Phase 7+8: launch interactive dashboard"),
    ("python -m pytest tests/ -v","Phase 9: run full test suite (73 tests)"),
], widths=[8.5,7.5])
doc.add_page_break()

# ── 15. CONCLUSION ───────────────────────────────────────────────────────
h(doc,"15. Conclusion"); add_rule(doc)
p(doc,"The Hotel Bookings Analytics Platform demonstrates a complete, production-quality data science workflow applied to a real-world hospitality dataset. Starting from 119,390 raw booking records, the project delivers: a fully cleaned 118,564-row dataset, 33 exploratory charts, 20 validated SQL queries, a machine learning model achieving ROC-AUC 0.9157, comprehensive explainability via three independent methods, a seven-tab interactive dashboard, and AI-powered natural language insights — all backed by 73 automated tests.")
p(doc,"The most significant analytical finding is that deposit_type (Non-Refund vs No Deposit) is the strongest predictor of cancellation across all three explainability methods, with Non-Refund bookings canceling at 99.4%. Lead time and special requests are second and third respectively. The consensus across methods — MDI, SHAP TreeExplainer, and Permutation Importance — strengthens confidence in this ranking beyond what any single method could provide.")
p(doc,"These findings provide hotel operators with a clear priority order for intervention: monitor long-lead-time bookings proactively, incentivise special requests as commitment signals, and treat Groups and Online TA as highest-risk segments. Causal validation remains essential before operational deployment.")
p(doc,"The project is modular, reproducible, and extensible. Each of the nine pipeline phases is an independent, idempotent script. The codebase is fully version-controlled, documented, and tested, providing a solid foundation for future work including temporal retraining, production API deployment, and causal inference analysis.")
doc.add_paragraph()

h(doc,"15.1 Future Work", 2)
bl(doc,"Retrain on post-2020 data to improve temporal generalisability", prefix="Temporal Update: ")
bl(doc,"Cost-sensitive learning and probability threshold tuning for improved minority-class recall", prefix="Class Imbalance: ")
bl(doc,"LIME or Anchor explanations as local interpretability alternatives to SHAP", prefix="Explainability: ")
bl(doc,"FastAPI REST endpoint for real-time cancellation scoring from the trained model", prefix="API Deployment: ")
bl(doc,"DoWhy / EconML causal inference to estimate true intervention effects", prefix="Causal Analysis: ")
bl(doc,"Prophet or LSTM forecasting on the monthly ADR and booking volume time series", prefix="Forecasting: ")
doc.add_paragraph()

# ── 16. REFERENCES ───────────────────────────────────────────────────────
h(doc,"16. References"); add_rule(doc)
refs = [
    "Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126",
    "Kaggle Dataset: Hotel Booking Demand — Jesse Mostipak. https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 2017. https://arxiv.org/abs/1705.07874",
    "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
    "Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS 2017.",
    "Scikit-learn: Machine Learning in Python — Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.",
    "Streamlit Open-Source Framework — https://streamlit.io",
    "OpenAI API Documentation — https://platform.openai.com/docs",
    "GitHub Repository: https://github.com/chirag2005dv/hotel-bookings-analytics-platform",
]
for ref in refs:
    par = doc.add_paragraph(style="List Number")
    par.paragraph_format.space_after = Pt(4)
    par.add_run(ref).font.size = Pt(10)

doc.add_paragraph()
add_rule(doc)
fp = doc.add_paragraph(); fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = fp.add_run(f"Hotel Bookings Analytics Platform  |  chirag2005dv  |  {datetime.date.today().year}  |  MIT License")
fr.font.size = Pt(9); fr.font.color.rgb = RGBColor(0x88,0x88,0x88); fr.italic = True

# ── SAVE ─────────────────────────────────────────────────────────────────
out = ROOT / "Hotel_Bookings_Analytics_Platform_Report.docx"
doc.save(str(out))
print(f"\n  Report saved: {out}")
print(f"  Sections: 16  |  Tables: 30+  |  Approx. pages: 35-40")
