import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.loader import load_excel
from src.extract_features import (
    extract_main_features,
    extract_ec_features,
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DSE Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CSS — Adaptive Dark / Light
# Menggunakan CSS custom properties yang di-override
# via [data-theme="dark"] / [data-theme="light"] (Streamlit ≥ 1.35)
# dan @media prefers-color-scheme sebagai fallback.
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ══════════════════════════════════════════
   TOKEN DEFINITIONS
   Light = default; dark overrides below
══════════════════════════════════════════ */
:root {
    --accent:        #ff8a00;
    --accent2:       #a855f7;

    --accent-dim:    rgba(255,138,0,0.10);
    --accent-border: rgba(255,138,0,0.25);

    --bg-page:       #f7f7fa;
    --bg-card:       #ffffff;
    --bg-card2:      #f8f6fc;

    --border:        #ece8f3;
    --border-strong: #d8d2e5;

    --text-primary:  #1a1a1a;
    --text-secondary:#5b5b66;
    --text-muted:    #8f8f9d;

    --hero-bg:
        linear-gradient(
            135deg,
            rgba(255,138,0,0.08) 0%,
            rgba(255,77,184,0.08) 50%,
            rgba(168,85,247,0.08) 100%
        );

    --hero-border: rgba(255,138,0,0.18);

    --step-bg: rgba(255,255,255,0.7);
    --step-border: rgba(255,138,0,0.12);
    --step-text: #555;

    --section-line:
        linear-gradient(
            to right,
            rgba(255,138,0,0.3),
            transparent
        );

    --scrollbar-track: #f2f2f2;
    --scrollbar-thumb: #d8d2e5;
}

/* Dark mode via Streamlit attribute */
[data-theme="dark"] {
    --bg-page: #0f1016;
    --bg-card: #161824;
    --bg-card2: #1c2030;

    --border: #292d40;
    --border-strong: #3a3f59;

    --text-primary: #f3f4f6;
    --text-secondary: #b3b8c8;
    --text-muted: #7f879a;

    --hero-bg:
        linear-gradient(
            135deg,
            rgba(255,138,0,0.12) 0%,
            rgba(255,77,184,0.08) 50%,
            rgba(168,85,247,0.12) 100%
        );

    --hero-border: rgba(255,138,0,0.18);

    --step-bg: rgba(255,255,255,0.03);
    --step-border: rgba(255,255,255,0.05);
    --step-text: #b7bfd2;
}

/* System preference fallback */
@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
        --bg-page:       #0d0f14;
        --bg-card:       #12151d;
        --bg-card2:      #1a1e2a;
        --border:        #1e2535;
        --border-strong: #2e3a55;
        --text-primary:  #e8eaf0;
        --text-secondary:#7a8499;
        --text-muted:    #5c6680;
        --hero-bg:       linear-gradient(135deg, #0d1f2d 0%, #0d0f14 60%);
        --hero-border:   #1e3a4a;
        --step-bg:       rgba(255,255,255,0.03);
        --step-border:   rgba(255,255,255,0.07);
        --step-text:     #9ba5b8;
        --section-line:  linear-gradient(to right, #1e2535, transparent);
        --scrollbar-track: #0d0f14;
        --scrollbar-thumb: #1e2535;
    }
}

/* ══════════════════════════════════════════
   BASE
══════════════════════════════════════════ */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: var(--bg-page) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"]     { background: transparent !important; }
#MainMenu, footer            { visibility: hidden; }

            
/* ══════════════════════════════════════════
   HERO
══════════════════════════════════════════ */
.hero {
    background: var(--hero-bg);
    border: 1px solid var(--hero-border);

    border-radius: 24px;

    padding: 56px 64px;

    margin-bottom: 48px;

    box-shadow:
        0 10px 30px rgba(255,138,0,0.08),
        0 20px 60px rgba(168,85,247,0.05);
}
.hero::before {
    content: "";
    position: absolute;

    top: -80px;
    right: -80px;

    width: 400px;
    height: 400px;

    background:
        radial-gradient(
            circle,
            rgba(255,138,0,0.20) 0%,
            rgba(255,77,184,0.12) 35%,
            rgba(168,85,247,0.08) 55%,
            transparent 75%
        );

    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--accent-dim);
    border: 1px solid var(--accent-border);
    border-radius: 4px;
    padding: 4px 12px;
    margin-bottom: 20px;
}
.hero h1 {
    font-size: 42px;
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 12px 0;
    color: var(--text-primary);
    letter-spacing: -1px;
}
.hero h1 span {
    background:
        linear-gradient(
            90deg,
            #ff8a00,
            #ff4db8,
            #a855f7
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    font-size: 15px;
    color: var(--text-secondary);
    line-height: 1.7;
    max-width: 560px;
    margin: 0;
}

/* ── Steps ── */
.steps-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-top: 28px;
    margin-bottom: 10px;
    text-transform: uppercase;
}
.steps-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}
.step {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--step-bg);
    border: 1px solid var(--step-border);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 13px;
    color: var(--step-text);
}
.step-num {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 700;
    color: var(--accent);
    background: var(--accent-dim);
    border-radius: 4px;
    padding: 2px 8px;
    min-width: 24px;
    text-align: center;
}

/* ══════════════════════════════════════════
   SECTION HEADER
══════════════════════════════════════════ */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 40px 0 20px 0;
}
.section-line {
    flex: 1;
    height: 1px;
    background: var(--section-line);
}
.section-title {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
}

/* ══════════════════════════════════════════
   FILE UPLOADER
══════════════════════════════════════════ */
[data-testid="stFileUploaderDropzone"] {
    background: var(--accent-dim) !important;
    border: 2px dashed var(--accent-border) !important;
    border-radius: 12px !important;
    transition: all 0.2s ease;
    min-height: 220px !important;
    padding: 40px !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: rgba(14,168,130,0.06) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    justify-content: center !important;
    width: 100% !important;
    text-align: left !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    margin: 0 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    display: block !important;
    margin: 0 !important;
    text-align: left !important;
}
[data-testid="stFileUploader"] section > * {
    text-align: left !important;
}
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFileName"] {
    text-align: left !important;
    justify-content: flex-start !important;
}

/* ══════════════════════════════════════════
   METRIC CARDS
══════════════════════════════════════════ */
[data-testid="stMetric"] {
    background:
        linear-gradient(
            135deg,
            rgba(255,138,0,0.03),
            rgba(255,77,184,0.03),
            rgba(168,85,247,0.03)
        ) !important;

    border: 1px solid rgba(255,138,0,0.15) !important;

    border-radius: 18px !important;

    padding: 24px 28px !important;

    transition: all .25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: #ff8a00 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 34px !important;
    font-weight: 800 !important;
}
            
/* ══════════════════════════════════════════
   CHART CARDS
══════════════════════════════════════════ */

.chart-card {
    background: var(--bg-card);
    border: 1px solid rgba(255,138,0,0.12);
    border-radius: 20px;
    padding: 20px 16px 8px 16px;
}
.chart-title {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 0 0 4px 4px;
}
.chart-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    color: var(--text-muted);
    margin: 0 0 12px 4px;
}

/* ══════════════════════════════════════════
   ALERTS
══════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
    font-family: 'Syne', sans-serif !important;
}

/* ══════════════════════════════════════════
   DATAFRAME
══════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] thead tr th {
    background: var(--bg-card2) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted) !important;
}

            
/* ══════════════════════════════════════════
   DOWNLOAD BUTTON
══════════════════════════════════════════ */
[data-testid="stDownloadButton"] button {
    background: #ff8a00 !important;
    color: #ffffff !important;

    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;

    letter-spacing: 0.5px;

    border: none !important;
    border-radius: 8px !important;

    padding: 10px 24px !important;

    transition: all 0.2s ease !important;
}

[data-testid="stDownloadButton"] button:hover {
    background: #e67c00 !important;
}

/* ══════════════════════════════════════════
   DIVIDER & SCROLLBAR
══════════════════════════════════════════ */
hr {
    border-color: var(--border) !important;
    margin: 32px 0 !important;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--scrollbar-track); }
::-webkit-scrollbar-thumb { background: var(--scrollbar-thumb); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-strong); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero Section
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>DSE <span>Anomaly</span><br>Detection</h1>
    <p>
        Mengidentifikasi DSE yang menunjukkan indikasi ketidakwajaran pada aktivitas maupun pencapaian kinerja
        dari mayoritas DSE lainnya. Upload file Performance SLA DSE untuk memulai analisis.
    </p>
    <div class="steps-label">Cara Penggunaan</div>
    <div class="steps-row">
        <div class="step"><span class="step-num">1</span> Upload file Performance SLA DSE</div>
        <div class="step"><span class="step-num">2</span> Tunggu proses analisis</div>
        <div class="step"><span class="step-num">3</span> Tinjau hasil deteksi</div>
        <div class="step"><span class="step-num">4</span> Download laporan</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Upload Section
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-title">Upload Data</span>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & drop file di sini, atau klik untuk memilih",
    type=["xlsm", "xlsx"],
    help="Format yang didukung: .xlsm, .xlsx — Pastikan nama sheet dan kolom sesuai template."
)
st.caption("⚠️ Pastikan format file, nama sheet, dan nama kolom sesuai template. Tidak boleh ada typo pada data.")

# ─────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────
def generate_reason(row):

    reasons = []

    if row['N_HARI_KERJA'] < 10:
        reasons.append("Hari kerja/aktif relatif rendah")

    if row['EC_RATE'] < 0.3:
        reasons.append("EC rendah")

    if row['KPI_AVG_ACH'] < 50:
        reasons.append("KPI rendah")

    if row['BELOW_THRESHOLD_RATE'] > 0.7:
        reasons.append("Sering di bawah threshold")

    if row['SELLIN_PER_VISIT'] < 10000:
        reasons.append("Sellin per visit rendah")

    if row['EOM_CONCENTRATION'] > 0.4:
        reasons.append("Aktivitas menumpuk akhir bulan")

    if not reasons:
        reasons.append("Pola aktivitas berbeda dari mayoritas DSE")

    return ", ".join(reasons)

if uploaded_file is not None:

    with st.spinner("Menganalisis data DSE..."):

        # Load excel
        data = load_excel(uploaded_file)
        df_im3    = data["df_im3"]
        df_3id    = data["df_3id"]
        df_im3_ec = data["df_im3_ec"]
        df_3id_ec = data["df_3id_ec"]

        # Feature Engineering
        feat_im3_main = extract_main_features(df_im3, brand="IM3")
        feat_3id_main = extract_main_features(df_3id, brand="3ID")
        df_main_feat  = pd.concat([feat_im3_main, feat_3id_main], ignore_index=True)

        feat_im3_ec = extract_ec_features(df_im3_ec, brand="IM3")
        feat_3id_ec = extract_ec_features(df_3id_ec, brand="3ID")
        df_ec_feat  = pd.concat([feat_im3_ec, feat_3id_ec], ignore_index=True)

        # Merge
        df_feat = df_ec_feat.merge(
            df_main_feat.drop(columns=["BRAND_SRC"], errors="ignore"),
            on="DSE_ID",
            how="left")
        df_feat.columns = df_feat.columns.str.strip()

        # Fill missing
        num_cols = df_feat.select_dtypes(include=np.number).columns
        for col in num_cols:
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())

        # Load Model
        iso            = joblib.load("models/isolation_forest.pkl")
        scaler         = joblib.load("models/scaler.pkl")
        model_features = joblib.load("models/features.pkl")

        # Predict
        X        = df_feat[model_features]
        X_scaled = scaler.transform(X)

        df_feat["ANOMALY_SCORE"]  = -iso.decision_function(X_scaled)
        df_feat["IS_ANOMALY"]     = (iso.predict(X_scaled) == -1).astype(int)
        df_feat["ANOMALY_FLAG"]   = (df_feat["IS_ANOMALY"] == -1).astype(int)
        df_feat["ANOMALY_REASON"] = df_feat.apply(generate_reason, axis=1)
        df_feat["ANOMALY_REASON"] = df_feat.apply(
            lambda row: row["ANOMALY_REASON"] if row["IS_ANOMALY"] == 1 else "-", axis=1)
        df_feat["STATUS"] = df_feat["IS_ANOMALY"].map({1: "Anomali", 0: "Normal"})

        # ─────────────────────────────────────────
        # Quadrant Segmentation
        # ─────────────────────────────────────────
        KPI_COLS = [f for f in [
            'SP_ACH','OSA_ACH','FWA_ACH','KPI_AVG_ACH',
            'SP_PACE_GAP','OSA_PACE_GAP','FWA_PACE_GAP','KPI_AVG_ACH_MTD',
            'SCORE_SELLIN','CHAMP_TOTAL_MTD','COVERAGE_RATE'
        ] if f in df_feat.columns]

        EC_COLS = [f for f in [
            'EC_RATE','EC_TREND_WEEKLY','AVG_VISIT','VISIT_CV','ZERO_VISIT_RATE',
            'MAX_GAP_HARI','AVG_SELLIN','SELLIN_CV','SELLIN_PER_VISIT',
            'EOM_CONCENTRATION','SOM_CONCENTRATION','MID_CONCENTRATION','BELOW_THRESHOLD_RATE'
        ] if f in df_feat.columns]

        pca_kpi = PCA(n_components=1)
        pca_ec  = PCA(n_components=1)

        axis_x = pca_kpi.fit_transform(
            StandardScaler().fit_transform(df_feat[KPI_COLS].fillna(0))
        ).ravel()
        axis_y = pca_ec.fit_transform(
            StandardScaler().fit_transform(df_feat[EC_COLS].fillna(0))
        ).ravel()

        df_feat['PCA_KPI'] = axis_x
        df_feat['PCA_EC']  = axis_y

        def assign_quadrant(x, y):
            if x >= 0 and y >= 0:   return 'Q1: High KPI - High EC'
            elif x < 0 and y >= 0:  return 'Q2: Low KPI - High EC'
            elif x >= 0 and y < 0:  return 'Q3: High KPI - Low EC'
            else:                    return 'Q4: Low KPI - Low EC'

        df_feat['QUADRANT'] = [assign_quadrant(x, y) for x, y in zip(axis_x, axis_y)]

        # anomaly_df dibuat SETELAH QUADRANT ada
        anomaly_df = (
            df_feat[df_feat["IS_ANOMALY"] == 1]
            .sort_values("ANOMALY_SCORE", ascending=False)
        )

    # ─────────────────────────────────────────
    # Summary Metrics
    # ─────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Ringkasan Analisis</span>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    total_dse     = len(df_feat)
    total_anomaly = (df_feat["IS_ANOMALY"] == 1).sum()
    anomaly_pct   = (total_anomaly / total_dse * 100) if total_dse > 0 else 0
    normal_dse    = total_dse - total_anomaly

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total DSE",   f"{total_dse:,}")
    c2.metric("✅ DSE Normal",  f"{normal_dse:,}")
    c3.metric("🚨 DSE Anomali", f"{total_anomaly:,}")
    c4.metric("📈 Persentase",  f"{anomaly_pct:.1f}%")

    # ─────────────────────────────────────────
    # Status Banner
    # ─────────────────────────────────────────
    st.markdown("")
    if total_anomaly == 0:
        st.success("✓ Tidak ditemukan DSE anomali dalam periode ini.")
    elif anomaly_pct < 5:
        st.info(f"ℹ Ditemukan **{total_anomaly} DSE anomali** ({anomaly_pct:.1f}%) — dalam batas wajar.")
    else:
        st.warning(f"⚠ Ditemukan **{total_anomaly} DSE** ({anomaly_pct:.1f}%) yang memerlukan perhatian lebih lanjut.")

    # ─────────────────────────────────────────
    # Anomaly Table
    # ─────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Daftar DSE Anomali</span>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    show_cols = [
        "DSE_ID", "BRAND_SRC", "OUTLET", "PARTNER_NAME", "MICRO_CLUSTER",
        "BRANCH", "AREA", "QUADRANT","STATUS", "ANOMALY_SCORE", "ANOMALY_REASON"
    ]
    show_cols = [c for c in show_cols if c in anomaly_df.columns]

    if len(anomaly_df) > 0:
        st.dataframe(
            anomaly_df[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "DSE_ID"        : "DSE ID",
                "BRAND_SRC"     : "Brand",
                "OUTLET"        : "Outlet",
                "PARTNER_NAME"  : "Partner Name",
                "MICRO_CLUSTER" : "Micro Cluster",
                "BRANCH"        : "Branch",
                "AREA"          : "Area",
                "QUADRANT"      : "Segmentasi",
                "STATUS"        : "Status",
                "ANOMALY_REASON": "Alasan",
                "ANOMALY_SCORE" : st.column_config.ProgressColumn(
                    "Anomaly Score",
                    help="Semakin tinggi, semakin menyimpang",
                    format="%.4f",
                    min_value=float(anomaly_df["ANOMALY_SCORE"].min()),
                    max_value=float(anomaly_df["ANOMALY_SCORE"].max()),
                ),
            }
        )
    else:
        st.info("Tidak ada data anomali untuk ditampilkan.")

    # ─────────────────────────────────────────
    # Download
    # ─────────────────────────────────────────
    st.markdown("")
    col_dl1, col_dl2, col_empty = st.columns([1, 1, 2])

    with col_dl1:
        csv_anomaly = anomaly_df[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download DSE Anomali",
            data=csv_anomaly,
            file_name="anomaly_result.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_dl2:
        csv_all = df_feat[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Semua DSE",
            data=csv_all,
            file_name="all_dse_result.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ─────────────────────────────────────────────
    # Visualisasi — 3 kolom sejajar
    # ─────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Insight Anomali</span>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    CHART_H   = 360
    GRID_CLR  = "rgba(128,128,128,0.10)"
    BG_TRANSP = "rgba(0,0,0,0)"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Top Branch</div>
            <div class="chart-subtitle">Jumlah DSE anomali terbanyak per branch</div>
        """, unsafe_allow_html=True)

        branch_anom = (
            anomaly_df.groupby("BRANCH").size()
            .reset_index(name="Jumlah DSE")
            .sort_values("Jumlah DSE", ascending=True)
            .tail(10)
        )
        fig_branch = px.bar(branch_anom, x="Jumlah DSE", y="BRANCH",
                            orientation="h", color_discrete_sequence=["#ff8a00"])
        fig_branch.update_layout(
            paper_bgcolor=BG_TRANSP, plot_bgcolor=BG_TRANSP, height=CHART_H,
            margin=dict(t=4, b=4, l=4, r=16), yaxis_title=None, xaxis_title=None,
            xaxis=dict(showgrid=True, gridcolor=GRID_CLR, tickfont=dict(size=11), zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=11)),
        )
        fig_branch.update_traces(marker_line_width=0,
                                  hovertemplate="<b>%{y}</b><br>%{x} DSE<extra></extra>")
        st.plotly_chart(fig_branch, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Brand IM3 vs 3ID</div>
            <div class="chart-subtitle">Distribusi anomali per brand di tiap area</div>
        """, unsafe_allow_html=True)

        brand_area = (
            anomaly_df.groupby(["AREA", "BRAND_SRC"]).size()
            .reset_index(name="Jumlah")
        )
        fig_brand = px.bar(brand_area, x="AREA", y="Jumlah", color="BRAND_SRC",
                           barmode="stack",
                           color_discrete_map={"IM3": "#ff8a00", "3ID": "#a855f7"},
                           labels={"BRAND_SRC": "Brand", "AREA": "", "Jumlah": "Jumlah DSE"})
        fig_brand.update_layout(
            paper_bgcolor=BG_TRANSP, plot_bgcolor=BG_TRANSP, height=CHART_H,
            margin=dict(t=4, b=4, l=4, r=4),
            xaxis=dict(showgrid=False, tickfont=dict(size=10), tickangle=-30, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor=GRID_CLR, tickfont=dict(size=11),
                       title=None, zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=-0.45,
                        xanchor="center", x=0.5, font=dict(size=11), title=None),
        )
        fig_brand.update_traces(marker_line_width=0,
                                 hovertemplate="<b>%{x}</b> — %{data.name}<br>%{y} DSE<extra></extra>")
        st.plotly_chart(fig_brand, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Profil DSE</div>
            <div class="chart-subtitle">Perbandingan dimensi kinerja normal vs anomali</div>
        """, unsafe_allow_html=True)

        radar_cols = {k: v for k, v in {
            "EC_RATE": "EC Rate", "SP_ACH": "SP Ach", "OSA_ACH": "OSA Ach",
            "FWA_ACH": "FWA Ach", "AVG_SELLIN": "Sellin Avg", "AVG_VISIT": "Visit Avg",
        }.items() if k in df_feat.columns}

        def norm_series(s):
            mn, mx = s.min(), s.max()
            return ((s - mn) / (mx - mn) * 100) if mx > mn else s * 0

        radar_normal  = [norm_series(df_feat[df_feat["IS_ANOMALY"] == 0][c]).mean() for c in radar_cols]
        radar_anomali = [norm_series(df_feat[df_feat["IS_ANOMALY"] == 1][c]).mean() for c in radar_cols]
        labels        = list(radar_cols.values())

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_normal + [radar_normal[0]], theta=labels + [labels[0]],
            fill="toself", fillcolor="rgba(255,138,0,0.15)",
            line=dict(color="#ff8a00", width=2), name="Normal"))
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_anomali + [radar_anomali[0]], theta=labels + [labels[0]],
            fill="toself", fillcolor="rgba(168,85,247,0.15)",
            line=dict(color="#a855f7", width=2), name="Anomali"))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                angularaxis=dict(tickfont=dict(size=11), linecolor="rgba(128,128,128,0.15)"),
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9),
                                gridcolor="rgba(128,128,128,0.12)",
                                linecolor="rgba(128,128,128,0.12)", tickvals=[25, 50, 75, 100]),
            ),
            paper_bgcolor=BG_TRANSP, height=CHART_H,
            margin=dict(t=16, b=16, l=16, r=16), showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                        xanchor="center", x=0.5, font=dict(size=11)),
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # Segmentasi DSE
    # ─────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Segmentasi DSE</span>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    QUADRANT_COLORS = {
        'Q1: High KPI - High EC': '#2ecc71',
        'Q2: Low KPI - High EC' : '#f39c12',
        'Q3: High KPI - Low EC' : '#3498db',
        'Q4: Low KPI - Low EC'  : '#e74c3c',
    }
    
    QUADRANT_ORDER = [
    'Q1: High KPI - High EC',
    'Q2: Low KPI - High EC',
    'Q3: High KPI - Low EC',
    'Q4: Low KPI - Low EC'
    ]

    quadrant_count = df_feat['QUADRANT'].value_counts().reset_index()
    quadrant_count.columns = ['Kuadran', 'Jumlah DSE']
    quadrant_count['Kuadran'] = pd.Categorical(
        quadrant_count['Kuadran'],
        categories=QUADRANT_ORDER,
        ordered=True
    )

    quadrant_count = quadrant_count.sort_values('Kuadran')

    fig_cluster = px.bar(quadrant_count, x='Kuadran', y='Jumlah DSE',color='Kuadran', 
                         color_discrete_map=QUADRANT_COLORS, category_orders={"Kuadran": QUADRANT_ORDER})
    fig_cluster.update_layout(height=400, showlegend=False, xaxis=dict(tickangle=-15))

    plot_df = pd.DataFrame({
        'PCA_KPI' : axis_x,
        'PCA_EC'  : axis_y,
        'KUADRAN': df_feat['QUADRANT'],
        'ANOMALY' : np.where(df_feat['IS_ANOMALY'] == 1, 'Anomaly', 'Normal'),
        'DSE_ID'  : df_feat['DSE_ID'],
    })

    fig_pca = px.scatter(plot_df, x='PCA_KPI', y='PCA_EC', color='KUADRAN',
                         symbol='ANOMALY', color_discrete_map=QUADRANT_COLORS, category_orders={
                                "KUADRAN": QUADRANT_ORDER,
                                "ANOMALY": ["Normal", "Anomaly"]},
                         hover_data=['DSE_ID'], opacity=0.75,
                         title='Segmentasi Kuadran (KPI vs Effective Call)')
    fig_pca.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
    fig_pca.add_vline(x=0, line_dash='dash', line_color='gray', line_width=1)
    fig_pca.update_layout(
        height=400,
        xaxis_title=f'PC1 KPI ({pca_kpi.explained_variance_ratio_[0]*100:.1f}% var) → KPI Performance ↑',
        yaxis_title=f'PC1 EC ({pca_ec.explained_variance_ratio_[0]*100:.1f}% var) → EC Efficiency ↑',
    )

    col_cluster, col_pca = st.columns(2)
    with col_cluster:
        st.markdown("##### Distribusi Kuadran")
        st.plotly_chart(fig_cluster, use_container_width=True)
    with col_pca:
        st.markdown("##### Segmentasi KPI vs EC")
        st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("Profil Kuadran")
    quadrant_profile = (
        df_feat.groupby('QUADRANT')[KPI_COLS + EC_COLS + ['ANOMALY_SCORE']]
        .mean().round(2)
    )
    st.dataframe(quadrant_profile, use_container_width=True)

    with st.expander("📖 Deskripsi Feature"):
        st.markdown("""
| Feature | Deskripsi |
|---|---|
| EC_RATE | Persentase hari dengan EC = 100 |
| EC_TREND_WEEKLY | Tren EC antar minggu (positif=membaik, negatif=memburuk) |
| AVG_VISIT | Rata-rata jumlah kunjungan |
| VISIT_CV | Konsistensi kunjungan; makin kecil = makin stabil |
| AVG_SELLIN | Rata-rata transaksi sell in per hari kerja |
| SELLIN_CV | Variasi sell in harian |
| SELLIN_PER_VISIT | Efektivitas kunjungan menghasilkan sell in |
| EOM_CONCENTRATION | Proporsi visit di 5 hari terakhir bulan |
| SOM_CONCENTRATION | Proporsi visit di 5 hari pertama bulan |
| MID_CONCENTRATION | Proporsi visit di hari tengah bulan (hari 8–22) |
| BELOW_THRESHOLD_RATE | Proporsi hari di bawah threshold EC |
| SP_ACH | Achievement SP |
| OSA_ACH | Achievement OSA |
| FWA_ACH | Achievement FWA |
| KPI_AVG_ACH | Rata-rata achievement KPI |
| SCORE_SELLIN | Skor performa sell in |
| COVERAGE_RATE | Rasio SP actual terhadap target sell-in |
| ANOMALY_SCORE | Tingkat penyimpangan dari mayoritas DSE |
""")

    q4_count = (df_feat['QUADRANT'] == 'Q4: Low KPI - Low EC').sum()
    q1_count = (df_feat['QUADRANT'] == 'Q1: High KPI - High EC').sum()

    c1, c2 = st.columns(2)

    with c1:
        st.warning(f"⚠️ {q4_count:,} DSE masuk Q4 (Low KPI & Low EC) — prioritas intervensi.")
    with c2:
        st.success(f"🏆 {q1_count:,} DSE masuk Q1 (High KPI & High EC) — top performer.")

    # ─────────────────────────────────────────
    # Footer
    # ─────────────────────────────────────────
    st.markdown("---")
    st.caption("""
    📊 Sumber Data

    • Aktivitas harian (Visit, Sell In, dan EC) berasal dari sheet DSE_IM3_EC dan DSE_3ID_EC.

    • Data performa DSE (SP Achievement, OSA Achievement, dan FWA Achievement) berasal dari sheet DSE_IM3 dan DSE_3ID.
    """)