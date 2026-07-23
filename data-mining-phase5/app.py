"""
Phase 5 Dashboard — Knowledge Discovery in Banking Datasets (Lending Club)
============================================================================
Dibangun dengan Plotly Dash untuk Insight Communicator.

Cara menjalankan:
    pip install -r requirements.txt
    python app.py
Lalu buka http://127.0.0.1:8050 di browser.

Strategi performa <100ms per callback:
1. Semua figure STATIS dibangun SEKALI saat startup (FIG_*), bukan di dalam
   fungsi tab. Tab function hanya merakit HTML yang referensikan figure tersebut.
2. Data scatter di-pre-sample menjadi ~4.000 baris saat startup (SCATTER_*).
   Callback hanya filter DataFrame kecil — tidak pernah menyentuh DataFrame besar.
3. Tidak ada operasi berat (groupby, merge, sample dari banyak baris) di dalam
   callback manapun.
"""

import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import data_layer as dl

# ===========================================================================
# 1. LOAD DATA — sekali saat startup
# ===========================================================================
cluster_sample, cluster_is_real  = dl.load_clustered_sample()
rules_raw,      rules_is_real    = dl.load_rules()
anomaly_sample, anomaly_is_real  = dl.load_anomaly_report()

# Pastikan tipe kolom cluster konsisten
if "kmeans_cluster" in cluster_sample.columns:
    cluster_sample["kmeans_cluster"] = cluster_sample["kmeans_cluster"].astype(int)

# Deteksi nama kolom tipologi anomali
TYPOLOGY_COL = "Anomaly_Typology"
if TYPOLOGY_COL not in anomaly_sample.columns:
    typ_cols = [c for c in anomaly_sample.columns if "typology" in c.lower()]
    TYPOLOGY_COL = typ_cols[0] if typ_cols else anomaly_sample.columns[0]

PALETTE = ["#1B998B", "#E4572E", "#F4A261", "#0B2545", "#5A6B85"]

# ===========================================================================
# 2. PRE-SAMPLE DATA SCATTER — dipotong sekali, callback hanya filter ~4k baris
# ===========================================================================
_label_map = {int(r["cluster"]): r["label"] for _, r in dl.KMEANS_CLUSTER_PROFILE.iterrows()}
_color_map  = {r["label"]: r["color"]        for _, r in dl.KMEANS_CLUSTER_PROFILE.iterrows()}

# Ambil 2.000 baris per cluster agar saat filter per-cluster masih cukup banyak titik
_cluster_parts = []
for _, _c in dl.KMEANS_CLUSTER_PROFILE.iterrows():
    _mask = cluster_sample["kmeans_cluster"] == int(_c["cluster"])
    _sub  = cluster_sample[_mask]
    _n    = min(2000, len(_sub))
    _cluster_parts.append(_sub.sample(_n, random_state=42))
SCATTER_CLUSTER = pd.concat(_cluster_parts, ignore_index=True).copy()
SCATTER_CLUSTER["Segmen"] = SCATTER_CLUSTER["kmeans_cluster"].map(_label_map)

# Anomaly: ambil 1.000 baris per tipologi
_anomaly_parts = []
for _, _t in dl.ANOMALY_TYPOLOGY.iterrows():
    _mask = anomaly_sample[TYPOLOGY_COL] == _t["typology"]
    _sub  = anomaly_sample[_mask]
    _n    = min(1000, len(_sub))
    if _n > 0:
        _anomaly_parts.append(_sub.sample(_n, random_state=42))
SCATTER_ANOMALY = pd.concat(_anomaly_parts, ignore_index=True).copy() if _anomaly_parts else anomaly_sample.copy()
_anomaly_color_map = {r["typology"]: r["color"] for _, r in dl.ANOMALY_TYPOLOGY.iterrows()}

# ===========================================================================
# 3. PRE-BUILD SEMUA FIGURE STATIS — tidak pernah dibangun ulang di callback
# ===========================================================================

# --- Tab 1: Funnel KDD ---
FIG_FUNNEL = go.Figure(go.Funnel(
    y=dl.KDD_FUNNEL["stage"],
    x=dl.KDD_FUNNEL["rows"],
    textinfo="value+percent initial",
    textfont={"size": 13, "family": "Inter"},
    marker={
        "color": ["#0B2545", "#1B998B", "#F4A261", "#E4572E"],
        "line": {"width": 1, "color": "white"},
    },
    connector={"line": {"color": "#CBD5E1", "width": 1}},
))
FIG_FUNNEL.update_layout(
    margin=dict(l=20, r=20, t=20, b=10), height=320,
    paper_bgcolor="white", plot_bgcolor="white",
    font_family="Inter", font_size=13,
)

# --- Tab 2: Algorithm comparison bar ---
_comp = dl.CLUSTER_ALGO_COMPARISON.copy()
_comp["Silhouette_plot"] = _comp["Silhouette"].fillna(0)
_comp["text_label"] = _comp["Silhouette"].apply(
    lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
)
FIG_COMP = px.bar(
    _comp, x="Algorithm", y="Silhouette_plot", color="Algorithm",
    color_discrete_sequence=PALETTE, text="text_label",
)
FIG_COMP.update_traces(texttemplate="%{text}", textposition="outside")
FIG_COMP.update_layout(
    showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=300,
    paper_bgcolor="white", plot_bgcolor="white", font_family="Inter",
    title="Perbandingan Kualitas Algoritma Clustering (Silhouette Score)",
    title_font_size=14, yaxis_title="Silhouette Score", xaxis_title="",
    yaxis_range=[0, 0.25],
)

# --- Tab 2: Cluster size pie ---
_profile = dl.KMEANS_CLUSTER_PROFILE
FIG_PIE = px.pie(
    _profile, names="label", values="size", color="label",
    color_discrete_map={r["label"]: r["color"] for _, r in _profile.iterrows()},
    hole=0.55,
)
FIG_PIE.update_traces(textinfo="label+percent", textfont_size=12)
FIG_PIE.update_layout(
    margin=dict(l=10, r=10, t=40, b=10), height=300,
    paper_bgcolor="white", font_family="Inter",
    title="Proporsi Segmen Peminjam (sampel 100.000 baris)",
    title_font_size=14,
    legend=dict(orientation="h", yanchor="bottom", y=-0.15),
)

# --- Tab 2: Cluster profile 1x5 subplots (tiap metrik punya skala Y sendiri agar jelas dan tidak kerdil) ---
_profile = dl.KMEANS_CLUSTER_PROFILE
_feats   = ["annual_inc", "dti", "fico_range_low", "revol_util", "default_rate_pct"]
_labels  = ["Pendapatan ($)", "Rasio Utang (%)", "Skor FICO", "Kartu Kredit (%)", "Gagal Bayar (%)"]
_fmt     = ["$%{y:,.0f}", "%{y:.1f}%", "%{y:.0f}", "%{y:.1f}%", "%{y:.2f}%"]
_ranges  = [[0, 105000], [0, 26], [600, 780], [0, 75], [0, 21]]

FIG_PROFILE = make_subplots(rows=1, cols=5, subplot_titles=_labels, shared_yaxes=False, horizontal_spacing=0.06)
for _i, (_f, _lbl, _mfmt, _rng) in enumerate(zip(_feats, _labels, _fmt, _ranges)):
    for _idx, _r in _profile.iterrows():
        _short_label = "Subprime" if _idx == 0 else "Prime"
        FIG_PROFILE.add_trace(go.Bar(
            name=_r["label"], x=[_short_label], y=[_r[_f]],
            marker_color=_r["color"], text=[_r[_f]], texttemplate=_mfmt, textposition="outside",
            showlegend=(_i == 0)
        ), row=1, col=_i + 1)
    FIG_PROFILE.update_yaxes(range=_rng, row=1, col=_i + 1, showgrid=True, gridcolor="#F0F2F5")
    FIG_PROFILE.update_xaxes(showgrid=False, row=1, col=_i + 1)

FIG_PROFILE.update_layout(
    margin=dict(l=15, r=15, t=55, b=45), height=380,
    paper_bgcolor="white", plot_bgcolor="white", font_family="Inter",
    title="Profil Rata-Rata Tiap Segmen: Perbandingan Langsung 5 Metrik Kunci (Tiap Metrik Berskala Sendiri)",
    title_font_size=13.5,
    legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    barmode="group"
)

# --- Tab 3: Bubble chart rules ---
_r = rules_raw.copy()
if "confidence" not in _r.columns or "support" not in _r.columns:
    _r = dl.TOP_RULES_RAW.copy()
FIG_BUBBLE = px.scatter(
    _r, x="confidence", y="support",
    size="lift", color="lift",
    color_continuous_scale="teal", size_max=50,
    hover_name="antecedents",
    hover_data={"consequents": True, "lift": ":.2f", "confidence": ":.2f", "support": ":.3f"},
)
FIG_BUBBLE.update_layout(
    margin=dict(l=10, r=10, t=40, b=10), height=400,
    paper_bgcolor="white", plot_bgcolor="white", font_family="Inter",
    title="Peta Kekuatan Pola: Confidence vs Support (ukuran & warna = Lift)",
    title_font_size=13,
    xaxis_title="Confidence (seberapa sering aturan terbukti benar)",
    yaxis_title="Support (seberapa umum pola ini di populasi)",
    coloraxis_colorbar=dict(title="Lift"),
)

# --- Tab 3: Category bar ---
_cat_counts = dl.BUSINESS_RULES_DF["cat"].value_counts().reset_index()
_cat_counts.columns = ["Kategori", "Jumlah Aturan"]
FIG_CATBAR = px.bar(
    _cat_counts, x="Jumlah Aturan", y="Kategori", orientation="h",
    color="Kategori", color_discrete_sequence=PALETTE, text="Jumlah Aturan",
)
FIG_CATBAR.update_traces(textposition="outside")
FIG_CATBAR.update_layout(
    showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=280,
    paper_bgcolor="white", plot_bgcolor="white", font_family="Inter",
    title="15 Aturan Bisnis Terkurasi per Tema",
    title_font_size=13, xaxis_title="", yaxis_title="",
)

# --- Tab 3: Rules DataTable config (data dihitung sekali) ---
_rules_table_data    = _r.round(4).to_dict("records")
_rules_table_cols    = [{"name": c, "id": c} for c in _r.columns]
_rules_table_tooltip = [{c: {"value": str(row[c]), "type": "markdown"} for c in _r.columns}
                        for row in _rules_table_data]

# --- Tab 4: Method bar ---
FIG_METHOD = px.bar(
    dl.ANOMALY_METHOD_COUNTS, x="method", y="count",
    text="pct", color="method", color_discrete_sequence=PALETTE,
)
FIG_METHOD.update_traces(texttemplate="%{text}%", textposition="outside")
FIG_METHOD.update_layout(
    showlegend=False, margin=dict(l=10, r=10, t=40, b=40), height=340,
    paper_bgcolor="white", plot_bgcolor="white", font_family="Inter",
    title="Berapa Banyak Anomali Ditemukan Tiap Metode?",
    title_font_size=13, xaxis_title="", yaxis_title="Jumlah Baris",
    xaxis_tickangle=-10,
)

# --- Tab 4: Confidence funnel ---
_conf_sorted = dl.ANOMALY_CONFIDENCE.sort_values("count", ascending=True).copy()
FIG_CONF = go.Figure(go.Funnel(
    y=_conf_sorted["confidence"],
    x=_conf_sorted["count"],
    textinfo="value+percent initial",
    marker={"color": ["#CBD5E1", "#F4A261", "#E4572E", "#0B2545"]},
    textfont={"size": 12, "family": "Inter"},
))
FIG_CONF.update_layout(
    margin=dict(l=10, r=10, t=40, b=10), height=320,
    paper_bgcolor="white", font_family="Inter",
    title="Berapa Banyak yang Disepakati Lebih Banyak Metode?",
    title_font_size=13,
)

# --- Tab 4: Donut anomaly typology ---
FIG_DONUT = px.pie(
    dl.ANOMALY_TYPOLOGY, names="typology", values="count", hole=0.55,
    color="typology",
    color_discrete_map={r["typology"]: r["color"] for _, r in dl.ANOMALY_TYPOLOGY.iterrows()},
)
FIG_DONUT.update_traces(
    textinfo="label+percent", textfont_size=11,
    hovertemplate="<b>%{label}</b><br>%{value:,} anomali (%{percent})<extra></extra>",
)
FIG_DONUT.update_layout(
    margin=dict(l=10, r=10, t=40, b=10), height=360,
    paper_bgcolor="white", font_family="Inter",
    title=f"Komposisi {dl.ANOMALY_STRONG_TOTAL:,} Anomali Kuat (Skor ≥ 2)",
    title_font_size=13,
    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
)

# ===========================================================================
# 4. DASH APP
# ===========================================================================
app = dash.Dash(
    __name__,
    title="Lending Club Knowledge Discovery Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server


# ---------------------------------------------------------------------------
# Helper components
# ---------------------------------------------------------------------------
def badge(is_real):
    if is_real:
        return html.Span("✓ data asli", className="data-badge badge-real")
    return html.Span("⚠ sampel ilustratif", className="data-badge badge-sim")


def kpi_card(value, label, icon=""):
    return html.Div([
        html.Div(icon, className="kpi-icon"),
        html.Div(value, className="kpi-value"),
        html.Div(label, className="kpi-label"),
    ], className="kpi-card")


def insight_box(text, color_accent=None):
    style = {}
    if color_accent:
        style["borderLeft"] = f"4px solid {color_accent}"
    return html.Div(text, className="narrative-box", style=style)


def section_header(title, subtitle=""):
    elems = [html.Div(title, className="tab-content-title")]
    if subtitle:
        elems.append(html.Div(subtitle, className="tab-content-sub"))
    return html.Div(elems, className="section-header")


# ---------------------------------------------------------------------------
# HEADER & KPI ROW (built once at module level)
# ---------------------------------------------------------------------------
header = html.Div([
    html.Div([
        html.Div("📊", className="header-icon"),
        html.Div([
            html.H1("Lending Club Loans Knowledge Discovery Dashboard"),
            html.P("Phase 5: Visualization & Knowledge Presentation"),
        ])
    ], className="header-title-row"),

    html.Div([
        html.Div([
            html.Span("❓ Pertanyaan Sentral: ", className="cq-label"),
            "Apa yang kita temukan dari 889.991 pinjaman Lending Club yang ",
            html.B("tidak terlihat dari data mentah?"),
        ], className="central-question"),
    ]),

    html.Div([
        html.Div([html.Span("1", className="finding-num"),
                  "Peminjam terbelah jadi 2 arketipe risiko kontras meski datanya kontinu"],
                 className="finding-chip"),
        html.Div([html.Span("2", className="finding-num"),
                  "Kombinasi FICO + DTI 3× lebih prediktif dari sinyal tunggal"],
                 className="finding-chip"),
        html.Div([html.Span("3", className="finding-num"),
                  "54% anomali adalah nasabah SEHAT, bukan risiko atau fraud"],
                 className="finding-chip"),
    ], className="findings-row"),
], className="app-header")

kpi_row = html.Div([
    kpi_card(f"{dl.PHASE1_SUMMARY['final_rows']:,}", "Pinjaman Dianalisis", "🏦"),
    kpi_card(f"{dl.PHASE1_SUMMARY['final_cols']}",   "Fitur Kunci", "📋"),
    kpi_card("2",                                     "Segmen Risiko", "👥"),
    kpi_card(f"{dl.APRIORI_STATS['rules_after_confidence_filter']}", "Pola Tersembunyi", "🔗"),
    kpi_card(f"{dl.ANOMALY_STRONG_TOTAL:,}",          "Anomali Terdeteksi", "🔍"),
], className="kpi-row")


# ===========================================================================
# TAB CONTENT FUNCTIONS
# Setiap fungsi hanya merakit HTML. Semua figure sudah ada sebagai FIG_*
# → tidak ada komputasi figure di dalam fungsi ini.
# ===========================================================================

def tab_overview():
    pipeline_cards = html.Div([
        html.Div([
            html.Div([html.Span("1", className="phase-num"), "Phase 1: Preprocessing"], className="phase-title"),
            insight_box(
                f"Data mentah 890.000 baris × 151 kolom dipadatkan menjadi "
                f"{dl.PHASE1_SUMMARY['final_rows']:,} baris × {dl.PHASE1_SUMMARY['final_cols']} fitur kunci. "
                f"{dl.PHASE1_SUMMARY['cols_gt50pct_missing_dropped']} kolom dibuang karena >50% datanya kosong. "
                f"Fitur int_rate dihapus karena redundan dengan grade (korelasi >0.95).",
                "#1B998B",
            ),
        ], className="card phase-card"),
        html.Div([
            html.Div([html.Span("2", className="phase-num"), "Phase 2: Segmentasi Peminjam"], className="phase-title"),
            insight_box(
                "K-Means (K=2, Silhouette 0.189) adalah algoritma terbaik, memberi dua segmen yang punya "
                "profil risiko sangat kontras. DBSCAN mengonfirmasi bahwa data sebenarnya kontinu (tidak ada "
                "\"celah\" alami), artinya dikotomi Subprime vs Prime adalah penyederhanaan yang berguna "
                "secara bisnis — bukan pembagian yang terjadi di alam.",
                "#0B2545",
            ),
        ], className="card phase-card"),
        html.Div([
            html.Div([html.Span("3", className="phase-num"), "Phase 3: Pola Asosiasi Tersembunyi"], className="phase-title"),
            insight_box(
                f"Dari {dl.APRIORI_STATS['frequent_itemsets']} kombinasi item yang sering muncul bersama, "
                f"ditemukan {dl.APRIORI_STATS['rules_generated']} aturan awal → disaring menjadi "
                f"{dl.APRIORI_STATS['rules_after_confidence_filter']} aturan valid. "
                "Pola paling kuat: nasabah dengan FICO bagus + beban utang rendah → nyaris pasti Grade A "
                "(3,24× lebih sering dari rata-rata).",
                "#F4A261",
            ),
        ], className="card phase-card"),
        html.Div([
            html.Div([html.Span("4", className="phase-num"), "Phase 4: Deteksi Anomali"], className="phase-title"),
            insight_box(
                f"4 metode deteksi disilangkan via ensemble scoring. "
                f"{dl.ANOMALY_STRONG_TOTAL:,} pinjaman (~7%) dianggap anomali kuat (≥2 metode setuju). "
                "Temuan mengejutkan: 53,7% anomali ini adalah nasabah dengan profil terlalu bagus "
                "(gaji tinggi, pinjaman kecil) — bukan sinyal bahaya.",
                "#E4572E",
            ),
        ], className="card phase-card"),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "16px"})

    return html.Div([
        html.Div([
            section_header(
                "Perjalanan Data: Dari 890.000 Baris Mentah ke Insight Bisnis",
                "Setiap tahap KDD menyaring dan memperkaya data, hasilnya bukan sekadar angka, "
                "tapi pengetahuan yang bisa dipakai.",
            ),
            dcc.Graph(figure=FIG_FUNNEL, config={"displayModeBar": False}),
        ], className="card"),
        pipeline_cards,
    ])


def tab_segmentation():
    profile = dl.KMEANS_CLUSTER_PROFILE
    return html.Div([
        section_header(
            "Temuan #1: Dua Wajah Peminjam yang Sangat Berbeda",
            "K-Means menemukan bahwa 62% peminjam punya profil risiko tinggi dengan gagal bayar 17%,  "
            "sementara 38% sisanya relatif aman dengan gagal bayar hanya 7%. Perbedaan 2,4× ini tidak "
            "terlihat dari angka individual.",
        ),

        html.Div([
            html.Div(dcc.Graph(figure=FIG_COMP,    config={"displayModeBar": False}), className="card"),
            html.Div(dcc.Graph(figure=FIG_PIE,     config={"displayModeBar": False}), className="card"),
        ], className="row-2col"),

        html.Div([dcc.Graph(figure=FIG_PROFILE, config={"displayModeBar": False})], className="card"),

        html.Div([
            html.H3("Ringkasan Profil Tiap Segmen", style={"marginTop": 0}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("●", style={"color": r["color"], "fontSize": "22px", "marginRight": "8px"}),
                        html.B(f"Segmen {r['cluster']}: \"{r['label']}\" - {r['pct_total']}% dari populasi"),
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
                    insight_box(
                        f"Pendapatan rata-rata ${r['annual_inc']:,.0f} · "
                        f"Rasio utang {r['dti']}% · "
                        f"FICO {r['fico_range_low']:.0f} · "
                        f"Pemakaian kartu kredit {r['revol_util']}% · "
                        f"Grade rata-rata {r['grade']:.1f} · "
                        f"Tingkat gagal bayar {r['default_rate_pct']}%",
                        r["color"],
                    ),
                ], style={"padding": "4px"}) for _, r in profile.iterrows()
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}),
        ], className="card"),

        html.Div([
            html.H3([
                "Peta Sebaran Interaktif: Eksplorasi Multi-Dimensi Segmen  ",
                badge(cluster_is_real),
            ], style={"marginTop": 0}),
            html.P(
                "Pilih variabel sumbu X dan Y untuk melihat bagaimana K-Means memisahkan nasabah. "
                "Perhatikan saat memilih 'Pendapatan Tahunan vs Rasio Utang', titiknya tumpang tindih. "
                "Namun saat diganti ke 'Skor FICO vs Pemakaian Kartu Kredit', kedua segmen terbelah dengan sangat jelas!",
                style={"fontSize": "13px", "color": "#5A6B85", "marginBottom": "12px"},
            ),
            html.Div([
                html.Div([
                    html.Label("Sumbu Mendatar (X):", style={"fontSize": "12px", "fontWeight": "600", "color": "#0B2545", "display": "block", "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="cluster-x-select",
                        options=[
                            {"label": "Annual Income ($)", "value": "annual_inc"},
                            {"label": "Debt To Income/ DTI Ratio (%)", "value": "dti"},
                            {"label": "FICO Score", "value": "fico_range_low"},
                            {"label": "Revolving Utilization (%)", "value": "revol_util"},
                            {"label": "Loan Amount ($)", "value": "loan_amnt"},
                            {"label": "Employment Length (Year)", "value": "emp_length"},
                        ],
                        value="revol_util",
                        clearable=False,
                        searchable=False,
                    ),
                ], style={"flex": "1", "minWidth": "200px"}),
                html.Div([
                    html.Label("Sumbu Tegak (Y):", style={"fontSize": "12px", "fontWeight": "600", "color": "#0B2545", "display": "block", "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="cluster-y-select",
                        options=[
                            {"label": "Annual Income ($)", "value": "annual_inc"},
                            {"label": "Debt To Income/ DTI Ratio (%)", "value": "dti"},
                            {"label": "FICO Score", "value": "fico_range_low"},
                            {"label": "Revolving Utilization (%)", "value": "revol_util"},
                            {"label": "Loan Amount ($)", "value": "loan_amnt"},
                            {"label": "Employment Length (Year)", "value": "emp_length"},
                        ],
                        value="fico_range_low",
                        clearable=False,
                        searchable=False,
                    ),
                ], style={"flex": "1", "minWidth": "200px"}),
                html.Div([
                    html.Label("Filter Segmen:", style={"fontSize": "12px", "fontWeight": "600", "color": "#0B2545", "display": "block", "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="cluster-filter",
                        options=[{"label": "Semua Segmen", "value": "all"}] +
                                [{"label": r["label"], "value": r["cluster"]} for _, r in profile.iterrows()],
                        value="all",
                        clearable=False,
                        searchable=False,
                    ),
                ], style={"flex": "1", "minWidth": "200px"}),
            ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "14px"}),
            dcc.Graph(id="cluster-scatter", config={"displayModeBar": False}),
        ], className="card"),

        html.Div([
            html.H3("📌 Mengapa DBSCAN Hanya Menemukan 1 Cluster?", style={"marginTop": 0}),
            insight_box(
                "DBSCAN dijalankan di ruang 5-dimensi hasil UMAP pada 20.000 data sampel. "
                "Pencarian parameter sistematis (eps 0.10–1.00) tidak menemukan satu pun konfigurasi "
                "yang menghasilkan lebih dari 1 cluster bermakna — 107 titik (0,5%) teridentifikasi "
                "sebagai noise. Ini bukan kegagalan DBSCAN, melainkan konfirmasi bahwa risiko kredit "
                "di data ini bersifat gradasi kontinu, bukan kelompok terpisah tajam seperti warna "
                "pada spektrum cahaya. K-Means \"memotong\" spektrum ini di satu titik untuk "
                "menghasilkan dua segmen yang informatif secara bisnis.",
            ),
        ], className="card"),
    ])


def tab_rules():
    return html.Div([
        section_header(
            "Temuan #2: Kombinasi Sinyal Jauh Lebih Kuat dari Sinyal Tunggal",
            f"Dari {dl.APRIORI_STATS['frequent_itemsets']} kombinasi yang sering muncul bersama, "
            f"disaring menjadi {dl.APRIORI_STATS['rules_after_confidence_filter']} aturan valid. "
            "Pola terkuat: nasabah FICO bagus + utang rendah → Grade A dengan kekuatan 3,24× di atas rata-rata.",
        ),

        html.Div([dcc.Graph(figure=FIG_BUBBLE, config={"displayModeBar": False})], className="card"),

        html.Div([
            html.Div(dcc.Graph(figure=FIG_CATBAR, config={"displayModeBar": False}), className="card"),
            html.Div([
                html.H3("Filter Aturan Berdasarkan Tema", style={"marginTop": 0}),
                html.P(
                    "Pilih tema untuk melihat aturan bisnis yang relevan beserta rekomendasi aksinya.",
                    style={"fontSize": "13px", "color": "#5A6B85", "marginBottom": "10px"},
                ),
                dcc.Dropdown(
                    id="rule-cat-filter",
                    options=[{"label": "Semua Tema", "value": "all"}] +
                            [{"label": c, "value": c} for c in dl.BUSINESS_RULES_DF["cat"].unique()],
                    value="all",
                    clearable=False,
                    searchable=False,
                    style={"marginBottom": "12px"},
                ),
                html.Div(id="rule-list-output"),
            ], className="card"),
        ], className="row-2col"),

        html.Div([
            html.H3(
                [f"Tabel Top-{len(_r)} Aturan Terurut by Lift  ", badge(rules_is_real)],
                style={"marginTop": 0},
            ),
            html.P(
                "Lift > 1 berarti aturan ini lebih sering terjadi dari yang diharapkan secara kebetulan. "
                "Lift 3,24 berarti 3,24× lebih sering — itu sinyal yang sangat kuat.",
                style={"fontSize": "13px", "color": "#5A6B85", "marginBottom": "10px"},
            ),
            dash_table.DataTable(
                data=_rules_table_data,
                columns=_rules_table_cols,
                style_cell={
                    "fontFamily": "Inter", "fontSize": "12.5px",
                    "padding": "8px 10px", "textAlign": "left",
                    "whiteSpace": "normal", "height": "auto",
                    "maxWidth": "260px", "overflow": "hidden", "textOverflow": "ellipsis",
                },
                style_header={
                    "backgroundColor": "#0B2545", "color": "white",
                    "fontWeight": "600", "fontSize": "12px",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#F4F6F9"},
                    {"if": {"filter_query": "{lift} >= 3"}, "color": "#1B998B", "fontWeight": "700"},
                ],
                tooltip_data=_rules_table_tooltip,
                tooltip_duration=None,
                page_size=8, sort_action="native",
            ),
        ], className="card"),
    ])


def tab_anomaly():
    return html.Div([
        section_header(
            "Temuan #3: Anomali Tidak Selalu Berarti Bahaya",
            f"4 metode deteksi menemukan {dl.ANOMALY_STRONG_TOTAL:,} anomali kuat (~7% populasi). "
            "Yang mengejutkan: lebih dari separuhnya adalah nasabah dengan profil keuangan yang terlalu bagus "
            "— bukan indikasi risiko. Menolak semua anomali secara otomatis adalah kesalahan mahal.",
        ),

        html.Div([
            html.Div(dcc.Graph(figure=FIG_METHOD, config={"displayModeBar": False}), className="card"),
            html.Div(dcc.Graph(figure=FIG_CONF,   config={"displayModeBar": False}), className="card"),
        ], className="row-2col"),

        html.Div([
            html.Div([dcc.Graph(figure=FIG_DONUT, config={"displayModeBar": False})], className="card"),
            html.Div([
                html.H3("Apa Arti Tiap Kategori Anomali?", style={"marginTop": 0}),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Span("●", style={"color": r["color"], "fontSize": "18px", "marginRight": "8px"}),
                            html.B(f"{r['typology']}"),
                            html.Span(f" — {r['count']:,} baris ({r['pct']}%)",
                                      style={"color": "#5A6B85", "fontSize": "12px"}),
                        ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
                        insight_box(r["desc"], r["color"]),
                    ], style={"marginBottom": "14px"}) for _, r in dl.ANOMALY_TYPOLOGY.iterrows()
                ]),
            ], className="card"),
        ], className="row-2col"),

        html.Div([
            html.H3([
                "Peta Sebaran Anomali: Pendapatan vs Nilai Pinjaman  ",
                badge(anomaly_is_real),
            ], style={"marginTop": 0}),
            html.P(
                "Filter berdasarkan tipe anomali untuk melihat pola sebaran tiap kelompok. "
                "Perhatikan bahwa 'Rare Legitimate Case' cenderung ada di pojok kiri atas "
                "(gaji sangat tinggi, pinjaman kecil).",
                style={"fontSize": "13px", "color": "#5A6B85", "marginBottom": "12px"},
            ),
            dcc.Dropdown(
                id="typology-filter",
                options=[{"label": "Semua Tipe Anomali", "value": "all"}] +
                        [{"label": t, "value": t} for t in dl.ANOMALY_TYPOLOGY["typology"]],
                value="all",
                clearable=False,
                searchable=False,
                style={"maxWidth": "360px", "marginBottom": "12px"},
            ),
            dcc.Graph(id="anomaly-scatter", config={"displayModeBar": False}),
        ], className="card"),
    ])


def tab_expo():
    qa_items = [
        {
            "q": "Q1. Aturan asosiasi mana yang paling mengejutkan, dan kenapa?",
            "a": (
                "{FICO Very Good, DTI Healthy} → {Grade A} dengan lift 3,24× adalah yang paling mengejutkan — "
                "bukan karena arahnya (sudah intuitif), tapi karena kekuatannya. Kombinasi dua kondisi "
                "sederhana ini menghasilkan Grade A tiga kali lebih sering dari yang diharapkan secara acak. "
                "Pola kedua yang lebih mengejutkan: {DTI High Risk} → {Revol_Util Maxed Out} — dua sinyal "
                "risiko yang tampak \"independen\" ternyata hampir selalu muncul bersamaan, menunjukkan "
                "bahwa kebangkrutan finansial punya pola yang sangat konsisten."
            ),
            "color": "#1B998B", "icon": "🔗",
        },
        {
            "q": "Q2. Metode clustering mana yang paling mudah dipahami untuk dataset ini?",
            "a": (
                "K-Means (K=2) jelas pemenangnya dari sisi interpretabilitas bisnis: menghasilkan dua segmen "
                "seimbang (62% vs 38%) dengan perbedaan default rate 2,4× yang bisa langsung diterjemahkan "
                "ke keputusan pricing dan retensi. Hierarchical Ward secara matematis mirip tapi "
                "Silhouette lebih rendah (0,107 vs 0,189). DBSCAN tidak menghasilkan segmentasi yang "
                "actionable (1 cluster besar + 107 noise) — tapi tetap berguna sebagai alat "
                "deteksi anomali, bukan alat segmentasi pelanggan."
            ),
            "color": "#0B2545", "icon": "👥",
        },
        {
            "q": "Q3. Anomali apa saja yang ditemukan, dan apa maknanya dalam konteks perbankan nyata?",
            "a": (
                f"Dari {dl.ANOMALY_TOTAL_ROWS:,} pinjaman, {dl.ANOMALY_STRONG_TOTAL:,} (~7%) dianggap anomali kuat "
                "oleh ≥2 metode. Tapi yang paling krusial: 53,7% di antaranya adalah 'Rare Legitimate Case' — "
                "nasabah dengan profil keuangan yang terlalu bagus (gaji sangat tinggi, FICO sempurna, "
                "pinjaman sangat kecil). Hanya 32% yang benar-benar merupakan sinyal risiko. "
                "Implikasi praktis: jika bank menolak semua anomali secara otomatis, mereka kehilangan "
                "33.000+ nasabah premium yang sebenarnya adalah klien terbaik mereka."
            ),
            "color": "#E4572E", "icon": "🔍",
        },
        {
            "q": "Q4. Bagaimana perbandingan temuan ini dengan kelompok lain?",
            "a": (
                "[Diisi setelah sesi Mining Expo berlangsung] Pertanyaan pembanding yang diajukan: "
                "(a) Apakah domain lain (fraud detection, mortgage, AML) juga menemukan struktur cluster "
                "yang kontinu seperti ini, atau justru menemukan kelompok yang terpisah tajam? "
                "(b) Seberapa tinggi threshold lift yang kelompok lain pakai — apakah 1,15 terlalu "
                "konservatif atau terlalu longgar? "
                "(c) Apakah proporsi 'anomali yang sebenarnya sah' di domain lain juga didominasi "
                "kasus sah seperti di sini, atau justru sebaliknya?"
            ),
            "color": "#F4A261", "icon": "🌐",
        },
    ]

    qa_cards = [
        html.Div([
            html.Div([
                html.Span(item["icon"], style={"fontSize": "24px", "marginRight": "10px"}),
                html.Span(item["q"], className="expo-question"),
            ], style={"display": "flex", "alignItems": "flex-start", "marginBottom": "10px"}),
            insight_box(item["a"], item["color"]),
        ], className="card", style={"borderTop": f"3px solid {item['color']}"})
        for item in qa_items
    ]

    return html.Div([
        section_header(
            "Mining Expo & Jawaban Central Discovery Question",
            "Draf jawaban untuk 4 pertanyaan wajib sesi lintas kelompok. "
            "Isi bagian perbandingan (Q4) setelah expo berlangsung.",
        ),

        html.Div(qa_cards[:2], className="row-2col"),
        html.Div(qa_cards[2:], className="row-2col"),

        html.Div([
            html.H3("🎯 Jawaban Lengkap: Apa yang Kita Temukan yang Tidak Terlihat dari Data Mentah?",
                    style={"marginTop": 0}),
            html.Div([
                html.Div([
                    html.Div("Temuan 1", className="finding-label"),
                    insight_box(
                        "Meski data finansial peminjam bersifat kontinu dan gradual, populasi mereka "
                        "terorganisir menjadi dua arketipe risiko yang kontras: 'High-Risk Subprime' (62,2%, "
                        "gagal bayar 17,07%) dan 'Low-Risk Prime' (37,8%, gagal bayar 7,21%). Perbedaan 2,4× "
                        "ini tidak tampak dari distribusi variabel individual — hanya muncul saat lima "
                        "variabel dievaluasi bersama melalui clustering.",
                        "#1B998B",
                    ),
                ]),
                html.Div([
                    html.Div("Temuan 2", className="finding-label"),
                    insight_box(
                        "FICO Very Good + DTI Healthy → Grade A (lift 3,24×) berarti kombinasi dua kondisi "
                        "sederhana ini menghasilkan Grade A tiga kali lebih sering dari yang diharapkan. "
                        "Kekuatan ini mustahil terlihat dari korelasi biasa karena melibatkan interaksi "
                        "multi-variabel yang non-linear.",
                        "#0B2545",
                    ),
                ]),
                html.Div([
                    html.Div("Temuan 3", className="finding-label"),
                    insight_box(
                        "Mayoritas 'anomali statistik' (53,7%) bukan sinyal bahaya — mereka adalah nasabah "
                        "dengan profil keuangan yang terlalu ideal sehingga terdeteksi sebagai outlier secara "
                        "matematis. Menolak semua outlier otomatis bukan hanya tidak efisien — ini bisa "
                        "mengakibatkan kehilangan puluhan ribu nasabah premium.",
                        "#E4572E",
                    ),
                ]),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "16px",
                      "marginTop": "16px"}),
        ], className="card"),
    ])


# ===========================================================================
# LAYOUT
# ===========================================================================
app.layout = html.Div([
    header,
    kpi_row,
    html.Div([
        dcc.Tabs(
            id="main-tabs",
            value="tab-overview",
            children=[
                dcc.Tab(label="📌 Overview",      value="tab-overview",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="👥 Segmentasi",    value="tab-segmentation",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="🔗 Pola Asosiasi", value="tab-rules",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="🔍 Deteksi Anomali", value="tab-anomaly",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="🌐 Mining Expo",   value="tab-expo",
                        className="custom-tab", selected_className="custom-tab--selected"),
            ],
        ),
        html.Div(id="tab-content"),
    ], className="content-wrap"),

    html.Div([
        html.Span("Data Mining Project — Lending Club Knowledge Discovery · Phase 5"),
        html.Span(" · ", style={"opacity": "0.4"}),
        html.Span("889.991 pinjaman · 2007–2018"),
    ], className="app-footer"),
])


# ===========================================================================
# CALLBACKS — hanya filter DataFrame kecil (pre-sampled), tidak ada komputasi berat
# ===========================================================================

@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    """Rakit HTML tab — figure sudah ada sebagai FIG_*, tidak dibangun ulang."""
    return {
        "tab-overview":     tab_overview,
        "tab-segmentation": tab_segmentation,
        "tab-rules":        tab_rules,
        "tab-anomaly":      tab_anomaly,
        "tab-expo":         tab_expo,
    }.get(tab, tab_overview)()


@app.callback(
    Output("cluster-scatter", "figure"),
    [Input("cluster-filter", "value"),
     Input("cluster-x-select", "value"),
     Input("cluster-y-select", "value")]
)
def update_cluster_scatter(selected, x_col, y_col):
    """
    Filter dan eksplorasi sumbu X/Y dari SCATTER_CLUSTER (~4.000 baris pre-sampled).
    Tidak ada sampling atau operasi berat — selalu <50ms.
    """
    df = SCATTER_CLUSTER if selected == "all" else \
         SCATTER_CLUSTER[SCATTER_CLUSTER["kmeans_cluster"] == int(selected)]

    # Pemetaan label Indonesia yang manusiawi
    col_labels = {
        "annual_inc": "Pendapatan Tahunan ($)",
        "dti": "Rasio Utang/Penghasilan (%)",
        "fico_range_low": "Skor FICO",
        "revol_util": "Pemakaian Kartu Kredit (%)",
        "loan_amnt": "Nilai Pinjaman ($)",
        "emp_length": "Lama Bekerja (Tahun)",
        "Segmen": "Segmen",
    }

    fig = px.scatter(
        df, x=x_col or "annual_inc", y=y_col or "dti",
        color="Segmen", color_discrete_map=_color_map,
        opacity=0.55,
        labels=col_labels,
        hover_data={
            "annual_inc": ":,.0f", "dti": ":.1f", "fico_range_low": ":.0f",
            "revol_util": ":.1f", "loan_amnt": ":,.0f", "emp_length": ":.0f"
        },
    )
    fig.update_traces(marker_size=5)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=400,
        paper_bgcolor="white", plot_bgcolor="#FAFBFC",
        font_family="Inter",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
    )
    return fig


@app.callback(Output("rule-list-output", "children"), Input("rule-cat-filter", "value"))
def update_rule_list(selected_cat):
    """Filter list aturan (15 baris max) — selalu <10ms."""
    df = dl.BUSINESS_RULES_DF if selected_cat == "all" else \
         dl.BUSINESS_RULES_DF[dl.BUSINESS_RULES_DF["cat"] == selected_cat]

    items = []
    for _, r in df.iterrows():
        lift_txt   = f"Lift {r['lift']:.2f}×" if pd.notna(r["lift"]) else "Observasi umum"
        lift_color = "#1B998B" if pd.notna(r["lift"]) and r["lift"] >= 3 else "#5A6B85"
        items.append(html.Li([
            html.B(r["rule"], style={"fontSize": "12.5px"}),
            html.Span(f" ({lift_txt})",
                      style={"color": lift_color, "fontWeight": "600", "fontSize": "12px"}),
            html.Br(),
            html.Span(f"→ {r['insight']}", style={"color": "#5A6B85", "fontSize": "12px"}),
        ], style={"marginBottom": "12px", "lineHeight": "1.5"}))

    return html.Ul(items,
                   style={"paddingLeft": "18px", "maxHeight": "280px",
                          "overflowY": "auto", "marginTop": "4px"})


@app.callback(Output("anomaly-scatter", "figure"), Input("typology-filter", "value"))
def update_anomaly_scatter(selected):
    """
    Filter dari SCATTER_ANOMALY (~4.000 baris pre-sampled).
    Tidak ada sampling atau operasi berat — harus <50ms.
    """
    df = SCATTER_ANOMALY if selected == "all" else \
         SCATTER_ANOMALY[SCATTER_ANOMALY[TYPOLOGY_COL] == selected]

    fig = px.scatter(
        df, x="annual_inc", y="loan_amnt",
        color=TYPOLOGY_COL, color_discrete_map=_anomaly_color_map,
        opacity=0.55,
        labels={
            "annual_inc": "Pendapatan Tahunan ($)",
            "loan_amnt":  "Nilai Pinjaman ($)",
            TYPOLOGY_COL: "Tipe Anomali",
        },
        hover_data={"annual_inc": ":,.0f", "loan_amnt": ":,.0f", TYPOLOGY_COL: True},
    )
    fig.update_traces(marker_size=5)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=400,
        paper_bgcolor="white", plot_bgcolor="#FAFBFC",
        font_family="Inter",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
