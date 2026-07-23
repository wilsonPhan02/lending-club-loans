"""
data_layer.py
=================
Lapisan data untuk Dashboard Phase 5 (Lending Club KDD Project).

Filosofi loading data (penting untuk dipahami tim):
-----------------------------------------------------
1. Notebook Phase 1-4 menghasilkan beberapa file CSV/PKL di working directory
   yang sama (lihat DATA_FILES di bawah). Jika file-file itu ADA di folder
   ini, fungsi load_* akan otomatis memakainya (data asli, 100% akurat).
2. Karena file CSV besar tersebut tidak ikut di-upload/di-share bersama
   notebook, modul ini menyediakan FALLBACK berupa angka-angka agregat NYATA
   yang diambil langsung dari output cell yang sudah dieksekusi di keempat
   notebook (bukan angka karangan). Fallback ini membuat dashboard tetap bisa
   di-demo & dikembangkan SEKARANG, dan otomatis "naik kelas" ke data asli
   begitu file CSV asli diletakkan di folder yang sama.
3. Untuk scatter plot level-baris (butuh ribuan titik individual) yang
   datanya tidak tersedia dalam bentuk agregat, kita generate SAMPEL
   ILUSTRATIF (synthetic) yang mengikuti statistik asli (mean, proporsi,
   skewness arah) dari notebook. Titik-titik ini diberi label jelas
   "(simulasi ilustratif)" di judul chart -- WAJIB diganti dengan data asli
   (phase2_clustered_sample.csv / phase4_anomaly_report.csv) sebelum
   submit final.

Cara mengganti ke data asli:
-----------------------------------------------------
Cukup copy file-file berikut ke folder yang sama dengan app.py dan masukan kedalam folder csv (kalo belom ada bisa dibuat dlu):
    cleaned_lending_club.csv
    cleaned_lending_club_no_winsorization.csv
    scaled_lending_club.csv
    phase2_clustered_sample.csv
    phase4_anomaly_report.csv
    (opsional) phase3_association_rules.csv  <- lihat catatan di README
Tidak ada perubahan kode yang dibutuhkan di app.py.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILES = {
    "cleaned": "csv/cleaned_lending_club.csv",
    "cleaned_no_winsor": "csv/cleaned_lending_club_no_winsorization.csv",
    "scaled": "csv/scaled_lending_club.csv",
    "clustered_sample": "csv/phase2_clustered_sample.csv",
    "dbscan_outliers": "csv/phase2_dbscan_outliers.csv",
    "anomaly_report": "csv/phase4_anomaly_report.csv",
    "rules": "csv/phase3_association_rules.csv",
}


def _path(key):
    return os.path.join(BASE_DIR, DATA_FILES[key])


def file_available(key):
    return os.path.exists(_path(key))


# ---------------------------------------------------------------------------
# PHASE 1 -- Preprocessing summary (angka asli dari notebook data-mining.ipynb)
# ---------------------------------------------------------------------------
PHASE1_SUMMARY = {
    "source": "Lending Club accepted_2007_to_2018Q4.csv",
    "raw_rows_sampled": 890_000,
    "raw_cols": 151,
    "final_rows": 889_991,
    "final_cols": 8,
    "cols_gt50pct_missing_dropped": 44,
    "cols_20_50pct_missing": 14,
    "cols_lt20pct_missing_imputed": 92,
    "final_features": [
        "loan_amnt", "grade", "annual_inc", "dti",
        "fico_range_low", "revol_util", "emp_length", "purpose_small_business",
    ],
    "dropped_high_corr": "int_rate (r > 0.95 dengan grade)",
    "mining_angle": "Cluster borrowers by risk profile using grade, income, dan debt ratio",
    "mining_rule_target": (
        "Borrowers dengan tujuan small business & masa kerja 10+ tahun cenderung "
        "mendapat Grade A dengan interest rate rendah"
    ),
}

# ---------------------------------------------------------------------------
# PHASE 2 -- Clustering results (angka asli dari data-mining-phase2.ipynb)
# ---------------------------------------------------------------------------
CLUSTER_ALGO_COMPARISON = pd.DataFrame([
    {"Algorithm": "K-Means", "N_Clusters": 2, "Silhouette": 0.1889, "Davies_Bouldin": 1.8888, "Noise_Points": 0, "Noise_Pct": 0.0},
    {"Algorithm": "Hierarchical (Ward)", "N_Clusters": 2, "Silhouette": 0.1067, "Davies_Bouldin": 2.5416, "Noise_Points": 0, "Noise_Pct": 0.0},
    {"Algorithm": "DBSCAN (UMAP space)", "N_Clusters": 1, "Silhouette": np.nan, "Davies_Bouldin": np.nan, "Noise_Points": 107, "Noise_Pct": 0.535},
])

KMEANS_CLUSTER_PROFILE = pd.DataFrame([
    {
        "cluster": 0, "label": "High-Risk Subprime Borrowers",
        "size": 62_197, "pct_total": 62.2,
        "loan_amnt": 14520.18, "grade": 3.22, "annual_inc": 69729.96,
        "dti": 20.33, "fico_range_low": 681.72, "revol_util": 61.56,
        "emp_length": 5.87, "default_rate_pct": 17.07, "smallbiz_pct": 0.97,
        "color": "#E4572E",
    },
    {
        "cluster": 1, "label": "Low-Risk Prime Borrowers",
        "size": 37_803, "pct_total": 37.8,
        "loan_amnt": 15959.02, "grade": 1.76, "annual_inc": 87307.97,
        "dti": 15.33, "fico_range_low": 726.34, "revol_util": 31.70,
        "emp_length": 6.05, "default_rate_pct": 7.21, "smallbiz_pct": 1.27,
        "color": "#1B998B",
    },
])

PCA_VARIANCE = pd.DataFrame({
    "n_components": [1, 2, 3, 4, 5, 6, 7],
    "cumulative_variance_pct": [26.67, 48.97, 63.83, 77.44, 88.09, 94.42, 100.00],
})

DBSCAN_EPS_SEARCH = pd.DataFrame([
    {"eps": 0.10, "clusters": 15, "noise_pct": 99.0, "silhouette": 0.930},
    {"eps": 0.15, "clusters": 166, "noise_pct": 86.1, "silhouette": 0.640},
    {"eps": 0.20, "clusters": 291, "noise_pct": 54.8, "silhouette": 0.340},
    {"eps": 0.25, "clusters": 157, "noise_pct": 27.3, "silhouette": -0.022},
    {"eps": 0.30, "clusters": 63, "noise_pct": 11.4, "silhouette": -0.240},
    {"eps": 0.35, "clusters": 17, "noise_pct": 4.9, "silhouette": -0.401},
    {"eps": 0.40, "clusters": 9, "noise_pct": 2.2, "silhouette": -0.321},
    {"eps": 0.45, "clusters": 2, "noise_pct": 1.1, "silhouette": -0.151},
    {"eps": 0.50, "clusters": 1, "noise_pct": 0.5, "silhouette": np.nan},
])


def load_clustered_sample():
    """Kembalikan (df, is_real). df punya kolom termasuk 'kmeans_cluster'."""
    if file_available("clustered_sample"):
        return pd.read_csv(_path("clustered_sample")), True
    return _synthetic_cluster_sample(), False


def _synthetic_cluster_sample(n_per_cluster=900, seed=42):
    """Sampel ILUSTRATIF mengikuti mean asli tiap cluster (bukan data asli)."""
    rng = np.random.default_rng(seed)
    rows = []
    stds = {"annual_inc": 0.42, "dti": 0.35, "fico_range_low": 0.07, "loan_amnt": 0.45, "revol_util": 0.30}
    for _, c in KMEANS_CLUSTER_PROFILE.iterrows():
        n = n_per_cluster
        annual_inc = rng.normal(c["annual_inc"], c["annual_inc"] * stds["annual_inc"], n).clip(8000, 500000)
        dti = rng.normal(c["dti"], max(c["dti"] * stds["dti"], 3), n).clip(0, 55)
        fico = rng.normal(c["fico_range_low"], c["fico_range_low"] * stds["fico_range_low"], n).clip(620, 845)
        loan_amnt = rng.normal(c["loan_amnt"], c["loan_amnt"] * stds["loan_amnt"], n).clip(1000, 40000)
        revol_util = rng.normal(c["revol_util"], max(c["revol_util"] * stds["revol_util"], 5), n).clip(0, 100)
        grade = rng.normal(c["grade"], 1.1, n).clip(1, 7).round()
        is_bad = rng.random(n) < (c["default_rate_pct"] / 100)
        rows.append(pd.DataFrame({
            "annual_inc": annual_inc, "dti": dti, "fico_range_low": fico,
            "loan_amnt": loan_amnt, "revol_util": revol_util, "grade": grade,
            "kmeans_cluster": c["cluster"], "is_bad_loan": is_bad,
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# PHASE 3 -- Association rules (angka asli dari data-mining-phase3.ipynb)
# ---------------------------------------------------------------------------
APRIORI_STATS = {
    "boolean_shape_rows": 889_991,
    "boolean_shape_cols": 29,
    "min_support": 0.04,
    "min_confidence": 0.5,
    "min_lift": 1.15,
    "max_len": 4,
    "frequent_itemsets": 566,
    "rules_generated": 678,
    "rules_after_confidence_filter": 108,
}

# Top-15 "interesting rules" -- angka support/confidence/lift PERSIS dari
# output notebook (cell interesting_rules.head(15)), diurutkan lift desc.
TOP_RULES_RAW = pd.DataFrame([
    {"antecedents": "DTI Healthy (<20%), FICO Very Good (740-799)", "consequents": "Grade A", "support": 0.040084, "confidence": 0.625103, "lift": 3.244869},
    {"antecedents": "Grade A, FICO Very Good (740-799)", "consequents": "Revol_Util Excellent (<30%)", "support": 0.040318, "confidence": 0.727953, "lift": 3.129271},
    {"antecedents": "FICO Very Good (740-799), Revol_Util Excellent (<30%)", "consequents": "Grade A", "support": 0.040318, "confidence": 0.584214, "lift": 3.032616},
    {"antecedents": "FICO Very Good (740-799)", "consequents": "Grade A", "support": 0.055386, "confidence": 0.574611, "lift": 2.982768},
    {"antecedents": "Loan_Amnt Small (<10k), Grade A", "consequents": "Revol_Util Excellent (<30%)", "support": 0.043571, "confidence": 0.520874, "lift": 2.239093},
    {"antecedents": "DTI Healthy (<20%), Emp_Length Short (0-3 yrs)", "consequents": "Loan_Amnt Small (<10k)", "support": 0.043897, "confidence": 0.696213, "lift": 1.754895},
    {"antecedents": "Annual_Inc Low/Entry (<50k), Revol_Util Excellent", "consequents": "Loan_Amnt Small (<10k)", "support": 0.056661, "confidence": 0.691041, "lift": 1.741857},
    {"antecedents": "Annual_Inc Low/Entry (<50k), Grade B", "consequents": "Loan_Amnt Small (<10k)", "support": 0.061889, "confidence": 0.684041, "lift": 1.724212},
    {"antecedents": "DTI Healthy (<20%), Annual_Inc Low/Entry (<50k)", "consequents": "Loan_Amnt Small (<10k)", "support": 0.107201, "confidence": 0.674686, "lift": 1.700632},
    {"antecedents": "Grade C, Loan_Amnt Small (<10k)", "consequents": "Annual_Inc Low/Entry (<50k)", "support": 0.060586, "confidence": 0.537576, "lift": 1.699709},
    {"antecedents": "FICO Good (670-739), Annual_Inc Low/Entry", "consequents": "Loan_Amnt Small (<10k)", "support": 0.041704, "confidence": 0.673832, "lift": 1.698479},
    {"antecedents": "DTI Healthy (<20%), Annual_Inc Low/Entry (<50k)+", "consequents": "Loan_Amnt Small (<10k)", "support": 0.043244, "confidence": 0.670739, "lift": 1.690684},
    {"antecedents": "FICO Fair/Poor (<670), Annual_Inc Low/Entry", "consequents": "Loan_Amnt Small (<10k)", "support": 0.057570, "confidence": 0.665735, "lift": 1.678072},
    {"antecedents": "FICO Good (670-739), DTI Healthy (<20%)", "consequents": "Loan_Amnt Small (<10k)", "support": 0.064105, "confidence": 0.660404, "lift": 1.664634},
    {"antecedents": "Annual_Inc Low/Entry (<50k), Emp_Length Mid (4-7 yrs)", "consequents": "Loan_Amnt Small (<10k)", "support": 0.067624, "confidence": 0.634923, "lift": 1.600405},
])

# 15 rules bisnis terkurasi (naratif) dikelompokkan 4 tema -- teks asli notebook
BUSINESS_RULES = [
    dict(cat="A. Profil Premium", rule="{FICO Very Good (740-799), DTI Healthy (<20%)} -> {Grade A}", lift=3.24,
         insight="Kandidat kuat untuk Auto-Approval System guna memangkas waktu operasional."),
    dict(cat="A. Profil Premium", rule="{FICO Very Good (740-799)} -> {Revol_Util Excellent (<30%)}", lift=3.07,
         insight="Kelompok aman untuk penawaran kenaikan limit kartu kredit otomatis."),
    dict(cat="A. Profil Premium", rule="{Revol_Util Excellent (<30%), Grade A} -> {DTI Healthy (<20%)}", lift=1.27,
         insight="Cocok ditawarkan produk Wealth Management, bukan pinjaman konsumtif."),
    dict(cat="A. Profil Premium", rule="{Loan_Amnt Medium (10k-25k), Annual_Inc Upper-Middle (100k-150k)} -> {DTI Healthy (<20%)}", lift=1.22,
         insight="Kandidat utama untuk penawaran KPR tambahan."),
    dict(cat="B. Pendapatan Rendah & Kredit Mikro", rule="{Annual_Inc Low/Entry (<50k), Grade B} -> {Loan_Amnt Small (<10k)}", lift=1.72,
         insight="Berpotensi dijadikan paket Micro-Loan khusus dengan syarat ringkas."),
    dict(cat="B. Pendapatan Rendah & Kredit Mikro", rule="{Grade C, Loan_Amnt Small (<10k)} -> {Annual_Inc Low/Entry (<50k)}", lift=1.69,
         insight="Produk asuransi perlindungan kredit mikro perlu disematkan."),
    dict(cat="B. Pendapatan Rendah & Kredit Mikro", rule="{DTI Manageable (20-35%), Loan_Amnt Small (<10k)} -> {Annual_Inc Low/Entry (<50k)}", lift=1.78,
         insight="Pastikan pengajuan baru tidak membuat DTI menembus batas regulasi 43%."),
    dict(cat="B. Pendapatan Rendah & Kredit Mikro", rule="{Emp_Length Mid (4-7 yrs), Loan_Amnt Small (<10k)} -> {Annual_Inc Low/Entry (<50k)}", lift=1.79,
         insight="Lebih cocok ditawari pembiayaan pelatihan karier ketimbang utang konsumtif."),
    dict(cat="B. Pendapatan Rendah & Kredit Mikro", rule="{FICO Fair/Poor (<670), Annual_Inc Low/Entry (<50k)} -> {Loan_Amnt Small (<10k)}", lift=1.67,
         insight="Kandidat kartu kredit berbasis deposit untuk memperbaiki riwayat kredit."),
    dict(cat="C. Tulang Punggung Perbankan", rule="{Loan_Amnt Medium, Emp_Length Long (8+ yrs), DTI Manageable} -> {Annual_Inc Middle (50k-100k)}", lift=1.27,
         insight="Segmen loyal, motor utama perputaran bunga -- jaga dengan program retensi."),
    dict(cat="C. Tulang Punggung Perbankan", rule="{Annual_Inc Middle, Revol_Util Moderate, DTI Manageable} -> {Loan_Amnt Medium (10k-25k)}", lift=1.23,
         insight="Kandidat kampanye pre-approved limit $20,000 otomatis."),
    dict(cat="D. Pertanda Peringatan Risiko", rule="{DTI High Risk (>43%)} -> {Revol_Util Maxed Out (>90%)}", lift=None,
         insight="Pengajuan utang baru wajib ditolak otomatis hingga utilisasi turun."),
    dict(cat="D. Pertanda Peringatan Risiko", rule="{Revol_Util High (60-90%), Grade C} -> {DTI Borderline (36-43%)}", lift=None,
         insight="Perlu intervensi tim konsolidasi/restrukturisasi sebelum macet penuh."),
    dict(cat="D. Pertanda Peringatan Risiko", rule="{Emp_Length Short (0-3 yrs), Grade A} -> {FICO Exceptional (800+)}", lift=None,
         insight="Profil outlier bagus -- dorong ke produk asuransi/investasi jangka panjang."),
    dict(cat="D. Pertanda Peringatan Risiko", rule="{Grade D, Emp_Length Long (8+ yrs)} -> {Loan_Amnt Medium (10k-25k)}", lift=None,
         insight="Validasi dokumen fisik tetap wajib, jangan andalkan model semata."),
]
BUSINESS_RULES_DF = pd.DataFrame(BUSINESS_RULES)


def load_rules():
    if file_available("rules"):
        df = pd.read_csv(_path("rules"))
        # Sort by lift descending dan ambil top-15 agar konsisten dengan
        # label "Tabel Top-15 Rule" di app.py dan tidak membuat bubble chart
        # terlalu padat saat semua 100+ rules di-plot sekaligus.
        df = df.sort_values("lift", ascending=False).head(15).reset_index(drop=True)
        return df, True
    return TOP_RULES_RAW.copy(), False


# ---------------------------------------------------------------------------
# PHASE 4 -- Anomaly detection (angka asli dari data-mining-phase4.ipynb)
# ---------------------------------------------------------------------------
ANOMALY_METHOD_COUNTS = pd.DataFrame([
    {"method": "IQR (univariate)", "count": 87_574, "pct": 9.84},
    {"method": "Robust Z-Score (MAD)", "count": 53_154, "pct": 5.97},
    {"method": "Isolation Forest", "count": 44_500, "pct": 5.00},
    {"method": "DBSCAN noise (subset 20k)", "count": 107, "pct": 0.535},
])

ANOMALY_CONFIDENCE = pd.DataFrame([
    {"confidence": "Normal (0 metode)", "count": 796_296, "pct": 89.47},
    {"confidence": "Low (1 metode)", "count": 31_734, "pct": 3.57},
    {"confidence": "Medium (2 metode)", "count": 32_288, "pct": 3.63},
    {"confidence": "High (>=3 metode)", "count": 29_673, "pct": 3.33},
])

ANOMALY_TYPOLOGY = pd.DataFrame([
    {"typology": "Rare Legitimate Case", "count": 33_281, "pct": 53.7,
     "desc": "Kasus valid namun jarang (mis. gaji tinggi, pinjaman kecil).", "color": "#1B998B"},
    {"typology": "Potential Risk Signal", "count": 19_808, "pct": 32.0,
     "desc": "Indikasi risiko gagal bayar tinggi (mis. FICO rendah + pinjaman besar).", "color": "#E4572E"},
    {"typology": "Unclassified Outlier", "count": 8_861, "pct": 14.3,
     "desc": "Anomali struktural yang butuh peninjauan analis manual.", "color": "#F4A261"},
    {"typology": "Data Error", "count": 11, "pct": 0.02,
     "desc": "Nilai mustahil secara finansial (negatif / di luar rentang valid).", "color": "#6C757D"},
])

ANOMALY_STRONG_TOTAL = 61_961  # Anomaly_Score >= 2 (disepakati >=2 metode)
ANOMALY_TOTAL_ROWS = 889_991
ISO_FOREST_SKEWNESS = pd.DataFrame([
    {"feature": "annual_inc", "skewness": 491.06},
    {"feature": "dti", "skewness": 30.71},
    {"feature": "fico_range_low", "skewness": 1.19},
    {"feature": "loan_amnt", "skewness": 0.78},
    {"feature": "revol_util", "skewness": 0.04},
])


def load_anomaly_report():
    if file_available("anomaly_report"):
        return pd.read_csv(_path("anomaly_report")), True
    return _synthetic_anomaly_sample(), False


def _synthetic_anomaly_sample(seed=7):
    """Sampel ILUSTRATIF per tipologi anomali (bukan data asli 61,961 baris)."""
    rng = np.random.default_rng(seed)
    n_map = {"Rare Legitimate Case": 300, "Potential Risk Signal": 300,
             "Unclassified Outlier": 150, "Data Error": 15}
    rows = []
    for _, t in ANOMALY_TYPOLOGY.iterrows():
        n = n_map[t["typology"]]
        if t["typology"] == "Rare Legitimate Case":
            income = rng.uniform(150000, 450000, n)
            loan = rng.uniform(1000, 12000, n)
            dti = rng.uniform(2, 15, n)
            fico = rng.uniform(720, 845, n)
        elif t["typology"] == "Potential Risk Signal":
            income = rng.uniform(18000, 55000, n)
            loan = rng.uniform(15000, 40000, n)
            dti = rng.uniform(32, 55, n)
            fico = rng.uniform(620, 690, n)
        elif t["typology"] == "Data Error":
            income = rng.choice([-5000, 999999999], n)
            loan = rng.uniform(1000, 40000, n)
            dti = rng.uniform(1000, 5000, n)
            fico = rng.choice([300, 950], n)
        else:
            income = rng.uniform(40000, 200000, n)
            loan = rng.uniform(5000, 35000, n)
            dti = rng.uniform(15, 45, n)
            fico = rng.uniform(650, 800, n)
        rows.append(pd.DataFrame({
            "annual_inc": income, "loan_amnt": loan, "dti": dti, "fico_range_low": fico,
            "Anomaly_Typology": t["typology"],
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# KDD Pipeline funnel (dipakai di tab Overview)
# ---------------------------------------------------------------------------
KDD_FUNNEL = pd.DataFrame([
    {"stage": "Raw data (sampled)", "rows": 890_000},
    {"stage": "Setelah cleaning & feature selection (Phase 1)", "rows": 889_991},
    {"stage": "Sampel untuk clustering (Phase 2)", "rows": 100_000},
    {"stage": "Anomali kuat / Anomaly_Score>=2 (Phase 4)", "rows": ANOMALY_STRONG_TOTAL},
])
