"""
data_layer.py
=================
Lapisan data untuk Dashboard Phase 5 (Lending Club KDD Project).

Filosofi loading data:
-----------------------------------------------------
1. Notebook Phase 1-4 menghasilkan beberapa file CSV di data/processed/.
   Modul ini menyalin subset yang relevan ke csv/ (lihat DATA_FILES) supaya
   dashboard dapat dijalankan mandiri (folder phase5/ saja) tanpa membaca
   data/processed/ langsung.
2. Semua angka agregat di bawah (PHASE1_SUMMARY, CLUSTER_ALGO_COMPARISON, dst.)
   adalah angka aktual dari output notebook Phase 1-4 yang telah dieksekusi
   dan diverifikasi, bukan estimasi. Field yang tidak pernah dihitung pada
   notebook (misalnya Davies-Bouldin, Silhouette per-algoritma di luar
   K-Means) dibiarkan kosong (NaN) alih-alih diisi angka karangan.
3. Untuk scatter plot level-baris, `load_*` membaca CSV asli di csv/ bila
   tersedia (is_real=True). Bila file itu tidak ada (misalnya csv/ tidak
   ikut disalin), fallback membangun sampel ilustratif dari statistik
   agregat asli di atas, diberi label "sampel ilustratif" pada dashboard.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILES = {
    "clustered_sample": "csv/phase2_clustered_sample.csv",
    "dbscan_outliers": "csv/phase2_dbscan_outliers.csv",
    "anomaly_report": "csv/phase4_anomaly_report.csv",
    "rules": "csv/phase3_association_rules.csv",
    "umap_sample": "csv/phase2_umap_sample.csv",
}


def _path(key):
    return os.path.join(BASE_DIR, DATA_FILES[key])


def file_available(key):
    return os.path.exists(_path(key))


# ---------------------------------------------------------------------------
# PHASE 1 -- Preprocessing summary (angka aktual dari notebook Phase 1)
# ---------------------------------------------------------------------------
PHASE1_SUMMARY = {
    "source": "Lending Club accepted_2007_to_2018Q4.csv",
    "raw_rows": 2_260_701,
    "raw_cols": 151,
    "final_rows": 2_260_668,
    "final_cols": 12,
    "cols_gt50pct_missing_dropped": 44,
    "cols_20_50pct_missing": 14,
    "cols_lt20pct_missing_imputed": 92,
    "final_features": [
        "grade", "annual_inc", "dti", "fico_range_low", "revol_util",
        "emp_length", "loan_amnt", "term",
        "bc_open_to_buy", "total_bc_limit", "all_util", "inq_last_6mths",
    ],
    "dropped_high_corr_n": 16,
    "dropped_high_corr_example": "int_rate (redundan dengan grade, |r|>0.85)",
    "mining_angle": "Segmentasi peminjam berdasarkan profil risiko: grade, income, dan debt ratio",
    "mining_rule_target": (
        "Contoh pada dokumen penugasan: peminjam dengan tujuan small business dan "
        "masa kerja 10+ tahun cenderung mendapat Grade A dengan bunga rendah -- "
        "TIDAK terkonfirmasi pada data aktual (0 dari 1.023 rule retained melibatkan small_business)."
    ),
}

# ---------------------------------------------------------------------------
# PHASE 2 -- Clustering results (angka aktual dari notebook Phase 2)
# ---------------------------------------------------------------------------
# Silhouette dihitung untuk K-Means (K final, rata-rata 5 sample). Hierarchical
# dan DBSCAN tidak memiliki Silhouette standalone yang dihitung/divalidasi di
# notebook -- dibiarkan NaN alih-alih diisi angka yang tidak pernah dikomputasi.
CLUSTER_ALGO_COMPARISON = pd.DataFrame([
    {"Algorithm": "K-Means", "N_Clusters": 2, "Silhouette": 0.161, "Noise_Points": 0, "Noise_Pct": 0.0},
    {"Algorithm": "Hierarchical (Ward)", "N_Clusters": 2, "Silhouette": np.nan, "Noise_Points": 0, "Noise_Pct": 0.0},
    {"Algorithm": "DBSCAN (UMAP densMAP)", "N_Clusters": 5, "Silhouette": np.nan, "Noise_Points": 501, "Noise_Pct": 0.50},
])
HIERARCHICAL_COPHENETIC = 0.360   # linkage Ward, sample 12.000
HIERARCHICAL_ARI = 0.365          # Adjusted Rand Index K-Means vs Hierarchical

KMEANS_CLUSTER_PROFILE = pd.DataFrame([
    {
        "cluster": 0, "label": "Higher-Risk Borrowers",
        "size": 1_407_028, "pct_total": 62.2,
        "loan_amnt": 12086.5, "grade": 3.0, "annual_inc": 63474.5,
        "dti": 19.6, "fico_range_low": 684.1, "revol_util": 58.4,
        "emp_length": 5.7, "default_rate_pct": 14.4,
        "color": "#E4572E",
    },
    {
        "cluster": 1, "label": "Prime Borrowers",
        "size": 853_640, "pct_total": 37.8,
        "loan_amnt": 19926.6, "grade": 2.1, "annual_inc": 101921.8,
        "dti": 17.5, "fico_range_low": 722.5, "revol_util": 37.0,
        "emp_length": 6.4, "default_rate_pct": 7.7,
        "color": "#1B998B",
    },
])

PCA_VARIANCE = pd.DataFrame({
    "n_components": list(range(1, 10)),
    "cumulative_variance_pct": [24.8, 41.5, 52.1, 61.6, 69.9, 78.0, 83.1, 87.9, 91.5],
})


def load_umap_sample():
    """Embedding UMAP 2D (sample render 40k dari 100k) + label K-Means & noise DBSCAN.
    Kembalikan (df, is_real); None bila file tidak tersedia (chart dilewati)."""
    if file_available("umap_sample"):
        return pd.read_csv(_path("umap_sample")), True
    return None, False


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
# PHASE 3 -- Association rules (angka aktual dari notebook Phase 3)
# ---------------------------------------------------------------------------
APRIORI_STATS = {
    "sample_rows": 250_000,          # sample acak seeded, bukan populasi penuh (2.260.668)
    "boolean_shape_cols": 53,
    "min_support": 0.04,
    "min_confidence": 0.5,
    "min_lift": 1.15,
    "max_len": 4,
    "frequent_itemsets": 1_947,
    "rules_generated": 14_764,       # sebelum filtering
    "rules_after_filter": 1_023,     # setelah lift>1.15 & confidence>0.5
    "lift_min": 1.15,
    "lift_max": 5.35,
}

# Fallback top-15 rule (dipakai hanya bila csv/phase3_association_rules.csv
# tidak tersedia) -- disalin persis dari data/processed/phase3_association_rules.csv,
# 15 baris teratas terurut menurun berdasarkan lift.
TOP_RULES_RAW = pd.DataFrame([
    {"antecedents": "grade=Grade A, loan_amnt=Loan Small(<10k)", "consequents": "int_rate=IntRate Low, revol_util=Util Excellent(<30)", "support": 0.042, "confidence": 0.515, "lift": 5.35},
    {"antecedents": "grade=Grade A, revol_util=Util Moderate(30-60)", "consequents": "fico_range_low=FICO Good(670-739), int_rate=IntRate Low", "support": 0.058, "confidence": 0.774, "lift": 4.83},
    {"antecedents": "grade=Grade A, revol_util=Util Excellent(<30)", "consequents": "int_rate=IntRate Low, loan_amnt=Loan Small(<10k)", "support": 0.042, "confidence": 0.524, "lift": 4.81},
    {"antecedents": "int_rate=IntRate Low, revol_util=Util Moderate(30-60)", "consequents": "fico_range_low=FICO Good(670-739), grade=Grade A", "support": 0.058, "confidence": 0.573, "lift": 4.76},
    {"antecedents": "emp_length=Emp Long(8+), grade=Grade A", "consequents": "home=Home MORTGAGE, int_rate=IntRate Low", "support": 0.051, "confidence": 0.650, "lift": 4.72},
    {"antecedents": "fico_range_low=FICO VeryGood(740-799), int_rate=IntRate Low", "consequents": "grade=Grade A", "support": 0.054, "confidence": 0.890, "lift": 4.70},
    {"antecedents": "int_rate=IntRate Low, revol_util=Util Excellent(<30)", "consequents": "dti=DTI Healthy(<20), grade=Grade A", "support": 0.060, "confidence": 0.629, "lift": 4.70},
    {"antecedents": "annual_inc=Inc Low(<50k), int_rate=IntRate High", "consequents": "grade=Grade C, loan_amnt=Loan Small(<10k)", "support": 0.047, "confidence": 0.530, "lift": 4.69},
    {"antecedents": "annual_inc=Inc Low(<50k), grade=Grade B", "consequents": "int_rate=IntRate Medium, loan_amnt=Loan Small(<10k)", "support": 0.047, "confidence": 0.516, "lift": 4.59},
    {"antecedents": "int_rate=IntRate Low, purpose=credit_card", "consequents": "fico_range_low=FICO Good(670-739), grade=Grade A", "support": 0.043, "confidence": 0.549, "lift": 4.56},
    {"antecedents": "home=Home MORTGAGE, int_rate=IntRate Low, revol_util=Util Excellent(<30)", "consequents": "grade=Grade A", "support": 0.043, "confidence": 0.849, "lift": 4.48},
    {"antecedents": "dti=DTI Healthy(<20), int_rate=IntRate Low, revol_util=Util Excellent(<30)", "consequents": "grade=Grade A", "support": 0.060, "confidence": 0.844, "lift": 4.46},
    {"antecedents": "int_rate=IntRate Low, loan_amnt=Loan Small(<10k), revol_util=Util Excellent(<30)", "consequents": "grade=Grade A", "support": 0.042, "confidence": 0.843, "lift": 4.45},
    {"antecedents": "grade=Grade A, loan_amnt=Loan Medium(10-25k)", "consequents": "annual_inc=Inc Mid(50-100k), int_rate=IntRate Low", "support": 0.047, "confidence": 0.550, "lift": 4.45},
    {"antecedents": "annual_inc=Inc Low(<50k), int_rate=IntRate Medium", "consequents": "grade=Grade B, loan_amnt=Loan Small(<10k)", "support": 0.047, "confidence": 0.585, "lift": 4.42},
])

# 12 rule bisnis terkurasi (non-circular; tanpa grade & int_rate kecuali
# dicatat khusus) -- angka & interpretasi identik dengan Section 9.3 notebook
# Phase 3 dan Finding 2 pada Knowledge Discovery Report.
BUSINESS_RULES = [
    dict(cat="A. Kredit & Utilisasi Sehat", rule="{DTI Healthy(<20%), FICO Very Good(740-799)} -> {Revol Util Excellent(<30%)}", lift=3.18,
         insight="Peminjam berkredit sangat baik & utang ringan menjaga utilisasi rendah -- aman menawarkan kenaikan limit kartu kredit."),
    dict(cat="A. Kredit & Utilisasi Sehat", rule="{FICO Very Good(740-799)} -> {Revol Util Excellent(<30%)}", lift=3.07,
         insight="Versi satu-atribut dari rule di atas: FICO tinggi konsisten berasosiasi dengan disiplin utilisasi."),
    dict(cat="B. Pendapatan Rendah & Pinjaman Kecil", rule="{DTI Manageable, Home RENT, Loan Small(<10k)} -> {Annual Inc Low(<50k)}", lift=2.15,
         insight="Penyewa dengan pinjaman kecil = segmen income rendah -- cocok untuk produk kredit mikro/starter."),
    dict(cat="B. Pendapatan Rendah & Pinjaman Kecil", rule="{Home RENT, Loan Small(<10k)} -> {Annual Inc Low(<50k)}", lift=1.85,
         insight="Pola masif (support 11% populasi): penyewa + pinjaman kecil hampir selalu income rendah."),
    dict(cat="B. Pendapatan Rendah & Pinjaman Kecil", rule="{Annual Inc Low(<50k), DTI Healthy, Home RENT} -> {Loan Small(<10k)}", lift=1.80,
         insight="Segmen entry-level konservatif -- bukan kandidat produk pinjaman besar."),
    dict(cat="B. Pendapatan Rendah & Pinjaman Kecil", rule="{Annual Inc Low(<50k), Revol Util Excellent} -> {Loan Small(<10k)}", lift=1.75,
         insight="Berpendapatan rendah tapi disiplin kredit tetap mengambil pinjaman kecil."),
    dict(cat="B. Pendapatan Rendah & Pinjaman Kecil", rule="Annual Inc Low(<50k) -> Loan Small(<10k)", lift=1.55,
         insight="Korelasi inti terbesar (support 19,5% populasi): income rendah hampir selalu berpasangan pinjaman kecil."),
    dict(cat="C. Tujuan Pinjaman & Kepemilikan Rumah", rule="{purpose=home_improvement} -> {Home MORTGAGE}", lift=1.55,
         insight="Pemohon dana renovasi hampir pasti pemilik rumah -- tawarkan produk home-equity/HELOC."),
    dict(cat="C. Tujuan Pinjaman & Kepemilikan Rumah", rule="{Annual Inc Upper(100-150k), Emp Long(8+)} -> {Home MORTGAGE}", lift=1.44,
         insight="Mapan + kerja lama -> kandidat kuat produk KPR tambahan/refinancing."),
    dict(cat="C. Tujuan Pinjaman & Kepemilikan Rumah", rule="{Emp Long(8+), Loan Large(>25k)} -> {Home MORTGAGE}", lift=1.40,
         insight="Pinjaman besar + kerja lama berasosiasi dengan kepemilikan rumah (jaminan implisit)."),
    dict(cat="D. Catatan Metodologis: Grade vs Suku Bunga", rule="{FICO Very Good(740-799)} -> {Grade A}", lift=2.99,
         insight="Satu-satunya dari 1.023 rule yang memprediksi Grade A dari atribut peminjam TANPA suku bunga -- konsisten karena FICO memang input langsung model grading."),
    dict(cat="D. Catatan Metodologis: Grade vs Suku Bunga", rule="{Grade A} <-> {IntRate Low}   (semua rule ber-lift tertinggi memuat pasangan ini)", lift=4.00,
         insight="Grade internal Lending Club nyaris sepenuhnya cerminan suku bunga -- jangan pakai keduanya sebagai fitur independen di model risiko hilir."),
]
BUSINESS_RULES_DF = pd.DataFrame(BUSINESS_RULES)


def load_rules():
    if file_available("rules"):
        df = pd.read_csv(_path("rules"))
        df = df.sort_values("lift", ascending=False).head(15).reset_index(drop=True)
        return df, True
    return TOP_RULES_RAW.copy(), False


# ---------------------------------------------------------------------------
# PHASE 4 -- Anomaly detection (angka aktual dari notebook Phase 4)
# ---------------------------------------------------------------------------
# Jumlah baris ter-flag TIAP metode (dihitung independen, boleh tumpang tindih
# antar metode -- lihat corroboration di bawah). Basis: 2.260.668 baris.
ANOMALY_METHOD_COUNTS = pd.DataFrame([
    {"method": "IQR (3x, univariat)", "count": 236_492, "pct": 10.46},
    {"method": "Z-score (|z|>3)", "count": 143_542, "pct": 6.35},
    {"method": "Mahalanobis (chi2 99.9%)", "count": 47_104, "pct": 2.08},
    {"method": "Isolation Forest (1%)", "count": 22_607, "pct": 1.00},
])

# Distribusi jumlah metode yang menandai tiap baris (0-4), dari 2.260.668 total.
ANOMALY_CONFIDENCE = pd.DataFrame([
    {"confidence": "Normal (0 metode)", "count": 1_989_525, "pct": 88.01},
    {"confidence": "Low (1 metode)", "count": 154_918, "pct": 6.85},
    {"confidence": "Medium (2 metode)", "count": 69_621, "pct": 3.08},
    {"confidence": "High (>=3 metode)", "count": 46_604, "pct": 2.06},
])

# Klasifikasi anomali terkonfirmasi (>=2 metode, n=116.225): data_error / risk_signal
# / rare_case -- lihat notebook Phase 4 Section 6-6c untuk metodologi anti-circularity
# (risk_signal dinilai dari atribut pra-pinjaman, DIVALIDASI ke outcome, bukan
# didefinisikan dari outcome).
ANOMALY_TYPOLOGY = pd.DataFrame([
    {"typology": "Rare Legitimate Case", "count": 102_213, "pct": 87.9,
     "desc": "Anomali statistik yang plausibel dan bukan sinyal risiko -- mayoritas profil affluent/limit kartu besar yang membayar lancar.", "color": "#1B998B"},
    {"typology": "Risk Signal", "count": 11_358, "pct": 9.8,
     "desc": "Tekanan kredit tinggi (DTI, utilisasi, inquiry) dinilai independen dari status pinjaman, lalu tervalidasi: bad-rate 19,2% vs populasi 13,1% (lift 1,47x).", "color": "#E4572E"},
    {"typology": "Data Error", "count": 2_654, "pct": 2.3,
     "desc": "Nilai tidak plausibel secara finansial (mis. annual_inc > $5 juta, atau DTI bernilai sentinel 999).", "color": "#6C757D"},
])

ANOMALY_STRONG_TOTAL = 116_225   # terkonfirmasi >=2 metode
ANOMALY_CANDIDATES_TOTAL = 271_143   # >=1 metode, sebelum corroboration
ANOMALY_TOTAL_ROWS = 2_260_668
ANOMALY_BADRATE_POPULATION = 13.1
ANOMALY_BADRATE_ALL_CONFIRMED = 8.7
ANOMALY_BADRATE_RISK_SIGNAL = 19.2


def load_anomaly_report():
    if file_available("anomaly_report"):
        return pd.read_csv(_path("anomaly_report")), True
    return _synthetic_anomaly_sample(), False


def _synthetic_anomaly_sample(seed=7):
    """Sampel ILUSTRATIF per tipologi anomali (bukan data asli 116.225 baris)."""
    rng = np.random.default_rng(seed)
    n_map = {"Rare Legitimate Case": 400, "Risk Signal": 300, "Data Error": 40}
    rows = []
    for _, t in ANOMALY_TYPOLOGY.iterrows():
        n = n_map[t["typology"]]
        if t["typology"] == "Rare Legitimate Case":
            income = rng.uniform(150000, 450000, n)
            loan = rng.uniform(1000, 12000, n)
            dti = rng.uniform(2, 15, n)
            fico = rng.uniform(700, 845, n)
        elif t["typology"] == "Risk Signal":
            income = rng.uniform(18000, 55000, n)
            loan = rng.uniform(15000, 40000, n)
            dti = rng.uniform(32, 55, n)
            fico = rng.uniform(620, 690, n)
        else:  # Data Error
            income = rng.choice([-5000, 9_000_000], n)
            loan = rng.uniform(1000, 40000, n)
            dti = rng.uniform(500, 999, n)
            fico = rng.choice([300, 950], n)
        rows.append(pd.DataFrame({
            "annual_inc": income, "loan_amnt": loan, "dti": dti, "fico_range_low": fico,
            "Anomaly_Typology": t["typology"],
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# KDD Pipeline funnel (dipakai di tab Overview)
# ---------------------------------------------------------------------------
KDD_FUNNEL = pd.DataFrame([
    {"stage": "Raw data (2007-2018)", "rows": PHASE1_SUMMARY["raw_rows"]},
    {"stage": "Setelah cleaning & feature selection (Phase 1)", "rows": PHASE1_SUMMARY["final_rows"]},
    {"stage": "Sampel utk DBSCAN/Hierarchical (UMAP, Phase 2)", "rows": 100_000},
    {"stage": "Anomali kuat terkonfirmasi >=2 metode (Phase 4)", "rows": ANOMALY_STRONG_TOTAL},
])
