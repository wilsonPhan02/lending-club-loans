"""
Konfigurasi terpusat pipeline Phase 1 (Lending Club).
Semua parameter, ambang, dan daftar kolom domain didefinisikan di sini, bukan tersebar
di dalam modul, agar setiap keputusan dapat ditinjau & dijustifikasi di satu tempat.
"""
import os

# Anchor path ke root proyek via __file__ (robust, tak bergantung working directory).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def _raw(f):  return os.path.join(_ROOT, "data", "raw", f)
def _proc(f): return os.path.join(_ROOT, "data", "processed", f)

# ---------------------------------------------------------------- Data & umum
DATA_PATH   = _raw("accepted_2007_to_2018Q4.csv")
USE_FULL    = True            # True = 2,26 juta baris; False = sample SAMPLE_SIZE
SAMPLE_SIZE = 890_000
RANDOM_SEED = 42
TOTAL_ROWS  = 2_260_701       # untuk mode sampling (skiprows)

# ---------------------------------------------------------------- Cleaning
# Identifier / teks bebas berkardinalitas tinggi -> tidak analitis untuk clustering
ID_TEXT_COLS = ["id", "member_id", "url", "emp_title", "title", "desc", "zip_code"]

# Kolom outcome pasca-origination (data leakage) -> tidak boleh jadi fitur profil peminjam
# Catatan: collections_12_mths_ex_med sebenarnya atribut riwayat kredit pra-origination;
# ikut di-drop secara konservatif karena zero-inflated (>98% bernilai 0) dan ambigu
# terhadap momen pencatatan, bukan karena leakage murni.
POST_LOAN_COLS = [
    "funded_amnt", "funded_amnt_inv", "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_amnt",
    "last_fico_range_high", "last_fico_range_low", "collections_12_mths_ex_med",
]
# Prefix kolom post-loan (hardship/settlement) yang juga di-drop
POST_LOAN_PREFIXES = ("hardship_", "settlement_", "debt_settlement")
# Kolom tanggal string yang tak dipakai langsung untuk clustering
DATE_COLS = ["issue_d", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d", "earliest_cr_line"]

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10,
}
MTHS_SENTINEL   = 999   # 'tak pernah terjadi' (>> maks teramati ~176; satuan bulan)
MISS_THRESHOLD  = 40    # % missing; di atas ini kolom di-drop (lazim 30-50%)

# ---------------------------------------------------------------- Transform
NOMINAL_FOR_ARM = ["purpose", "home_ownership", "verification_status", "addr_state"]
GRADE_MAP       = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
# Nominal murni -> di-drop dari matriks clustering (bukan one-hot: mendistorsi jarak Euclidean)
DROP_NOMINAL    = ["home_ownership", "purpose", "addr_state",
                   "verification_status", "pymnt_plan", "disbursement_method"]

BIN_SPECS = {
    "annual_inc":     (4, ["Low", "Medium", "High", "Very High"]),
    "dti":            (4, ["Low", "Medium", "High", "Very High"]),
    "int_rate":       (4, ["Low", "Medium", "High", "Very High"]),
    "loan_amnt":      (4, ["Small", "Medium", "Large", "Very Large"]),
    "fico_range_low": (5, ["Very Low", "Low", "Medium", "High", "Very High"]),
}

WINSORIZE   = True
WINSOR_LOW, WINSOR_HIGH = 0.01, 0.99
# Winsorization bersifat DATA-DRIVEN (bukan daftar hardcode): setiap fitur kontinu
# ber-|skew| > ambang ini di-cap 1/99, dengan guard q99>median agar fitur zero-inflated
# tidak kolaps. Menangani heavy-tail secara sistematis, termasuk fitur yang baru terpilih.
WINSOR_SKEW_THRESHOLD = 1.0
LOG_SKEW_THRESHOLD = 1.0     # |skew| > ini -> kandidat log (diterapkan hanya bila memperbaiki)
QUASI_CONST_RATIO  = 0.99    # nilai dominan >= ini -> fitur quasi-constant, di-drop
# Fitur inti mining angle -> diproteksi dari drop quasi-constant
PROTECT_COLS = {
    "grade", "sub_grade", "annual_inc", "dti", "int_rate", "fico_range_low",
    "fico_range_high", "emp_length", "loan_amnt", "revol_util", "term",
}
# Kolom ordinal/biner/flag -> dikecualikan dari log transform
LOG_EXCLUDE_BASE = ["grade", "sub_grade", "term", "emp_length", "is_joint", "initial_list_status"]

# ---------------------------------------------------------------- Feature selection
MI_TARGET       = "grade"
CORR_THRESHOLD  = 0.85       # |r| > ini -> multikolinearitas, buang salah satu
# Prioritas mempertahankan saat pasangan redundan (makin awal makin diprioritaskan)
CORR_PRIORITY   = ["grade", "annual_inc", "dti", "int_rate", "fico_range_low", "loan_amnt",
                   "emp_length", "revol_util", "term", "open_acc", "mort_acc"]
MI_SAMPLE       = 50_000
MI_NEIGHBORS    = 5
MINING_CORE     = ["grade", "annual_inc", "dti", "fico_range_low", "revol_util",
                   "emp_length", "loan_amnt", "term"]
TARGET_K        = 12         # ukuran subset final (selaras mining angle + interpretabilitas)
CORR_GUARD      = 0.70       # tolak fitur MI-driven yang redundan (|r| > ini) dengan subset

# ---------------------------------------------------------------- Dimensionality reduction
PCA_VARIANCE   = 0.90
UMAP_SAMPLE    = 100_000
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST  = 0.1
# densMAP: varian UMAP yang mempertahankan densitas lokal -> embedding lebih setia
# untuk deteksi outlier berbasis densitas (DBSCAN) di Phase 2. Set False = UMAP biasa.
UMAP_DENSMAP   = True

# ---------------------------------------------------------------- Output (di data/processed/)
OUTPUTS = {
    "cleaned":        _proc("cleaned_lending_club.csv"),                 # subset final, TER-TRANSFORM (winsor+log); untuk profiling bisnis pakai no_winsor
    "no_winsor":      _proc("cleaned_lending_club_no_winsorization.csv"),# nilai asli -> Phase 4 anomaly
    "scaled":         _proc("scaled_lending_club.csv"),                  # subset final, scaled
    "pca":            _proc("lending_club_pca.csv"),                     # -> K-Means / Hierarchical
    "umap":           _proc("lending_club_umap.csv"),                    # -> DBSCAN
    "arm":            _proc("lending_club_apriori_binned.csv"),          # -> Phase 3 Apriori
    "loan_status":    _proc("loan_status_reference.csv"),                # label referensi
}
