"""
Phase 4 — Anomaly & Outlier Detection (Lending Club)
=====================================================
Metode: IQR + Z-score (univariat statistik), Mahalanobis (multivariat),
Isolation Forest (struktural). Anomali dikorroborasi (>=2 metode), di-cross-reference
dengan noise DBSCAN Phase 2, lalu diklasifikasi HANYA dari plausibilitas atribut:
data_error (mustahil/placeholder) vs rare_case (ekstrem tapi plausibel).
Dimensi risiko dinilai TERPISAH dari atribut pra-pinjaman dan DIVALIDASI terhadap
outcome (loan_status) — bukan didefinisikan dari outcome (menghindari sirkularitas).

WAJIB memakai data NON-winsorized (nilai asli) — winsorization menghapus anomali.

Jalankan:  python phase4_anomaly.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Anchor path ke data/processed via __file__ (script ada di <root>/pipeline/).
_PROC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
def _p(f): return os.path.join(_PROC, f)

RANDOM_SEED = 42
IQR_MULT   = 3.0     # 3x IQR = outlier "ekstrem" (bukan 1.5x yang terlalu longgar pd data masif)
Z_THRESH   = 3.0     # |z| > 3 = di luar 99.7% distribusi normal
MAHA_PCTL  = 0.999   # persentil chi-square utk ambang Mahalanobis
IF_CONTAM  = 0.01    # Isolation Forest: asumsikan ~1% anomali

# ---------------------------------------------------------------- 1. Load
print('=== 1. Load data (NON-winsorized = nilai asli) ===')
df   = pd.read_csv(_p('cleaned_lending_club_no_winsorization.csv'))
loan = pd.read_csv(_p('loan_status_reference.csv'))['loan_status']
dbs  = pd.read_csv(_p('phase2_dbscan_labels.csv'))   # orig_index, dbscan_label, is_noise
assert len(df) == len(loan), 'baris tidak selaras'
print(f'  df: {df.shape} | loan_status: {len(loan):,} | dbscan sample: {len(dbs):,}')

# Fitur untuk deteksi (numerik; grade ordinal disertakan tapi minim outlier)
feats = [c for c in df.columns if c != 'grade']
X = df[feats].values.astype(float)
Xz = StandardScaler().fit_transform(X)   # standar utk Mahalanobis & Isolation Forest

# ---------------------------------------------------------------- 2. IQR + Z-score (univariat)
print('\n=== 2. Univariat: IQR (3x) + Z-score (|z|>3) ===')
iqr_flag = np.zeros(len(df), dtype=bool)
z_flag   = np.zeros(len(df), dtype=bool)
per_feat = {}
for j, c in enumerate(feats):
    col = X[:, j]
    q1, q3 = np.percentile(col, [25, 75]); iqr = q3 - q1
    lo, hi = q1 - IQR_MULT * iqr, q3 + IQR_MULT * iqr
    fi = (col < lo) | (col > hi)
    std = col.std()
    fz = (np.abs((col - col.mean()) / std) > Z_THRESH) if std > 0 else np.zeros(len(col), bool)
    iqr_flag |= fi; z_flag |= fz
    per_feat[c] = (int(fi.sum()), int(fz.sum()))
print(f'  {"fitur":22s} {"IQR>3x":>8s} {"|z|>3":>8s}')
for c, (a, b) in per_feat.items():
    print(f'  {c:22s} {a:>8,} {b:>8,}')
print(f'  -> baris ter-flag IQR: {iqr_flag.sum():,} | Z-score: {z_flag.sum():,}')

# ---------------------------------------------------------------- 3. Mahalanobis (multivariat)
print('\n=== 3. Multivariat: Mahalanobis (chi-square 99.9%) ===')
mu  = Xz.mean(axis=0)
inv = np.linalg.pinv(np.cov(Xz, rowvar=False))
diff = Xz - mu
md2 = np.einsum('ij,jk,ik->i', diff, inv, diff)     # jarak Mahalanobis kuadrat
thr = chi2.ppf(MAHA_PCTL, df=len(feats))
maha_flag = md2 > thr
print(f'  ambang chi2({len(feats)}, {MAHA_PCTL}) = {thr:.1f} | ter-flag: {maha_flag.sum():,}')

# ---------------------------------------------------------------- 4. Isolation Forest (struktural)
print('\n=== 4. Isolation Forest (contamination=1%) ===')
iso = IsolationForest(contamination=IF_CONTAM, random_state=RANDOM_SEED, n_jobs=-1)
if_pred = iso.fit_predict(Xz)
if_flag = if_pred == -1
print(f'  ter-flag Isolation Forest: {if_flag.sum():,}')

# ---------------------------------------------------------------- 5. Korroborasi + cross-ref DBSCAN
print('\n=== 5. Korroborasi (>=2 metode) + cross-reference DBSCAN ===')
flags = pd.DataFrame({'IQR': iqr_flag, 'Zscore': z_flag,
                      'Mahalanobis': maha_flag, 'IsolationForest': if_flag})
n_methods = flags.sum(axis=1)
corroborated = n_methods >= 2
print(f'  total kandidat (>=1 metode): {(n_methods>=1).sum():,}')
print(f'  terkonfirmasi (>=2 metode) : {corroborated.sum():,}')
print(f'  terkonfirmasi (>=3 metode) : {(n_methods>=3).sum():,}')

noise_idx = dbs.loc[dbs['is_noise'] == 1, 'orig_index'].values
db_nmethods = n_methods.values[noise_idx]
print(f'  DBSCAN noise (n={len(noise_idx)}): juga ter-flag >=1 metode statistik/IF: '
      f'{(db_nmethods>=1).sum()} ({(db_nmethods>=1).mean()*100:.0f}%), '
      f'>=2 metode: {(db_nmethods>=2).sum()}')

# ---------------------------------------------------------------- 6. Deteksi data_error (plausibilitas)
# Rubrik menuntut 3 kategori interpretasi: data error / rare case / risk signal.
# Agar TIDAK sirkular, tak satupun kategori didefinisikan dari loan_status:
#   - data_error  : nilai mustahil/placeholder (Bagian 6, plausibilitas atribut)
#   - risk_signal : anomali high-stress dari atribut pra-pinjaman (Bagian 6b) — DIVALIDASI oleh outcome
#   - rare_case   : sisanya (ekstrem tapi plausibel, bukan high-stress)
print('\n=== 6. Deteksi data_error (implausibilitas nilai atribut) ===')
inc = df['annual_inc'].values
dti = df['dti'].values

idx_conf = np.where(corroborated.values)[0]
# Ambang implausibilitas berjustifikasi domain (bukan sirkular):
#   annual_inc > $5jt : peminjam pinjaman konsumen <$40rb dgn income segini tak masuk akal (teramati s/d $110jt)
#   dti > 100         : cicilan utang > 100% pendapatan tak berkelanjutan; mencakup sentinel dti=999
is_error = (inc[idx_conf] > 5_000_000) | (dti[idx_conf] > 100)
print(f'  data_error terdeteksi: {is_error.sum():,} (income>$5jt atau dti>100)')

# ---------------------------------------------------------------- 6b. Penilaian risiko INDEPENDEN + validasi
# Skor tekanan kredit dari atribut pra-pinjaman (BUKAN outcome): dti, revol_util, all_util, inq.
# Semua di-z-score lalu dijumlah; makin tinggi = makin tertekan. Ambang = persentil 90 skor
# di antara anomali NON-error (data-driven). Lalu DIVALIDASI: bandingkan CO-rate subset ini
# vs populasi. Jika lebih tinggi -> baru boleh disebut sinyal risiko (dibuktikan, bukan diasumsikan).
print('\n=== 6b. Skor risiko independen (atribut pra-pinjaman) + validasi outcome ===')
co_status = ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)',
             'Does not meet the credit policy. Status:Charged Off']
is_co = loan.isin(co_status).values
risk_cols = ['dti', 'revol_util', 'all_util', 'inq_last_6mths']
Zrisk = StandardScaler().fit_transform(df[risk_cols].values)
risk_score_all = Zrisk.sum(axis=1)
rs_conf = risk_score_all[idx_conf]
non_err = ~is_error
thr_risk = np.percentile(rs_conf[non_err], 90) if non_err.any() else np.inf
risk_flag = (~is_error) & (rs_conf >= thr_risk)   # anomali high-stress, bukan error
print(f'  fitur risiko: {risk_cols} | ambang skor (p90 non-error) = {thr_risk:.2f}')
print(f'  anomali high-risk (independen): {risk_flag.sum():,}')

co_all      = is_co.mean() * 100
co_anom     = is_co[idx_conf].mean() * 100
co_highrisk = is_co[idx_conf][risk_flag].mean() * 100 if risk_flag.any() else float('nan')
print(f'  VALIDASI CO-rate -> populasi: {co_all:.1f}% | semua anomali: {co_anom:.1f}% | '
      f'anomali high-risk: {co_highrisk:.1f}%')
verdict = ('TERBUKTI sinyal risiko' if co_highrisk > co_all else
           'TIDAK terbukti lebih berisiko')
print(f'  -> anomali high-risk {verdict} (dibanding populasi {co_all:.1f}%)')
print(f'  Catatan jujur: anomali statistik secara agregat ({co_anom:.1f}%) '
      f'{"<" if co_anom < co_all else ">="} populasi ({co_all:.1f}%) '
      f'-> outlier di sini didominasi profil affluent low-risk, BUKAN sinyal risiko.')

# ---------------------------------------------------------------- 6c. Klasifikasi 3-kategori (rubrik) + tipe outlier
# Prioritas: data_error > risk_signal > rare_case. risk_signal = high-stress independen (6b),
# BUKAN dari loan_status -> tetap non-sirkular meski namanya 'risk'.
print('\n=== 6c. Klasifikasi final (data_error / risk_signal / rare_case) ===')
cls = np.where(is_error, 'data_error',
      np.where(risk_flag, 'risk_signal', 'rare_case'))
classes = pd.Series(cls).value_counts()
for k in ['data_error', 'risk_signal', 'rare_case']:
    print(f'    {k:12s}: {classes.get(k, 0):,}')

# Tipe outlier (untuk Section 2 report): point / contextual / collective.
#   collective : bagian region kepadatan-rendah yang ditemukan DBSCAN (noise) -> ke sample
#   point      : ekstrem pada >=1 dimensi tunggal (ter-flag IQR/Z-score univariat)
#   contextual : anomali hanya pada kombinasi multivariat (Maha/IsoForest), normal per-dimensi
is_noise_full = np.zeros(len(df), dtype=bool)
is_noise_full[noise_idx] = True
uni_flag = iqr_flag | z_flag
otype = np.where(is_noise_full[idx_conf], 'collective',
        np.where(uni_flag[idx_conf], 'point', 'contextual'))
print('  tipe outlier:', {k: int((otype == k).sum()) for k in ['point', 'contextual', 'collective']})

# ---------------------------------------------------------------- 7. Simpan
print('\n=== 7. Simpan hasil ===')
out = df.iloc[idx_conf].copy()
out['loan_status']   = loan.iloc[idx_conf].values
out['n_methods']     = n_methods.values[idx_conf]
out['flag_IQR']      = iqr_flag[idx_conf]
out['flag_Zscore']   = z_flag[idx_conf]
out['flag_Maha']     = maha_flag[idx_conf]
out['flag_IsoForest']= if_flag[idx_conf]
out['maha_dist2']    = md2[idx_conf].round(1)
out['risk_score']    = rs_conf.round(2)
out['classification']= cls           # 3 kategori rubrik (non-sirkular)
out['outlier_type']  = otype         # point / contextual / collective (Section 2 report)
out.insert(0, 'orig_index', idx_conf)
out.sort_values('maha_dist2', ascending=False).to_csv(_p('phase4_anomalies.csv'), index=False)
print(f'  tersimpan: phase4_anomalies.csv ({len(out):,} anomali terkonfirmasi)')
print('\nPhase 4 selesai.')
