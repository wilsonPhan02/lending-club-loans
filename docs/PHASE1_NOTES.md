# Phase 1 — Catatan & Informasi Penting (dari notebook lama)

Dokumen ini merangkum seluruh keputusan preprocessing dari `data-mining-phase1.ipynb` versi lama, sebagai referensi saat membangun ulang Phase 1 dari awal. Ditandai juga poin yang **bertentangan dengan ketentuan brief** dan perlu diperbaiki di versi baru.

---

## 1. Mining Angle (acuan seluruh keputusan)

- **Clustering**: kelompokkan peminjam berdasarkan profil risiko memakai `grade`, `income`, `debt ratio`.
- **Rule target**: peminjam dengan `purpose = small business` + `emp_length ≥ 10 tahun` cenderung dapat `Grade A` + bunga rendah.

## 2. Load Data

- Notebook lama membaca **sample 890.000 baris** dari `accepted_2007_to_2018Q4.csv` (total 2.260.701) via `skiprows` acak, `seed=42`, untuk mencegah out-of-memory.
- File asli: 151 kolom mentah.

## 3. EDA yang sudah dilakukan (Step 1)

- Ringkasan tipe data (`dtypes.value_counts`), `describe().T`.
- **Missing values**: kategori >50% (drop), 20–50% (evaluate), <20% (impute); divisualisasikan bar chart top-35.
- **Distribusi numerik kunci**: `loan_amnt, annual_inc, int_rate, dti, revol_bal, installment, fico_range_low, open_acc, total_acc` — histogram + KDE + skewness (flag |skew|>1.5).
- **Outlier**: boxplot + hitung outlier metode IQR (1.5×) per fitur.
- **Distribusi kategorik**: `grade, purpose, home_ownership, term, verification_status, emp_length, loan_status, application_type`.
- **Korelasi awal**: heatmap 14 fitur numerik; deteksi pasangan |r|>0.80.

## 4. Data Cleaning (Step 2)

- Drop kolom >50% missing.
- Drop kolom irrelevant: `id, url, emp_title, title, zip_code, issue_d, last_pymnt_d, next_pymnt_d, last_credit_pull_d, earliest_cr_line, disbursement_method, pymnt_plan, policy_code`.
- Remove duplicate rows.
- Fix inkonsistensi: `term` " 36 months"→36 int; `int_rate`/`revol_util` strip `%`→float; `emp_length`→ordinal numerik.
- Impute numerik dengan **median**; sisa baris ber-NaN di-drop.
- **Winsorization** (cap persentil 1 & 99) untuk: `annual_inc, revol_bal, dti, tot_coll_amt, tot_cur_bal, avg_cur_bal, bc_open_to_buy, delinq_amnt`. (Disimpan juga versi non-winsor `df_no_winsor` untuk Phase 2 & 4.)
- **Drop kolom post-loan (leakage)**: `funded_amnt, funded_amnt_inv, out_prncp, out_prncp_inv, total_pymnt, total_pymnt_inv, total_rec_prncp, total_rec_int, total_rec_late_fee, recoveries, collection_recovery_fee, last_pymnt_amnt, last_fico_range_high, last_fico_range_low`.
- `loan_status` disimpan terpisah sebagai **reference label** (bukan fitur).

## 5. Data Transformation (Step 3)

- **Log1p** untuk fitur |skew|>1 dan non-negatif (kecuali FICO). Urutan dosen: **log transform → standard scaler**.
- **Encoding**:
  - `grade` → ordinal (A=1 … G=7)
  - `sub_grade` → ordinal (A1=1 … G5=35)
  - `application_type` → binary (Individual=0, Joint=1)
  - `initial_list_status` → binary (f=0, w=1)
  - `home_ownership` → **ONE-HOT** ⚠️ (lihat konflik di bawah)
- **Binning** (`fico_bin, dti_bin, income_bin`, qcut) — hanya untuk reporting/deskriptif, bukan untuk clustering.
- **Variance threshold** (0.01) buang fitur near-constant.
- **StandardScaler** ke seluruh fitur numerik.

## 6. Feature Selection (Step 4)

- Buang multikolinearitas: pasangan |r|>0.85, drop salah satu.
- **Mutual Information** dengan `grade` sebagai proxy target (sample 50K, `mutual_info_classif`).
- Final subset **9 fitur** selaras mining angle:
  `loan_amnt, int_rate, grade, annual_inc, dti, fico_range_low, revol_util, emp_length, purpose_small_business`.

## 7. Output (Step 5)

- `cleaned_lending_club.csv` — cleaned, log-transformed, UNSCALED
- `cleaned_lending_club_no_winsorization.csv` — versi tanpa winsorisasi
- `scaled_lending_club.csv` — scaled (untuk PCA→K-Means/Hierarchical)
- `loan_status_reference.csv` — label referensi evaluasi
- (PCA & UMAP dilakukan di pipeline `.py`, bukan di notebook lama ini.)

---

## 8. ⚠️ Konflik dengan ketentuan brief (WAJIB diperbaiki di versi baru)

1. **One-hot encoding `home_ownership`** — brief §2 poin 4 melarang one-hot untuk fitur nominal pada konteks clustering (mendistorsi jarak Euclidean). Nominal murni sebaiknya **di-drop**, bukan di-one-hot. `purpose_small_business` juga hasil one-hot dari `purpose`. → Perlu ditinjau: drop nominal, atau justifikasi khusus bila dipertahankan.
2. **Struktur file** — brief §2 poin 5 menetapkan Phase 1 harus berupa **pipeline `.py`** (modul bertahap), sedangkan EDA + justifikasi di notebook. Notebook lama mengerjakan SEMUA (cleaning, transform, feature selection) inline. → Perlu dipisah sesuai ketentuan.
3. **Winsorization** — brief §7 mereferensikan output `no_winsorization`. Perlu keputusan tegas: pakai winsor atau tidak (dan konsisten dengan Phase 2 & 4).
4. **Urutan wajib** transform → scaling → PCA sudah benar; pastikan PCA (untuk K-Means/Hierarchical) dan UMAP (untuk DBSCAN) dihasilkan dan tervalidasi (di pipeline lama `lending_club_umap.csv` sempat kosong).
