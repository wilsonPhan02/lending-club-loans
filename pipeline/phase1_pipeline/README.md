# Phase 1 Preprocessing Pipeline — Lending Club

Pipeline preprocessing modular (`.py`) untuk Phase 1 proyek Data Mining, sesuai ketentuan brief: preprocessing dibangun sebagai skrip Python, sedangkan EDA & justifikasi keputusan didokumentasikan di notebook (`data-mining-phase1-v2.ipynb`).

Logika di sini **identik** dengan notebook yang telah divalidasi — hanya direfaktor menjadi modul yang dapat dipanggil ulang, reproducible, dan bebas duplikasi.

## Struktur

| File | Tahap | Isi |
|---|---|---|
| `config.py` | — | Seluruh parameter, ambang, dan daftar kolom domain (satu sumber kebenaran) |
| `data_loader.py` | 00 | Muat data, downcast dtype, konversi kolom persen |
| `cleaning.py` | 01 | Fix inkonsistensi, drop identifier/leakage, tangani nulls (structural/MAR/MCAR), dedup |
| `transform.py` | 02 | Encoding (ordinal+biner, nominal di-drop), binning, winsorization, log kondisional, quasi-constant, scaling |
| `feature_selection.py` | 03 | Korelasi (\|r\|>0.85, tie-break prioritas) + Mutual Information, subset hybrid anti-redundan |
| `dim_reduction.py` | 04 | PCA (≥90% varians) + UMAP 2D (sample) |
| `run_pipeline.py` | — | Orkestrator: load→clean→transform→select→reduce→simpan |

## Cara Pakai

Jalankan dari folder root repo (yang memuat `accepted_2007_to_2018Q4.csv`):

```bash
# sebagai skrip (disarankan)
python -m phase1_pipeline
```

```python
# dari notebook / Python
from phase1_pipeline.run_pipeline import run
art = run()                     # data penuh (config.USE_FULL=True), simpan semua output
art = run(nrows=50_000, save=False)   # uji cepat tanpa menyimpan
```

`run()` mengembalikan dict artefak: `df_final`, `df_final_scaled`, `df_pca`, `df_umap`,
`arm_df`, `mi_df`, `pca`, `final_features`, `loan_status_ref`.

## Output (disimpan ke folder kerja)

| File | Untuk |
|---|---|
| `cleaned_lending_club.csv` | Subset final (unscaled) — profiling cluster |
| `cleaned_lending_club_no_winsorization.csv` | Nilai asli — **Anomaly Detection Phase 4** |
| `scaled_lending_club.csv` | Subset final (scaled) |
| `lending_club_pca.csv` | Input K-Means / Hierarchical (Phase 2) |
| `lending_club_umap.csv` | Input DBSCAN (Phase 2) |
| `lending_club_apriori_binned.csv` | Input Apriori (Phase 3) |
| `loan_status_reference.csv` | Label referensi evaluasi |

## Justifikasi parameter

Setiap ambang di `config.py` terjustifikasi di notebook (bagian *Ringkasan Phase 1 → Tabel Justifikasi Parameter*): IQR 1.5×, missing 40%, winsor 1/99 + guard, log kondisional, quasi-constant 99%, korelasi 0.85, guard redundansi 0.70, PCA 90%, MI sample 50k, dsb.

## Dependensi

`pandas`, `numpy`, `scikit-learn`, dan `umap-learn` (untuk UMAP; jika belum terpasang,
UMAP dilewati otomatis dan output lain tetap terbentuk).
