# Lending Club — Data Mining (KDD) · Kelompok 2

Penerapan proses Knowledge Discovery in Databases pada dataset Lending Club (2,26 juta pinjaman, 151 fitur mentah): preprocessing, clustering, association rule mining, anomaly detection, dan knowledge presentation.

## Struktur folder

```
.
├── phase1/    Data Understanding & Preprocessing
│   ├── data-mining-phase1.ipynb   EDA + justifikasi keputusan preprocessing
│   └── phase1_pipeline/              Pipeline preprocessing modular (.py)
├── phase2/    Clustering
│   └── data-mining-phase2.ipynb   K-Means, Hierarchical, DBSCAN + profiling
├── phase3/    Association Rule Mining
│   └── data-mining-phase3.ipynb   Diskretisasi domain, Apriori, interpretasi rule
├── phase4/    Anomaly & Outlier Detection
│   └── data-mining-phase4.ipynb   IQR, Z-score, Mahalanobis, Isolation Forest
├── phase5/    Visualization & Knowledge Presentation
│   ├── Knowledge_Discovery_Report.docx / .pdf
│   ├── app.py, data_layer.py, assets/   Dashboard interaktif (Plotly Dash)
│   ├── csv/                             Subset data untuk dashboard (real, bukan sampel ilustratif)
│   └── requirements.txt
├── data/
│   ├── raw/         Data mentah (accepted_2007_to_2018Q4.csv — unduh dari Kaggle)
│   └── processed/   Output antar-fase; hasil mining utama ikut di-commit
└── docs/question/   Dokumen penugasan
```

## Cara menjalankan

```
pip install -r requirements.txt
```

Urutan: Phase 1 dulu (menghasilkan `data/processed/`), lalu fase lain bebas urutan.

- **Pipeline Phase 1**: `python -m phase1_pipeline` (dari dalam `phase1/`). Path ditambatkan ke lokasi file, jadi bisa dijalankan dari direktori mana pun.
- **Notebook**: tiap notebook memuat sel bootstrap yang otomatis berpindah ke root proyek, jalankan sel berurutan (Run All).
- **Phase 5**: `python phase5/app.py` menjalankan dashboard di `http://127.0.0.1:8050`. `data_layer.py` membaca CSV asli di `phase5/csv/` (hasil mining Phase 2-4); tanpa folder itu, dashboard otomatis jatuh ke sampel ilustratif berlabel jelas.

Data mentah tidak di-commit (unduh dari kaggle.com/datasets/wordsforthewise/lending-club, letakkan di `data/raw/`). Hasil mining utama (`phase2_cluster_profiles.csv`, `phase3_association_rules.csv`, `phase4_anomalies.csv`) ikut di-commit di `data/processed/` maupun `phase5/csv/` agar dapat diperiksa tanpa menjalankan ulang pipeline.

Catatan versi: `umap-learn` bergantung pada `numba` yang mensyaratkan NumPy ≤2.4, karena itu `requirements.txt` mengunci `numpy<2.5`. Pasang dependensi lewat `pip install -r requirements.txt` sebelum menjalankan Phase 1 agar sel UMAP dapat dihitung.

## Reproduktibilitas

Seluruh langkah memakai seed tetap (42), sehingga menjalankan ulang di mesin yang sama dengan versi library yang sama menghasilkan angka identik. Yang perlu diketahui saat dijalankan di environment berbeda:

| Bergantung versi library | Deterministik lintas environment |
|---|---|
| K-Means (MiniBatch), UMAP/densMAP, Isolation Forest, Mutual Information | Cleaning, transformasi, scaling, PCA, korelasi, binning, Apriori, IQR, Z-score, Mahalanobis |

Perbedaan versi umumnya menggeser angka pada desimal terakhir (mis. proporsi cluster 62,1 vs 62,2 persen) dan tidak mengubah struktur temuan.

Dua file dependensi dengan peran berbeda:

- `requirements.txt` — 12 dependensi langsung dengan rentang versi. Dipakai untuk pemasangan biasa; toleran terhadap versi baru.
- `requirements-lock.txt` — snapshot `pip freeze` dari environment yang menghasilkan angka pada laporan: 135 paket (termasuk dependensi transitif seperti `numba` dan `llvmlite`) terkunci ke versi persis, plus catatan versi interpreter. Pakai file ini bila ingin angka yang identik: `pip install -r requirements-lock.txt`.

Untuk menjalankan dari nol, satu-satunya berkas yang perlu diunduh manual adalah dataset mentah dari Kaggle ke `data/raw/` (tidak di-commit karena ukurannya). Sisanya diregenerasi oleh Phase 1. Hasil mining utama (`phase2_cluster_profiles.csv`, `phase3_association_rules.csv`, `phase4_anomalies.csv`) dan data dashboard (`phase5/csv/`) ikut di-commit, sehingga laporan dapat diperiksa dan dashboard dapat dijalankan tanpa mengunduh data mentah maupun menjalankan ulang pipeline.
