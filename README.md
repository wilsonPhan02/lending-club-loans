# Lending Club — Data Mining (KDD) · Kelompok 2

Penerapan proses Knowledge Discovery in Databases pada dataset Lending Club (2,26 juta pinjaman, 151 fitur mentah): preprocessing, clustering, association rule mining, anomaly detection, dan knowledge presentation.

## Struktur folder

```
.
├── phase1/    Data Understanding & Preprocessing
│   ├── data-mining-phase1-v2.ipynb   EDA + justifikasi keputusan preprocessing
│   └── phase1_pipeline/              Pipeline preprocessing modular (.py)
├── phase2/    Clustering
│   └── data-mining-phase2-v2.ipynb   K-Means, Hierarchical, DBSCAN + profiling
├── phase3/    Association Rule Mining
│   └── data-mining-phase3-v2.ipynb   Diskretisasi domain, Apriori, interpretasi rule
├── phase4/    Anomaly & Outlier Detection
│   └── data-mining-phase4-v2.ipynb   IQR, Z-score, Mahalanobis, Isolation Forest
├── phase5/    Visualization & Knowledge Presentation
│   ├── Knowledge_Discovery_Report.docx / .pdf
│   ├── dashboard.html                Dashboard interaktif (buka langsung di browser)
│   └── build_dashboard.py            Kode pembuat dashboard (dari output fase)
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
- **Phase 5**: `python phase5/build_dashboard.py` meregenerasi dashboard dari `data/processed/`.

Dashboard (`phase5/dashboard.html`) bersifat mandiri: buka langsung di browser, tanpa server.

Data mentah tidak di-commit (unduh dari kaggle.com/datasets/wordsforthewise/lending-club, letakkan di `data/raw/`). Hasil mining utama (`phase2_cluster_profiles.csv`, `phase3_association_rules.csv`, `phase4_anomalies.csv`) ikut di-commit agar dapat diperiksa tanpa menjalankan ulang pipeline.
