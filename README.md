# Lending Club — Data Mining (KDD) · Kelompok 2

Penerapan proses Knowledge Discovery in Databases pada dataset Lending Club (2,26 juta pinjaman, 151 fitur mentah).

## Struktur folder

```
.
├── notebooks/     Notebook tiap fase (Phase 1–5), deliverable utama
│   ├── data-mining-phase1-v2.ipynb   Preprocessing
│   ├── data-mining-phase2-v2.ipynb   Clustering
│   ├── data-mining-phase3-v2.ipynb   Association Rule Mining
│   ├── data-mining-phase4-v2.ipynb   Anomaly Detection
│   └── data-mining-phase5-v2.ipynb   Visualization (menghasilkan report/dashboard.html)
├── pipeline/      Kode pipeline modular
│   ├── phase1_pipeline/   Modul preprocessing (config, cleaning, transform, dst.)
│   ├── phase4_anomaly.py  Script deteksi anomali
│   └── regen_umap.py      Regenerasi embedding UMAP
├── data/
│   ├── raw/        Data mentah (accepted_2007_to_2018Q4.csv, datasets/)
│   └── processed/  Output antar-fase (cleaned, scaled, PCA, UMAP, label, rules, anomalies)
├── report/        Knowledge Discovery Report (.docx & .pdf), build_report.py, dashboard.html
└── docs/          PROJECT_BRIEF.md, PHASE1_NOTES.md, question/ (soal)
```

## Cara menjalankan

Notebook memuat sel bootstrap di awal yang otomatis berpindah ke root proyek, sehingga path `data/...` konsisten baik dijalankan dari `notebooks/` (Jupyter) maupun dari root (VS Code). Jalankan sel berurutan (Run All).

Pipeline dan `phase4_anomaly.py` menambatkan path lewat lokasi file-nya sendiri, jadi bisa dijalankan dari direktori mana pun:

```
python -m phase1_pipeline        # dari dalam pipeline/
python pipeline/phase4_anomaly.py
```

Dashboard (`report/dashboard.html`) bersifat mandiri: buka langsung di browser, tanpa server. Dashboard **dihasilkan** oleh `notebooks/data-mining-phase5-v2.ipynb`; report docx dihasilkan `python report/build_report.py`.

Dependensi: `pip install -r requirements.txt`. Data mentah (`data/raw/`) dan output (`data/processed/`) tidak masuk git — unduh dari Kaggle lalu jalankan pipeline Phase 1 untuk meregenerasi.
