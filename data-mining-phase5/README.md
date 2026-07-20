# Phase 5 Dashboard — Lending Club Knowledge Discovery

Starter dashboard Plotly Dash untuk Phase 5 (Insight Communicator).
Sudah berisi 5 tab, KPI header, dan chart yang dibangun dari **angka asli**
hasil eksekusi notebook Phase 1–4 (lihat `data_layer.py` untuk sumber tiap
angka). Beberapa scatter plot level-baris memakai **sampel ilustratif**
karena file CSV mentah tidak ikut diserahkan — tinggal ganti dengan CSV asli
tanpa mengubah kode (lihat bawah).

## Menjalankan

```bash
pip install -r requirements.txt
python app.py
```

Buka `http://127.0.0.1:8050`.

## Menghubungkan ke data ASLI (wajib sebelum submit final)

Copy file-file berikut (hasil `to_csv` dari notebook Phase 1–4) ke subfolder
**`csv/`** di dalam folder `data-mining-phase5/` ini:

| File | Dihasilkan oleh | Dipakai untuk |
|---|---|---|
| `cleaned_lending_club.csv` | Phase 1 | KPI, referensi umum |
| `cleaned_lending_club_no_winsorization.csv` | Phase 1 | dasar rules & anomaly |
| `scaled_lending_club.csv` | Phase 1 | referensi scaled features |
| `phase2_clustered_sample.csv` | Phase 2 | scatter & profil cluster |
| `phase2_dbscan_outliers.csv` | Phase 2 | catatan noise point DBSCAN |
| `phase4_anomaly_report.csv` | Phase 4 | scatter & tabel anomali |
| `phase3_association_rules.csv` *(baru, lihat di bawah)* | Phase 3 | tabel & bubble chart rules |

`data_layer.py` otomatis mendeteksi file ini (`file_available()`) dan
memakainya jika ada — tidak ada baris kode di `app.py` yang perlu diubah.

> **Catatan:** File-file di atas sudah tersedia di subfolder `csv/` pada
> versi ini. Jika di-run ulang, cukup pastikan file-file tersebut ada.

### Action item: Phase 3 belum mengekspor rules ke CSV

Notebook `data-mining-phase3.ipynb` saat ini hanya menampilkan
`interesting_rules` di layar (via `display()`), belum ada `.to_csv(...)`.
Tambahkan sel baru di akhir notebook Phase 3:

```python
export_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
filtered_rules[export_cols].to_csv('../data-mining-phase5/csv/phase3_association_rules.csv', index=False)
```

> File `phase3_association_rules.csv` sudah tersedia di subfolder `csv/`
> (hasil ekspor dari sesi sebelumnya), jadi Tab 3 sudah otomatis memakai
> data asli dengan badge hijau "data asli".

## Struktur file

```
data-mining-phase5/
├── app.py              # layout + callbacks (5 tab)
├── data_layer.py       # semua sumber data & fallback (baca komentar di sini dulu)
├── requirements.txt
├── assets/
│   └── style.css       # tema warna & tipografi (navy/teal/amber, font Inter)
├── csv/                # folder data CSV dari Phase 1-4
│   ├── cleaned_lending_club.csv
│   ├── cleaned_lending_club_no_winsorization.csv
│   ├── scaled_lending_club.csv
│   ├── phase2_clustered_sample.csv
│   ├── phase2_dbscan_outliers.csv
│   ├── phase3_association_rules.csv
│   └── phase4_anomaly_report.csv
├── README.md
└── Knowledge_Discovery_Report.md   # laporan tertulis (deliverable terpisah dari dashboard)
```

## Catatan performa (target rubrik: interaktif <100ms)

Semua chart dibangun dari data agregat kecil (<100 baris) kecuali dua
scatter plot, yang dibatasi maksimal ~4.000 titik per render
(`df.sample(4000)`) agar callback tetap responsif walau nanti dihubungkan ke
data asli 889,991 baris. Jangan hilangkan baris `.sample(...)` itu saat
mengganti ke data asli.

Untuk `load_rules()`, data asli difilter ke top-15 rules berdasarkan lift
tertinggi sebelum ditampilkan — agar bubble chart tidak terlalu padat dan
tabel "Top-15 Rule" konsisten dengan label di dashboard.

## Deliverable checklist (rubrik 5.5)

- [x] Dashboard interaktif 5 tab + filter dropdown (Tab 2, 3, 4)
- [x] Visualisasi cluster map, rule bubble chart, outlier donut plot
- [x] Semua 5 file CSV asli tersedia di subfolder `csv/`
- [x] Knowledge Discovery Report (`Knowledge_Discovery_Report.md`)
- [ ] Screenshot tiap tab untuk backup slide presentasi
- [ ] Deploy ke Render/PythonAnywhere (opsional, untuk akses tanpa laptop presenter)
