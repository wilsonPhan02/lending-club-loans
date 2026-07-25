# Lending Club Loans — Data Mining Project Brief

Dokumen ini merangkum seluruh instruksi dari materi soal (`question/`), ketentuan tambahan dari Shane, dan status implementasi saat ini di repo. Tujuannya sebagai referensi bersama sebelum melanjutkan pengerjaan tiap phase.

---

## 1. Konteks Proyek

| Item | Detail |
|---|---|
| Mata kuliah | Data Mining |
| Metodologi | KDD (Knowledge Discovery in Databases), 5 fase, seragam untuk semua kelompok |
| Ukuran kelompok | 5 mahasiswa |
| Durasi | 14 minggu |
| Fokus penilaian | Discovery dan interpretasi pola tersembunyi, **bukan** akurasi prediksi |
| Dataset kelompok | Lending Club Loans (Kelompok 2) |

### 1.1 Dataset

| Item | Detail |
|---|---|
| Domain | Penerbitan pinjaman, penilaian risiko (grade), dan karakteristik finansial peminjam |
| Jumlah baris | ±890.000 (repo saat ini memuat dataset penuh `accepted_2007_to_2018Q4.csv`, ~2,26 juta baris sebelum dedup) |
| Jumlah fitur mentah | 151 (kolom asli pada `accepted_2007_to_2018Q4.csv`) |
| Sumber | Kaggle — `kaggle.com/datasets/wordsforthewise/lending-club` |
| Login | Wajib akun Kaggle gratis |
| Arah penambangan (mining angle) | Mengelompokkan peminjam berdasarkan profil risiko (grade, income, debt ratio); menemukan aturan asosiasi seperti "peminjam dengan tujuan usaha kecil dan masa kerja ≥10 tahun cenderung mendapat Grade A dengan bunga rendah" |
| Catatan khusus | 151 fitur menuntut reduksi fitur yang serius di Phase 1. Subset analitis yang bermakna harus dipilih sejak awal, sebelum clustering. |

### 1.2 Peran dan Kepemilikan Fase

| Peran | Tanggung Jawab | Fase Utama |
|---|---|---|
| Data Engineer (2 orang) | Koleksi data, cleaning, integrasi, pipeline preprocessing | Fase 1 |
| Pattern Analyst (1 orang) | Association rule mining, frequent itemset | Fase 3 (bersama Data Engineer di Fase 4) |
| Segmentation Specialist (1 orang) | Clustering dan profiling entitas | Fase 2 |
| Insight Communicator (1 orang) | Visualisasi, storytelling, laporan akhir, presentasi | Fase 5 |

Catatan: seluruh anggota tetap wajib berkontribusi di Fase 1 dan Fase 5 meski ada peran utama masing-masing.

---

## 2. Instruksi Tambahan dari Shane (di luar dokumen soal)

Poin-poin berikut **tidak tercantum** di dokumen soal asli dan menjadi ketentuan tambahan wajib untuk seluruh pengerjaan:

1. **Gaya bahasa laporan**: seluruh narasi (notebook markdown, laporan, penjelasan) harus melalui proses humanizer (mengikuti pedoman `code/humanizer.md`) — bahasa akademis-profesional, faktual, aktif, tanpa klise AI, tanpa basa-basi konversasional, tanpa metafora berlebihan, tanpa paragraf penutup ringkasan kecuali diminta eksplisit.
2. **Cluster profiling**: setiap cluster wajib divisualisasikan secara jelas (misalnya radar chart, bar chart perbandingan rata-rata fitur, atau scatter plot pada ruang PCA/UMAP) disertai penjelasan interpretasi bisnis yang eksplisit — bukan sekadar deskripsi statistik.
3. **Urutan preprocessing sebelum clustering**: transformasi (mis. log-transform untuk skewness) → scaling (standardisasi) → baru PCA. Metode transformasi/scaling harus disesuaikan dengan tipe data masing-masing kolom (numerik kontinu vs lainnya).
4. **Tidak menggunakan one-hot encoding** untuk fitur nominal karena dianggap bad practice pada konteks clustering (mendistorsi jarak Euclidean, menciptakan sparsity semu). Untuk keperluan clustering, kolom kategorikal **nominal** (bukan ordinal) boleh di-drop sepenuhnya alih-alih di-encode. Kolom **ordinal** (mis. `grade`, `sub_grade`, `emp_length`) tetap dapat dipertahankan melalui ordinal/label encoding yang merefleksikan urutan aslinya.
5. **Struktur file Phase 1**: preprocessing harus dibangun sebagai pipeline skrip Python (`.py`), bukan di dalam notebook. EDA dan justifikasi keputusan preprocessing tetap dikerjakan dan didokumentasikan di notebook (`.ipynb`), yang memanggil/menampilkan hasil dari pipeline tersebut.
6. **Struktur file Phase 2–5**: cukup dikerjakan di notebook (`.ipynb`) saja, tidak perlu pipeline skrip terpisah.

---

## 3. Lima Fase Wajib (dari dokumen soal)

### Fase 1 — Data Understanding and Preprocessing
**Owner**: Data Engineer

Tugas:
- Eksplorasi dataset: distribusi, missing values, tipe data, outlier.
- Data cleaning: menangani nilai kosong, memperbaiki inkonsistensi, menghapus duplikat.
- Transformasi data: normalisasi/scaling, encoding, binning variabel kontinu.
- Feature selection menggunakan analisis korelasi dan ukuran entropi (mutual information).

Deliverable: dataset bersih hasil pipeline preprocessing yang ketat + Preprocessing Report (bagian dari laporan akhir).

Tujuan fase: menghasilkan dataset siap-analisis dan mendokumentasikan setiap keputusan preprocessing beserta justifikasinya. Fase ini menjadi fondasi faktual untuk seluruh tahap penambangan berikutnya.

**Implementasi teknis (sesuai ketentuan tambahan Shane)**:
- Ditulis sebagai pipeline `.py` bertahap (`run_pipeline.py` memanggil modul 01–05 secara berurutan).
- EDA dan justifikasi tetap didokumentasikan di `data-mining-phase1.ipynb`.
- Urutan wajib: cleaning → feature selection (korelasi + entropy) → transformasi (log-transform fitur skewed) → scaling (StandardScaler) → PCA (untuk input K-Means/Hierarchical) dan UMAP (untuk input DBSCAN) → discretization (untuk Fase 3 Apriori).
- Kolom nominal (object dtype non-ordinal) di-drop, bukan di-one-hot-encode, sebelum tahap clustering.

### Fase 2 — Segmentation via Clustering
**Owner**: Segmentation Specialist

Tugas:
- K-Means untuk segmentasi entitas berdasarkan atribut perilaku/finansial.
- DBSCAN untuk mengidentifikasi profil outlier dan noise points.
- Hierarchical clustering, membandingkan dendrogram antar metode linkage.
- Menentukan K optimal menggunakan Elbow Method dan Silhouette Score.
- Profiling tiap cluster: jenis entitas yang direpresentasikan dan karakteristik pembedanya.

Deliverable: profil cluster dengan interpretasi bisnis penuh.

Tujuan fase: menemukan pengelompokan alami dalam data dan menerjemahkan tiap kelompok menjadi profil bernama yang bermakna, menggambarkan perilaku atau karakteristik finansial nyata.

**Implementasi**: seluruhnya di notebook (`data-mining-phase2.ipynb`). Wajib menyertakan visualisasi profil cluster (radar/bar/scatter) dan narasi interpretasi bisnis per cluster.

### Fase 3 — Association Rule Mining
**Owner**: Pattern Analyst

Tugas:
- Diskretisasi variabel kontinu menjadi kelompok kategorikal bermakna.
- Apriori untuk menemukan frequent itemset.
- Menghitung Support, Confidence, dan Lift untuk tiap rule.
- Filter rule agar hanya menyisakan temuan yang bermakna, non-trivial, dan high-lift.
- Menemukan dan mendokumentasikan minimal 10 rule dengan interpretasi bisnis yang jelas.

Deliverable: tabel rule terurut dengan komentar bisnis.

Tujuan fase: mengungkap pola ko-okurensi yang mengungkap hubungan tidak jelas antar atribut, temuan yang tidak dapat dilihat lewat tabulasi atau agregasi sederhana.

**Implementasi**: notebook saja.

### Fase 4 — Anomaly and Outlier Detection
**Owner**: Data Engineer dan Pattern Analyst

Tugas:
- Metode statistik (IQR, Z-score) untuk menandai perilaku/nilai anomali.
- Isolation Forest sebagai alat deteksi anomali struktural.
- Cross-reference anomali yang terdeteksi dengan outlier cluster dari Fase 2.
- Investigasi tiap anomali: kesalahan data, kasus sah yang jarang terjadi, atau sinyal risiko potensial.

Deliverable: laporan anomali dengan record yang ditandai, penjelasan, dan interpretasi bisnis.

Tujuan fase: mengidentifikasi record yang menyimpang signifikan dari data lainnya dan menentukan apakah tiap penyimpangan merepresentasikan masalah kualitas data, kasus valid yang tidak biasa, atau sinyal yang perlu dieskalasi.

**Implementasi**: notebook saja.

### Fase 5 — Visualization and Knowledge Presentation
**Owner**: Insight Communicator

Tugas:
- Membangun dashboard interaktif (Google Looker Studio, Python Dash, Tableau Public, atau Power BI).
- Memvisualisasikan cluster map, rule network, outlier plot, dan distribusi relevan.
- Menulis Knowledge Discovery Report yang menerjemahkan seluruh temuan ke bahasa bisnis sederhana.
- Menjawab pertanyaan inti: apa yang ditemukan yang sebelumnya tidak terlihat jelas dari data mentah?

Deliverable: dashboard, presentasi kelompok 10 menit, dan laporan akhir tertulis.

Tujuan fase: mengomunikasikan pengetahuan yang ditemukan kepada audiens non-teknis secara jelas dan meyakinkan. Ukuran keberhasilan adalah apakah temuan bersifat actionable dan non-trivial, bukan apakah model mencapai akurasi tinggi.

**Implementasi**: notebook saja, plus dashboard dan laporan akhir (docx, mengikuti template).

---

## 4. Tech Stack yang Diizinkan

Python (pandas, mlxtend, scikit-learn), R, Jupyter Notebook, Matplotlib/Seaborn/Plotly, Mage/Prefect/Airflow (untuk pipeline), Plotly Dash/Bokeh (untuk dashboard), GitHub untuk kolaborasi dan version control.

---

## 5. Rubrik Penilaian

| Komponen | Bobot | Dinilai Berdasarkan |
|---|---|---|
| Preprocessing | 20% (rubrik detail: 15%) | Kelengkapan, justifikasi pilihan |
| Clustering Analysis | 20% (rubrik detail: 25%) | Validity metrics, interpretasi bisnis |
| Association Rule Mining | 20% (rubrik detail: 25%) | Kualitas rule, non-trivialitas, nilai lift |
| Anomaly Detection | 20% (rubrik detail: 15%) | Kedalaman investigasi, penjelasan |
| Final Presentation & Dashboard | 20% | Kejelasan, kualitas insight, storytelling |

### 5.1 Preprocessing — kriteria "Excellent" (85–100)
- **Data Cleaning**: seluruh missing values, duplikat, dan inkonsistensi ditangani dengan justifikasi jelas untuk tiap keputusan.
- **Data Transformation**: normalisasi, encoding, dan binning diterapkan secara benar dan sesuai untuk dataset.
- **Feature Selection**: dilakukan menggunakan korelasi **dan** entropy measures dengan temuan dijelaskan secara jelas.

### 5.2 Clustering Analysis — kriteria "Excellent"
- **Algorithm Application**: K-Means, DBSCAN, dan Hierarchical clustering seluruhnya diterapkan dengan benar, pilihan parameter tepat.
- **Optimal K Selection**: Elbow Method dan Silhouette Score digunakan keduanya, hasil diinterpretasikan dengan benar dan digunakan untuk menjustifikasi K final.
- **Cluster Profiling**: tiap cluster dideskripsikan dengan profil bernama, atribut pembeda, dan interpretasi bisnis yang bermakna.

### 5.3 Association Rule Mining — kriteria "Excellent"
- **Discretization**: variabel kontinu di-binning ke kategori yang bermakna dan relevan domain, dengan rasionale jelas.
- **Rule Generation**: Apriori diterapkan benar; Support, Confidence, Lift dihitung dan digunakan untuk memfilter rule secara efektif.
- **Rule Interpretation**: minimal 10 rule non-trivial didokumentasikan dengan interpretasi bisnis yang spesifik, akurat, dan actionable.

### 5.4 Anomaly Detection — kriteria "Excellent"
- **Detection Methods**: IQR, Z-score, dan Isolation Forest seluruhnya diterapkan benar, hasil dibandingkan secara sistematis.
- **Cross-referencing**: anomali yang terdeteksi disilangkan dengan outlier cluster Fase 2 dengan reasoning jelas.
- **Business Interpretation**: tiap anomali yang ditandai diklasifikasikan sebagai data error, rare case, atau risk signal dengan bukti pendukung spesifik.

### 5.5 Final Presentation & Dashboard — kriteria "Excellent"
- **Dashboard Quality**: interaktif (<100ms), menampilkan seluruh visualisasi wajib dengan jelas, dapat diakses audiens non-teknis.
- **Knowledge Report**: menerjemahkan seluruh temuan ke bahasa bisnis sederhana, menjawab pertanyaan inti discovery secara langsung dan spesifik.
- **Presentation**: presentasi 10 menit terstruktur baik, temuan dikomunikasikan jelas, tim menunjukkan pemahaman penuh atas hasil.

### 5.6 Catatan Khusus Dataset

1. Grup 3 dan 8 memiliki label fraud/laundering — tidak berlaku untuk Grup 2 (Lending Club).
2. Grup 4 (HMDA) wajib sampling 100.000 baris — tidak berlaku untuk Grup 2.
3. **Berlaku untuk Grup 2**: Lending Club memiliki 151 fitur mentah (dokumen soal menyebut 74; jumlah aktual pada file `accepted_2007_to_2018Q4.csv` adalah 151). Feature selection di Fase 1 bersifat kritis. Subset analitis bermakna wajib direduksi sebelum clustering.

---

## 6. Template Laporan Akhir (Knowledge Discovery Report)

File: `question/Data Mining Report Template.docx`. Batas maksimum: **10 halaman** (di luar Appendix, yang tidak dibatasi).

Struktur wajib (tidak boleh ditambah/dikurangi bagian):

1. **Executive Summary** (maks. 1 halaman) — 2–3 kalimat: temuan utama, apa yang belum diketahui/dilakukan bank saat ini, aksi yang direkomendasikan. Ditulis paling akhir, setelah semua temuan selesai.
2. **Dataset and Methodology** (1–3 halaman) — satu paragraf per fase, metode saja, bukan hasil.
   - Dataset: nama, sumber, jumlah record digunakan, jumlah fitur setelah feature selection, domain, metode sampling jika ada.
   - Phase 1: cleaning, transformasi (metode normalisasi, encoding, binning), hasil feature selection.
   - Phase 2: tiga algoritma (K-Means/K-Medoids, Hierarchical, DBSCAN), linkage method + cophenetic correlation, K final, nilai Elbow & Silhouette, Adjusted Rand Index (K-Means vs Hierarchical).
   - Phase 3: strategi diskretisasi, parameter Apriori (min Support, Confidence), jumlah rule sebelum/sesudah filtering, rentang nilai Lift.
   - Phase 4: tipe outlier yang dicari (point, contextual, collective), metode per tipe, threshold yang digunakan (IQR multiplier, Z-score threshold, Mahalanobis chi-square percentile, Isolation Forest contamination), jumlah kandidat sebelum korroborasi.
3. **Findings** (4–5 halaman) — **tepat 3 temuan** (boleh 1 tambahan), masing-masing dari fase berbeda (clustering, ARM, anomaly detection). Tiap temuan wajib lolos 4 uji translasi: mengandung minimal satu angka dari analisis, tidak obvious dari data mentah tanpa mining, mengimplikasikan aksi bank yang spesifik, dan menghindari bahasa kausal kecuali mekanisme telah dibuktikan. Tiap finding memiliki subbagian: Evidence, Corroboration, Business Implication, Recommended Action.
4. **Limitations** (1–2 halaman) — mencakup minimal 4 poin: Scope of Outlier Detection, Correlation versus Causation, Dataset Representativeness, What Additional Data Would Improve These Findings.
5. **Appendix** (tidak dibatasi) — Appendix A (Full Cluster Profiles), Appendix B (Full Association Rule Table: Antecedent, Consequent, Support, Confidence, Lift, diurutkan Lift menurun), Appendix C (Anomaly Detection Results per record: ID, metode yang menandai, skor, tipe outlier, interpretasi bisnis), Appendix D (Evaluation Metrics Summary — tabel referensi tunggal seluruh metrik kuantitatif).

Metrik wajib di Appendix D:
- Silhouette Score (metode utama, K final)
- Cophenetic Correlation (linkage terpilih)
- Adjusted Rand Index (K-Means vs Hierarchical)
- Jumlah rule dihasilkan
- Jumlah rule dipertahankan setelah filtering
- Nilai Lift tertinggi di retained rules
- Total kandidat anomali sebelum korroborasi
- Anomali terkonfirmasi (dikorroborasi ≥2 metode)
- Precision/Recall vs label (jika berlaku)

---

## 7. Status Implementasi Saat Ini (repo `lending-club-loans`)

Berdasarkan pemeriksaan folder `code/`:

- **Phase 1** (`code/Phase1/`): sudah ada pipeline modular — `01_data_cleaning.py`, `02_feature_selection.py`, `03_feature_engineering.py`, `04_dimensionality_reduction.py`, `05_discretization.py`, dikoordinasikan oleh `run_pipeline.py`. Notebook `data-mining-phase1.ipynb` dan `build_notebook.py` (generator notebook) juga sudah ada, beserta `run_eda_plots.py`, `trace_features.py`, dan `feature_fate_table.md` (dokumentasi nasib tiap fitur — drop/keep beserta alasannya).
  - Cleaning sudah menghapus kolom leakage (variabel pasca-origination pinjaman) dan kolom ID/deskripsi berkardinalitas tinggi.
  - Kolom nominal (`object` dtype, kecuali `loan_status` yang dipakai sementara sebagai surrogate target) **sudah di-drop**, bukan di-one-hot — sejalan dengan ketentuan tambahan Shane.
  - Feature selection sudah menggunakan filter korelasi (>0.85) dan mutual information terhadap `loan_status` sebagai surrogate.
  - Transformasi sudah menerapkan log1p pada fitur skewed (|skew| > 2.0), lalu StandardScaler, baru PCA (5 komponen) untuk input K-Means/Hierarchical dan UMAP (2 komponen) untuk input DBSCAN — urutan ini sudah konsisten dengan instruksi transformasi → scaling → PCA.
  - Output tervalidasi: `cleaned_lending_club_no_winsorization.csv`, `lending_club_pca.csv`, `lending_club_umap.csv` (saat ini kosong — kemungkinan modul `umap-learn` tidak terpasang di environment saat run terakhir, perlu diperiksa ulang), `lending_club_apriori_binned.csv`.
  - **Perlu ditinjau ulang**: apakah kolom ordinal (`grade`, `sub_grade`, `emp_length`, dsb.) ikut terhapus oleh filter `object_cols` di `01_data_cleaning.py`, atau sudah ditangani terpisah dengan ordinal encoding sebelum drop. Berdasarkan kode saat ini, seluruh kolom `object` (termasuk yang berpotensi ordinal) dihapus tanpa pembedaan — ini perlu diperbaiki agar variabel ordinal yang informatif tidak hilang percuma.
- **Phase 2** (`code/Phase2/`): sudah ada `data-mining-phase2.ipynb` dan `build_phase2_notebook.py`. Perlu ditinjau apakah visualisasi profil cluster dan narasi interpretasi bisnis sudah memenuhi ketentuan tambahan (radar/bar chart + penjelasan bisnis eksplisit).
- **Phase 3 & 4**: hanya ada draft awal di `code/data-mining-phase3.ipynb` dan `code/data-mining-phase4.ipynb` di root `code/` (bukan di subfolder terstruktur seperti Phase1/Phase2) — perlu direlokasi/dirapikan mengikuti pola folder yang sama.
- **Phase 5**: belum ada berkas sama sekali (dashboard maupun notebook visualisasi/report generation).

---

## 8. Rencana Kerja Bertahap (Ringkas)

1. **Audit ulang Phase 1**: pastikan pembedaan nominal vs ordinal ditangani eksplisit (ordinal encoding untuk `grade`/`sub_grade`/`emp_length`, drop murni untuk nominal murni seperti `purpose`, `home_ownership`, `addr_state` jika memang dipertahankan sebagai nominal). Perbaiki `lending_club_umap.csv` yang kosong. Selesaikan EDA + justifikasi lengkap di notebook.
2. **Rapikan Phase 2**: lengkapi tiga algoritma clustering, validitas (Elbow, Silhouette, Cophenetic Correlation, Adjusted Rand Index), profiling dengan visualisasi dan interpretasi bisnis per cluster.
3. **Bangun Phase 3**: pindahkan/rapikan notebook Apriori ke struktur folder konsisten, pastikan ≥10 rule dengan interpretasi bisnis dan tabel Support/Confidence/Lift lengkap.
4. **Bangun Phase 4**: terapkan IQR, Z-score, Isolation Forest; cross-reference dengan outlier cluster Fase 2; klasifikasikan tiap anomali.
5. **Bangun Phase 5**: dashboard interaktif, visualisasi cluster map/rule network/outlier plot, serta penyusunan Knowledge Discovery Report mengikuti template docx secara ketat (tepat 3 findings, maksimal +1 tambahan bila benar-benar diperlukan; bahasa humanizer; format Appendix lengkap).
6. **Finalisasi laporan**: isi seluruh section template docx, verifikasi setiap finding lolos 4 uji translasi, isi Appendix D dengan metrik lengkap dari seluruh fase.
