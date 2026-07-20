# Implementation Plan — Phase 5: Visualization & Knowledge Presentation
### Data Mining Course Project — Knowledge Discovery in Banking Datasets (Lending Club Loan Dataset)
### Role: Insight Communicator (dengan kontribusi seluruh anggota tim)

---

## 0. Ringkasan Eksekutif

Phase 5 adalah tahap penutup KDD pipeline: mengubah keempat hasil teknis (Preprocessing → Clustering → Association Rules → Anomaly Detection) menjadi **dashboard interaktif, Knowledge Discovery Report, dan presentasi 10 menit** yang bisa dipahami audiens non-teknis. Rubrik menilai *bukan* akurasi model, melainkan **kejelasan insight dan kemampuan menjawab pertanyaan sentral**: *apa yang kita temukan yang tidak terlihat dari data mentah?*

Plan ini disusun setelah mempelajari `Final_Project_Details.pdf` dan **keempat notebook** (`data-mining.ipynb`, `-phase2`, `-phase3`, `-phase4`) secara menyeluruh — termasuk output eksekusi asli (angka statistik nyata, bukan asumsi). Dua deliverable sudah disiapkan bersama dokumen ini:

1. **`phase5_dashboard/`** — starter dashboard Plotly Dash yang *sudah bisa dijalankan sekarang*, berisi 5 tab, dibangun dari angka-angka asli hasil eksekusi notebook Phase 1–4 (silakan lihat `app.py` & `data_layer.py`).
2. **Dokumen ini** — peta jalan lengkap untuk menyelesaikan seluruh requirement Phase 5 sesuai rubrik.

---

## 1. Recap Requirement Phase 5 (dari Final Project Details.pdf)

| Aspek | Ketentuan |
|---|---|
| **Owner** | Insight Communicator (tapi seluruh anggota wajib kontribusi — Phase 5 disebut eksplisit di aturan role) |
| **Tasks** | (1) Bangun dashboard interaktif (Looker Studio / Python Dash / Tableau / Power BI); (2) Visualisasikan cluster map, rule network, outlier plot, distribusi relevan; (3) Tulis Knowledge Discovery Report bahasa bisnis; (4) Jawab central question |
| **Central Question** | *"What did we discover that was not already obvious from the raw data?"* |
| **Deliverable** | Dashboard + presentasi 10 menit + laporan tertulis final |
| **DM Concepts** | Data storytelling, knowledge presentation, dashboard design, business communication |
| **Bobot Nilai** | 20% dari total (setara Preprocessing, Clustering, Rules, Anomaly) |

**Rincian rubrik 5.5 (Final Presentation & Dashboard, gradasi Excellent 85-100):**
- *Dashboard Quality*: interaktif (<100ms), semua visualisasi wajib tampil jelas, bisa diakses audiens non-teknis.
- *Knowledge Report*: menerjemahkan semua temuan ke bahasa bisnis awam, **menjawab central question secara langsung & spesifik**.
- *Presentation*: 10 menit, terstruktur, tim paham penuh hasil sendiri, mampu jawab pertanyaan.

Selain itu, di akhir semester ada **Mining Expo** — sesi lintas kelompok dengan 4 pertanyaan wajib (dibahas di Bagian 8).

---

## 2. Inventarisasi Aset dari Phase 1–4 (Data Audit)

Sebelum membangun apa pun, berikut peta lengkap apa yang **sudah ada** dan **apa yang masih kurang**, hasil pembacaan langsung kode + output eksekusi tiap notebook.

### 2.1 File yang dihasilkan tiap phase

| File | Dihasilkan di | Isi | Ukuran nyata |
|---|---|---|---|
| `cleaned_lending_club.csv` | Phase 1 | Data bersih, unscaled, winsorized, 8 fitur final | 889,991 baris × 8 kolom |
| `cleaned_lending_club_no_winsorization.csv` | Phase 1 | Sama, tapi outlier asli TIDAK dipotong (dipakai Phase 3 & 4) | 889,991 × 8 |
| `scaled_lending_club.csv` | Phase 1 | Versi StandardScaler, input clustering | 889,991 × 8 |
| `loan_status_reference.csv` | Phase 1 | Kolom `loan_status` asli (untuk hitung default rate) | 889,991 baris |
| `kmeans_final.pkl` | Phase 2 | Model K-Means terlatih (K=2) | — |
| `phase2_dbscan_outliers.csv` | Phase 2 | Index 107 noise point DBSCAN (dari subset 20k) | 107 baris |
| `phase2_clustered_sample.csv` | Phase 2 | Sampel 100k baris + label `kmeans_cluster`, `loan_status`, `is_bad_loan` | 100,000 × 12 |
| `phase4_anomaly_report.csv` | Phase 4 | Anomali kuat (Score≥2) + tipologi & penjelasan bisnis | 61,961 baris |
| *(belum ada)* `phase3_association_rules.csv` | Phase 3 | — | **GAP — lihat 2.2** |

### 2.2 Gap yang HARUS ditutup sebelum dashboard final

1. **Phase 3 belum mengekspor rules ke CSV.** Notebook hanya `display()` tabel `interesting_rules` di layar; tidak ada `.to_csv(...)`. Dashboard butuh file ini untuk chart interaktif (bukan angka hardcode). **Tindakan:** tambah 1 sel di akhir `data-mining-phase3.ipynb`:
   ```python
   export_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
   filtered_rules[export_cols].to_csv('phase3_association_rules.csv', index=False)
   ```
2. **DBSCAN Phase 2 hanya berjalan di subset 20.000 baris** (bukan populasi penuh), sehingga 107 noise point-nya tidak representatif secara skala terhadap 889,991 baris. Ini **bukan bug**, sudah didokumentasikan sendiri oleh notebook Phase 2 & 4 sebagai keterbatasan metodologis — dashboard harus menampilkan catatan ini secara eksplisit, jangan menyembunyikannya (lihat Tab Segmentasi).
3. **Row-level data untuk scatter plot** (per-individu, bukan agregat) belum tersedia sebagai file terpisah di luar `phase2_clustered_sample.csv` dan `phase4_anomaly_report.csv` — begitu dua file itu diletakkan di folder dashboard, scatter plot otomatis pakai data asli (lihat `data_layer.py`).
4. **Grading rubric mensyaratkan dashboard "interactive (<100ms)"** — implikasi teknis: hindari query/agregasi berat di dalam callback; pre-agregasi dilakukan di notebook atau di `data_layer.py`, bukan dihitung ulang tiap klik.

### 2.3 Ringkasan Angka Kunci per Phase (bahan baku dashboard & laporan)

**Phase 1** — 890,000 baris mentah × 151 kolom → 889,991 baris × 8 fitur final (`loan_amnt`, `grade`, `annual_inc`, `dti`, `fico_range_low`, `revol_util`, `emp_length`, `purpose_small_business`). `int_rate` dibuang karena korelasi >0.95 dengan `grade`.

**Phase 2** — K-Means K=2 (Silhouette 0.1889) adalah model terbaik: Cluster 0 "High-Risk Subprime" (62.2%, default rate 17.07%) vs Cluster 1 "Low-Risk Prime" (37.8%, default rate 7.21%). DBSCAN pada ruang UMAP tidak menemukan struktur densitas terpisah (1 cluster + 107 noise) — kesimpulan bisnis: risiko kredit bersifat **spektrum kontinu**, bukan kelompok terpisah tajam.

**Phase 3** — Apriori (min_support 0.04, min_confidence 0.5, min_lift 1.15) → 566 frequent itemsets → 678 rules → **108 rules valid**. Rule terkuat: `{FICO Very Good, DTI Healthy} → {Grade A}` (lift 3.24x). 15 rules dikurasi ke 4 tema bisnis (Premium / Mikro-kredit / Tulang punggung / Peringatan risiko).

**Phase 4** — 4 metode disilangkan: IQR (87,574 baris, 9.84%), Robust Z-Score (53,154, 5.97%), Isolation Forest (44,500, 5.00%), DBSCAN noise (107 dari subset). Konsensus ≥2 metode = **61,961 anomali kuat (6.96%)**, diklasifikasi: Rare Legitimate Case 53.7%, Potential Risk Signal 32.0%, Unclassified Outlier 14.3%, Data Error 0.02%.

> Semua angka di atas sudah dipakai langsung sebagai konten dashboard (bukan estimasi) — lihat `data_layer.py`.

---

## 3. Central Discovery Question — Jawaban Terintegrasi

> *"What did we discover that was not already obvious from the raw data?"*

Draf jawaban (dipakai sebagai *headline* dashboard & penutup Knowledge Discovery Report):

> Tabulasi mentah hanya menunjukkan angka per-individu. Pipeline KDD ini mengungkap **tiga hal yang tidak terlihat dari agregasi sederhana**:
> 1. **Populasi peminjam secara alami terbelah jadi dua arketipe risiko yang kontras** (Subprime vs Prime) meski data finansialnya sendiri kontinu/gradual — dikonfirmasi silang oleh K-Means (memaksa 2 kelompok tegas) *dan* DBSCAN (tidak menemukan celah densitas, one giant continuum). Kedua hasil yang tampak kontradiktif ini justru saling melengkapi.
> 2. **Kombinasi sinyal risiko jauh lebih prediktif daripada sinyal tunggal** — `FICO + DTI → Grade A` (lift 3.24x) jauh mengalahkan prediktor tunggal mana pun, sesuatu yang mustahil terlihat dari tabulasi silang biasa karena melibatkan 3+ variabel sekaligus.
> 3. **"Anomali" statistik mayoritas BUKAN kesalahan atau fraud** — 53.7% dari 61,961 anomali kuat adalah nasabah sah yang profilnya ekstrem (rich-but-borrows-little, dsb), bukan sinyal risiko. Temuan ini secara langsung mengubah cara tim risk management semestinya memprioritaskan investigasi manual: jangan menolak semua outlier secara otomatis.

Ketiga poin ini didesain agar **setiap tab dashboard secara eksplisit mendukung salah satu dari 3 klaim** di atas — sehingga central question benar-benar terjawab dengan bukti visual, bukan klaim kosong (kriteria "Excellent" di rubrik 5.5).

---

## 4. Dashboard Information Architecture

**Tool:** Plotly Dash (sesuai instruksi & tech stack resmi di PDF: *"Plotly Dash / Bokeh"*). Dipilih dibanding Looker Studio/Tableau/Power BI karena seluruh pipeline sudah di Python, dan Dash memungkinkan reuse langsung logika `pandas` dari notebook Phase 1–4 tanpa re-platforming data.

Struktur: **1 halaman, 5 tab** (`dcc.Tabs`), header tetap berisi *Central Question* + 5 KPI card, supaya audiens non-teknis langsung menangkap inti cerita dalam 5 detik pertama (sesuai kriteria "accessible to non-technical audience").

| Tab | Tujuan | Chart & Komponen | Sumber Data |
|---|---|---|---|
| **1. Overview** | Peta perjalanan KDD end-to-end | Funnel chart (jumlah baris tiap tahap), 4 kartu ringkasan naratif per phase | `KDD_FUNNEL`, ringkasan tiap phase |
| **2. Segmentasi** | Jawab klaim #1 (dikotomi vs kontinu) | Bar perbandingan algoritma (Silhouette), donut ukuran cluster, grouped bar profil cluster, **scatter Income vs DTI** (filter dropdown per cluster), catatan metodologi DBSCAN | `phase2_clustered_sample.csv` |
| **3. Association Rules** | Jawab klaim #2 (kombinasi sinyal) | **Bubble chart** Support×Confidence (ukuran/warna=Lift), bar jumlah rule per tema, tabel top-15 rule (sortable), filter tema (dropdown) dengan insight bisnis per rule | `phase3_association_rules.csv` (setelah gap ditutup) |
| **4. Anomaly Detection** | Jawab klaim #3 (anomali ≠ risiko) | Bar perbandingan 4 metode deteksi, funnel confidence level, **donut tipologi anomali**, scatter Income vs Loan Amount (filter tipologi) | `phase4_anomaly_report.csv` |
| **5. Mining Expo & Kesimpulan** | Persiapan sesi lintas kelompok | 4 jawaban pertanyaan expo (naratif, lihat Bagian 8) + jawaban central question versi lengkap | Sintesis semua phase |

**Interaktivitas** (memenuhi kriteria "interactive"): dropdown filter cluster (Tab 2), dropdown filter tema rule (Tab 3), dropdown filter tipologi anomali (Tab 4), tabel sortable (Tab 3). Semua callback beroperasi di data agregat kecil (<100 baris) atau sampel yang dibatasi (≤4,000 titik) — dirancang khusus supaya tetap <100ms sesuai rubrik meski nanti dihubungkan ke 889,991 baris data asli.

**Status implementasi:** starter dashboard di atas **sudah dibangun, di-syntax-check, dan di-smoke-test** (server berjalan, HTTP 200, seluruh callback diverifikasi tidak error) — lihat folder `phase5_dashboard/`. Yang tersisa bagi tim: (a) tutup gap Phase 3 CSV, (b) tempel 5 file CSV asli, (c) sesuaikan copy/insight jika angka berubah setelah re-run notebook di mesin tim.

---

## 5. Rencana Teknis Implementasi

### 5.1 Struktur proyek
```
phase5_dashboard/
├── app.py            # layout 5-tab + semua callback
├── data_layer.py      # loader data asli + fallback + generator sampel ilustratif
├── requirements.txt
├── assets/style.css   # tema visual (navy/teal/amber, font Inter)
└── README.md          # cara jalan + cara sambung ke data asli
```

### 5.2 Strategi data loading (penting dipahami tim)
`data_layer.py` mengecek `file_available()` untuk tiap CSV. Jika file asli ada di folder → dipakai langsung. Jika tidak → dashboard fallback ke **angka agregat asli** (diambil dari output notebook yang sudah dieksekusi, bukan karangan) supaya dashboard tetap bisa didemokan hari ini, dan untuk scatter plot row-level, ke **sampel ilustratif berlabel jelas** (badge oranye "sampel ilustratif" di UI) yang mengikuti statistik asli. Tim tinggal copy 5 CSV ke folder → dashboard otomatis "naik kelas" ke data 100% asli tanpa ubah kode.

### 5.3 Menjaga target performa <100ms
- Semua tabel ringkasan (cluster profile, rules, anomaly typology) berukuran <20 baris → render instan.
- Dua scatter plot row-level dibatasi `.sample(4000)` per render, walau sumber data 889,991 baris.
- Tidak ada agregasi ulang (`groupby`, `merge` besar) di dalam callback — semua agregasi berat sudah dilakukan sekali di notebook / `data_layer.py` saat load.

### 5.4 Checklist sebelum submit
- [ ] Tambah sel export CSV di notebook Phase 3 (lihat 2.2)
- [ ] Copy 5 CSV asli ke folder `phase5_dashboard/`
- [ ] Jalankan `python app.py`, cek semua 5 tab & filter di browser
- [ ] Screenshot / rekam tiap tab untuk backup di slide presentasi
- [ ] (Opsional) Deploy ke Render/PythonAnywhere jika ingin diakses tanpa laptop presenter

---

## 6. Knowledge Discovery Report — Outline

Laporan tertulis terpisah dari dashboard (rubrik menilai keduanya sebagai item berbeda). Struktur yang disarankan (bahasa bisnis, tanpa jargon algoritma):

1. **Ringkasan Eksekutif** (setengah halaman) — central question + 3 temuan utama (Bagian 3 di atas), ditulis untuk pembaca yang tidak tahu apa itu K-Means.
2. **Konteks & Data** — dataset apa, berapa banyak, kenapa relevan untuk bank (2-3 paragraf, tanpa detail teknis preprocessing).
3. **Temuan 1: Dua Arketipe Peminjam** — profil Subprime vs Prime, angka default rate, rekomendasi bisnis (mis. pricing differentiation, program retensi Cluster 1, program restrukturisasi dini Cluster 0).
4. **Temuan 2: Pola Tersembunyi di Balik Persetujuan Kredit** — 4-5 rule paling actionable (bukan semua 15), tiap rule diterjemahkan jadi rekomendasi operasional konkret (auto-approval, kampanye limit, dsb — sudah tersedia di `BUSINESS_RULES` pada `data_layer.py`).
5. **Temuan 3: Anomali Bukan Selalu Masalah** — angka 61,961, breakdown tipologi, rekomendasi workflow investigasi berjenjang (Data Error → tolak otomatis; Risk Signal → eskalasi manual; Rare Legitimate → whitelist, bukan tolak).
6. **Jawaban Central Question** (eksplisit, 1 paragraf tegas — lihat Bagian 3).
7. **Keterbatasan & Catatan Metodologis** — jujurkan keterbatasan DBSCAN subset 20k, silhouette score yang moderat (0.19) mencerminkan realita data finansial yang kontinu, bukan kegagalan model.
8. **Rekomendasi Lanjutan** — apa yang bisa dikerjakan tim berikutnya (mis. jalankan DBSCAN di populasi penuh, validasi rules dengan data out-of-time).

---

## 7. Presentasi 10 Menit — Outline Waktu

| Menit | Segmen | Pembicara disarankan |
|---|---|---|
| 0:00–1:00 | Hook: central question + 1 statistik paling mengejutkan (lift 3.24x atau 53.7% anomali sehat) | Insight Communicator |
| 1:00–2:30 | Konteks data & alur KDD (pakai funnel chart Tab 1) | Data Engineer |
| 2:30–4:30 | Temuan Segmentasi (Tab 2) — tampilkan live dashboard | Segmentation Specialist |
| 4:30–6:30 | Temuan Association Rules (Tab 3) — 3 rule paling actionable | Pattern Analyst |
| 6:30–8:00 | Temuan Anomaly (Tab 4) — tekankan "anomali ≠ fraud" | Pattern Analyst / Data Engineer |
| 8:00–9:30 | Jawaban central question + rekomendasi bisnis | Insight Communicator |
| 9:30–10:00 | Buka sesi tanya-jawab | Semua |

**Latihan wajib:** setiap anggota harus bisa menjawab "kenapa metode X dipilih" untuk phase yang bukan tanggung jawabnya — rubrik "Presentation" menilai *"team demonstrates full understanding"*, bukan hanya pembagian bagian per orang.

---

## 8. Mining Expo — Draf Jawaban 4 Pertanyaan Wajib

Draf sudah dituliskan penuh di **Tab 5 dashboard** (`tab_expo()`), ringkasannya:

1. **Rule paling mengejutkan?** → `{FICO Very Good, DTI Healthy} → {Grade A}` (lift 3.24x) — mengejutkan bukan dari arahnya, tapi kekuatannya; juga `{DTI High Risk} → {Revol_Util Maxed Out}` yang nyaris pasti, menunjukkan dua sinyal risiko "independen" ternyata sangat berkorelasi.
2. **Metode clustering paling interpretable?** → K-Means (K=2), karena segmen seimbang (62/38) & kontras jelas di semua metrik risiko; DBSCAN justru lebih berguna sebagai *detektor anomali* daripada *alat segmentasi*.
3. **Anomali apa & maknanya?** → 61,961 anomali (~7%), tapi 53.7%-nya sah (bukan risiko) — insight kunci: jangan otomatis menolak semua outlier.
4. **Perbandingan lintas kelompok?** → **Wajib diisi setelah expo berlangsung** — siapkan 3 pertanyaan pembanding: apakah domain lain juga menemukan struktur kontinu/dikotomis; apakah threshold lift/confidence sebanding; apakah proporsi "anomali sah vs anomali risiko" serupa di domain berbeda (fraud/mortgage/AML).

---

## 9. Timeline & Pembagian Tugas Phase 5 (disarankan 1.5–2 minggu)

| Hari | Aktivitas | PIC |
|---|---|---|
| 1 | Tutup gap Phase 3 (export CSV), kumpulkan 5 file CSV asli dari tiap phase | Data Engineer + Pattern Analyst |
| 2–3 | Setup & jalankan starter dashboard dengan data asli, verifikasi tiap tab & angka | Insight Communicator |
| 4–5 | Review bersama tim: apakah insight per tab akurat & tidak menyesatkan (semua anggota wajib review, sesuai aturan "kontribusi di Phase 5") | Semua |
| 6–7 | Tulis Knowledge Discovery Report mengikuti outline Bagian 6 | Insight Communicator (draft) + review semua |
| 8 | Susun slide presentasi 10 menit + screenshot dashboard sebagai backup | Insight Communicator |
| 9 | Gladi resik: latihan timing + latihan Q&A lintas-role (lihat Bagian 7) | Semua |
| 10 | Buffer / perbaikan berdasarkan feedback dosen atau peer review | Semua |

---

## 10. Self-Check terhadap Rubrik 5.5 (Final Presentation & Dashboard)

| Kriteria | Target "Excellent" | Bagaimana plan ini memenuhinya |
|---|---|---|
| Dashboard interaktif <100ms | Semua visualisasi jelas & bisa diakses non-teknis | Data agregat kecil + sampling di callback (Bagian 5.3); layout kartu & bahasa awam di setiap tab |
| Knowledge Report menjawab central question | Langsung & spesifik | Bagian 3 di plan ini dirancang sebagai jawaban 1-paragraf tegas yang dipakai ulang di laporan & dashboard |
| Presentasi terstruktur, tim paham penuh | Bisa jawab pertanyaan dasar | Outline waktu (Bagian 7) + latihan silang-role wajib |

---

## 11. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| Angka berubah setelah notebook di-run ulang di mesin lain (random seed, versi library) | Semua fungsi di notebook sudah pakai `random_state=42` — seharusnya stabil; jika berbeda signifikan, update konstanta di `data_layer.py` (satu tempat, tidak perlu ubah `app.py`) |
| CSV Phase 2/4 berukuran besar → dashboard lambat saat load pertama | `pd.read_csv` sekali saat start, bukan per callback; jika >200MB pertimbangkan `usecols=` untuk hanya load kolom yang dipakai dashboard |
| Presenter gugup / demo dashboard gagal saat presentasi live | Siapkan screenshot tiap tab sebagai slide cadangan (lihat checklist 5.4) |
| Anggota tim selain Insight Communicator kurang paham dashboard | Sesi review bersama di Hari 4-5 (Bagian 9) sifatnya wajib, bukan opsional |

---

## 12. Lampiran — Data Dictionary Fitur Final

| Kolom | Arti | Tipe |
|---|---|---|
| `loan_amnt` | Jumlah pinjaman (USD) | Numerik |
| `grade` | Risk grade Lending Club, A=1 … G=7 | Ordinal |
| `annual_inc` | Pendapatan tahunan (USD, sudah di-log1p di data scaled) | Numerik |
| `dti` | Debt-to-Income ratio (%) | Numerik |
| `fico_range_low` | Skor FICO batas bawah | Numerik |
| `revol_util` | Utilisasi kartu kredit revolving (%) | Numerik |
| `emp_length` | Lama bekerja (tahun, 0–10) | Ordinal |
| `purpose_small_business` | Tujuan pinjaman = usaha kecil (biner) | Biner |

---

**File terkait dokumen ini:**
- `phase5_dashboard/app.py`, `data_layer.py`, `assets/style.css`, `README.md`, `requirements.txt` — dashboard siap-jalan.
