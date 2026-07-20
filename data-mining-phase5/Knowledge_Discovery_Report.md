# Knowledge Discovery Report
## Lending Club Loan Dataset — Penemuan & Implikasi Bisnis

**Proyek:** Data Mining — Knowledge Discovery in Banking Datasets  
**Dataset:** Lending Club Accepted Loans 2007–2018 Q4  
**Pipeline:** KDD (Knowledge Discovery in Databases)  
**Frekuensi Data:** 889,991 pinjaman × 8 fitur final  
**Tanggal Laporan:** Juli 2026  

---

## 1. Ringkasan Eksekutif

> **Pertanyaan Sentral:** *"Apa yang kita temukan dari 889,991 pinjaman Lending Club yang TIDAK terlihat jelas dari data mentah?"*

Tabulasi mentah hanya menunjukkan angka per-individu — saldo pinjaman, skor kredit, pendapatan — tanpa mengungkap pola tersembunyi di baliknya. Pipeline KDD ini menemukan **tiga hal yang tidak terlihat dari agregasi sederhana:**

1. **Populasi peminjam secara alami terbelah menjadi dua arketipe risiko yang kontras** (Subprime vs Prime) meski data finansialnya kontinu/gradual — dikonfirmasi silang oleh K-Means *dan* DBSCAN yang tampak bertentangan namun justru saling melengkapi.

2. **Kombinasi sinyal risiko jauh lebih prediktif daripada sinyal tunggal** — `FICO Very Good + DTI Healthy → Grade A` mencapai lift 3.24×, sesuatu yang mustahil terlihat dari tabulasi silang biasa karena melibatkan tiga variabel sekaligus.

3. **"Anomali" statistik mayoritas BUKAN kesalahan atau fraud** — 53.7% dari 61,961 anomali kuat adalah nasabah sah dengan profil ekstrem, bukan sinyal risiko. Temuan ini secara langsung mengubah cara tim risk management memprioritaskan investigasi manual.

---

## 2. Konteks & Data

### Dataset
Lending Club adalah platform peer-to-peer lending terbesar di AS. Dataset yang dipakai mencakup semua pinjaman yang **disetujui** antara 2007–2018 (file `accepted_2007_to_2018Q4.csv`, ~1.7 GB). Dari 890,000 baris dan 151 kolom mentah, pipeline KDD menyarikan **889,991 baris × 8 fitur final** yang paling relevan untuk analisis risiko kredit.

### Mengapa dataset ini relevan untuk perbankan?
- Lending Club mewakili segmen pinjaman konsumtif tanpa agunan (personal loan) — salah satu segmen paling berisiko dalam portofolio perbankan.
- Data mencakup rentang ekonomi 11 tahun termasuk periode pasca-krisis 2008, memberikan variasi kondisi makroekonomi yang cukup.
- Variabel yang tersedia (FICO score, DTI, grade, tujuan pinjaman) adalah variabel standar yang dipakai oleh hampir semua lembaga kredit di Indonesia maupun global.

### 8 Fitur Final yang Dipakai

| Fitur | Deskripsi |
|---|---|
| `loan_amnt` | Jumlah pinjaman (USD) |
| `grade` | Risk grade Lending Club (A=1 risiko rendah … G=7 risiko tinggi) |
| `annual_inc` | Pendapatan tahunan peminjam (USD) |
| `dti` | Debt-to-Income ratio (%) |
| `fico_range_low` | Skor FICO batas bawah (620–845) |
| `revol_util` | Utilisasi kartu kredit revolving (%) |
| `emp_length` | Lama bekerja (0–10 tahun) |
| `purpose_small_business` | Tujuan pinjaman = usaha kecil (biner: 0/1) |

> **Catatan:** `int_rate` tidak dimasukkan karena korelasinya dengan `grade` sangat tinggi (r > 0.95), sehingga memasukkan keduanya akan menyebabkan multikolinearitas dan duplikasi informasi.

---

## 3. Temuan 1: Dua Arketipe Peminjam

### Apa yang ditemukan?

K-Means (K=2, Silhouette Score 0.189) — dipilih sebagai algoritma terbaik setelah membandingkan tiga metode (K-Means, Hierarchical Ward, DBSCAN) — mengidentifikasi dua segmen peminjam yang kontras secara bisnis:

| Metrik | Cluster 0: "High-Risk Subprime" | Cluster 1: "Low-Risk Prime" |
|---|---|---|
| **Proporsi populasi** | 62.2% (62,197 nasabah dari sampel 100k) | 37.8% (37,803 nasabah) |
| **Pendapatan tahunan rata-rata** | $69,730 | $87,308 |
| **DTI rata-rata** | 20.33% | 15.33% |
| **FICO rata-rata** | 681.72 | 726.34 |
| **Utilisasi kartu kredit** | 61.56% | 31.70% |
| **Grade rata-rata** | 3.22 (Grade C) | 1.76 (Grade B) |
| **Default rate** | **17.07%** | **7.21%** |

### Apa yang tidak terlihat dari data mentah?

Melihat distribusi masing-masing variabel secara terpisah, tidak ada "garis pemisah" yang jelas — FICO menyebar kontinu dari 620 ke 845, DTI dari 0% hingga 55%, dan seterusnya. Yang tidak terlihat tanpa clustering adalah bahwa **konfigurasi spesifik dari lima variabel bersama-sama** — bukan satu variabel saja — menciptakan dua kelompok yang punya profil risiko sangat berbeda.

Konfirmasi lebih lanjut datang dari DBSCAN (dijalankan di ruang UMAP): algoritma ini gagal menemukan lebih dari 1 cluster bermakna di 20,000 data sampel, menghasilkan 107 noise point (0.5%). Ini *bukan* kontradiksi dengan K-Means — justru melengkapi: **data memang kontinu** (tidak ada celah kepadatan yang jelas), tapi K-Means "memaksa" pemisahan yang secara bisnis sangat informatif karena mencerminkan perbedaan default rate 2.37× antara kedua segmen.

### Implikasi bisnis

1. **Pricing differentiation:** Cluster 0 (High-Risk Subprime) semestinya menerima suku bunga 1.5–2× lebih tinggi dari Cluster 1 untuk mencerminkan default rate yang 2.37× lebih besar, bukan hanya mengandalkan Grade Lending Club.

2. **Program retensi Cluster 1:** Nasabah Prime (37.8%) adalah portofolio paling berharga — program loyalitas, penawaran kenaikan limit otomatis, dan cross-selling produk investasi perlu diprioritaskan untuk segmen ini.

3. **Intervensi dini Cluster 0:** Nasabah dengan profil mendekati batas Cluster 0 (FICO 670–690, DTI 18–22%) perlu dipantau lebih ketat dengan alert otomatis sebelum masuk ke delinquency.

---

## 4. Temuan 2: Pola Tersembunyi di Balik Persetujuan Kredit

### Apa yang ditemukan?

Apriori algorithm (min_support=0.04, min_confidence=0.50, min_lift=1.15) menghasilkan 566 frequent itemsets → 678 rules awal → **108 rules valid**. Dari 108 rules ini, 15 rules terbaik (berdasarkan lift tertinggi) dikurasi menjadi **4 tema bisnis actionable:**

### 4 Rule Paling Penting (dari 108 total)

**Rule #1 — Auto-Approval Signal (Lift 3.24×)**  
`{FICO Very Good (740–799), DTI Healthy (<20%)} → {Grade A}`  
*Support: 4.0% | Confidence: 62.5%*

Peminjam dengan FICO 740+ dan DTI di bawah 20% punya kemungkinan **3.24 kali lebih tinggi** mendapat Grade A dibanding peminjam acak. Ini bukan sekedar "FICO tinggi → Grade A" (yang intuitif) — kekuatan kejutannya ada pada *kombinasi* dua sinyal yang bersama-sama jauh lebih prediktif.

**Rule #2 — Utilisasi Rendah sebagai Tanda Kesehatan (Lift 3.13×)**  
`{Grade A, FICO Very Good (740–799)} → {Revol_Util Excellent (<30%)}`  
*Support: 4.0% | Confidence: 72.8%*

Nasabah Grade A dengan FICO sangat baik hampir pasti memiliki utilisasi kartu kredit yang sehat (<30%). Ini menunjukkan bahwa pengelolaan kartu kredit yang disiplin adalah *gejala* dari profil keuangan sehat secara keseluruhan, bukan hanya faktor independen.

**Rule #3 — Sinyal Risiko Ganda (Hampir Pasti)**  
`{DTI High Risk (>43%)} → {Revol_Util Maxed Out (>90%)}`

Peminjam dengan DTI ekstrem (>43%, melampaui batas aman regulasi) hampir pasti juga sudah memaksimalkan batas kartu kredit mereka. Ini menunjukkan dua "sinyal risiko independen" yang ternyata sangat berkorelasi di data riil — kehadiran satu sinyal seharusnya langsung memicu pengecekan sinyal kedua.

**Rule #4 — Profil Mikro-Kredit (Lift 1.72×)**  
`{Annual_Inc Low/Entry (<50k), Grade B} → {Loan_Amnt Small (<10k)}`  
*Support: 6.2% | Confidence: 68.4%*

Nasabah berpendapatan rendah dengan credit score menengah hampir selalu mengajukan pinjaman kecil (<$10k). Segmen ini adalah kandidat kuat untuk produk micro-loan khusus dengan proses persetujuan yang disederhanakan.

### Apa yang tidak terlihat dari data mentah?

Tabulasi silang biasa bisa menunjukkan "FICO tinggi berkorelasi dengan Grade A" — tapi tidak bisa mengukur *seberapa kuat* kombinasi 2–3 variabel bersama-sama meningkatkan probabilitas. Lift 3.24× berarti kombinasi `FICO Very Good + DTI Healthy` menghasilkan Grade A 3× lebih sering dari yang diharapkan secara acak — sesuatu yang tidak mungkin terdeteksi dari scatter plot atau cross-tabulation biasa.

### Rekomendasi operasional

| Rule | Aksi Bisnis |
|---|---|
| `{FICO 740+, DTI <20%} → Grade A` | Implementasikan **Auto-Approval System** untuk memangkas waktu proses |
| `{Grade A, FICO 740+} → Revol_Util <30%` | Tawarkan **kenaikan limit kartu kredit otomatis** tanpa review manual |
| `{DTI >43%} → Revol_Util >90%` | **Tolak otomatis** pengajuan utang baru; arahkan ke program konsolidasi |
| `{Income <50k, Grade B} → Loan <10k` | Kembangkan **paket Micro-Loan** dengan syarat ringkas dan proses cepat |
| `{Income <50k, FICO Poor} → Loan <10k` | Tawarkan **kartu kredit berbasis deposit** untuk membangun riwayat kredit |

---

## 5. Temuan 3: Anomali Bukan Selalu Masalah

### Apa yang ditemukan?

Empat metode deteksi anomali disilangkan menggunakan ensemble scoring (setiap metode yang setuju menambah 1 poin ke Anomaly_Score):

| Metode | Jumlah Anomali | Persentase |
|---|---|---|
| IQR (univariate) | 87,574 | 9.84% |
| Robust Z-Score (MAD) | 53,154 | 5.97% |
| Isolation Forest | 44,500 | 5.00% |
| DBSCAN noise (subset 20k) | 107 | 0.54% |

**Anomali kuat (Anomaly_Score ≥ 2, disepakati ≥2 metode): 61,961 baris (6.96%)**

Namun yang paling penting: **bukan semuanya masalah.** Setelah diklasifikasikan via heuristik bisnis:

| Tipologi | Jumlah | Persentase | Makna |
|---|---|---|---|
| **Rare Legitimate Case** | 33,281 | **53.7%** | Nasabah sehat tapi profil ekstrem (gaji sangat tinggi, pinjaman sangat kecil, FICO sempurna) |
| **Potential Risk Signal** | 19,808 | 32.0% | FICO rendah + pinjaman besar, atau DTI ekstrem — perlu investigasi |
| **Unclassified Outlier** | 8,861 | 14.3% | Anomali struktural yang memerlukan review analis |
| **Data Error** | 11 | 0.02% | Nilai mustahil (income negatif, FICO di luar rentang valid) |

### Apa yang tidak terlihat dari data mentah?

Tanpa analisis tipologi, seorang analis yang melihat daftar "outlier" akan cenderung mengasumsikan semua anomali adalah masalah — dan menolak atau menandai mereka semua untuk investigasi. Temuan ini menunjukkan bahwa **lebih dari separuh anomali justru merupakan nasabah ideal**: pendapatan sangat tinggi yang meminjam sedikit, atau nasabah dengan FICO sempurna yang menjadi outlier justru karena profil keuangan mereka terlalu bagus.

### Implikasi bisnis: Workflow Investigasi Berjenjang

Alih-alih menolak semua outlier secara otomatis, rekomendasikan workflow tiga tingkat:

```
Anomali Terdeteksi (61,961 baris)
│
├── Data Error (11 baris) ──────────── TOLAK OTOMATIS
│   (nilai mustahil secara finansial)
│
├── Potential Risk Signal (19,808) ──── ESKALASI KE TIM RISK MANAGEMENT
│   (FICO rendah + pinjaman besar)      (review manual wajib)
│
├── Unclassified Outlier (8,861) ────── REVIEW ANALIS (prioritas kedua)
│   (anomali struktural)
│
└── Rare Legitimate Case (33,281) ───── WHITELIST (tidak perlu investigasi)
    (profil ekstrem tapi valid)          Pertimbangkan sebagai segmen premium
```

Dengan workflow ini, investigasi manual bisa difokuskan pada ~20,000 baris (32%) yang benar-benar memerlukan perhatian, alih-alih membuang sumber daya untuk 62,000 baris sekaligus.

---

## 6. Jawaban Central Discovery Question

> *"Apa yang kita temukan dari 889,991 pinjaman Lending Club yang TIDAK terlihat jelas dari data mentah?"*

**Tiga penemuan yang hanya muncul melalui pipeline KDD, tidak dari inspeksi data mentah:**

**Pertama**, meski data finansial peminjam bersifat kontinu dan gradual, populasi mereka secara alami terorganisir menjadi dua arketipe risiko yang kontras: "High-Risk Subprime" (62.2%, default rate 17.07%) dan "Low-Risk Prime" (37.8%, default rate 7.21%). Perbedaan default rate 2.37× ini tidak tampak dari distribusi variabel individual — hanya muncul saat lima variabel dievaluasi bersama-sama melalui clustering.

**Kedua**, beberapa kombinasi sinyal risiko memiliki kekuatan prediktif yang jauh melampaui sinyal individual: `FICO Very Good + DTI Healthy → Grade A` (lift 3.24×) berarti kombinasi dua kondisi sederhana ini menghasilkan Grade A tiga kali lebih sering dari yang diharapkan secara acak. Kekuatan ini mustahil terlihat dari korelasi bivariat atau tabulasi silang biasa karena melibatkan interaksi multi-variabel yang non-linear.

**Ketiga**, mayoritas "anomali statistik" (53.7%) bukanlah sinyal bahaya — melainkan nasabah dengan profil keuangan yang terlalu bagus atau terlalu unik sehingga menjadi outlier secara matematis. Hanya 32% anomali yang merupakan sinyal risiko nyata. Temuan ini mengubah paradigma risk management: menolak semua outlier secara otomatis bukan hanya tidak efisien, tetapi berpotensi menolak nasabah paling valuable yang portofolionya ada di dalam 33,281 "Rare Legitimate Case".

---

## 7. Keterbatasan & Catatan Metodologis

### Silhouette Score yang Moderat (0.189)

Silhouette Score K-Means sebesar 0.189 mencerminkan realita data finansial yang bersifat kontinu — **bukan kegagalan model**. Data finansial nasabah tidak pernah membentuk cluster yang bersih dan terpisah tajam seperti data sintetis. Skor 0.189 (vs Hierarchical Ward 0.107) tetap menunjukkan K-Means sebagai pilihan terbaik di antara alternatif yang dicoba. Konfirmasi dari DBSCAN (yang mengonfirmasi data kontinu) justru memperkuat interpretasi ini: dikotomi "Subprime vs Prime" adalah simplifikasi yang berguna secara bisnis dari spektrum risiko yang sebenarnya kontinu.

### DBSCAN Hanya Dijalankan di Subset 20,000 Baris

DBSCAN memiliki kompleksitas komputasi O(n²) yang membuatnya tidak praktis untuk 889,991 baris tanpa optimasi tambahan. 107 noise point yang teridentifikasi berasal dari subset 20,000 baris (bukan populasi penuh) sehingga tidak representatif secara skala. Ini adalah keterbatasan metodologis yang sudah didokumentasikan dan tidak mempengaruhi kesimpulan utama — karena DBSCAN dipakai sebagai alat deteksi anomali tambahan (bukan alat segmentasi utama), dan hasil utamanya (konfirmasi data kontinu) tidak bergantung pada ukuran subset.

### Dataset Hanya Mencakup Pinjaman yang Disetujui

Dataset yang tersedia adalah "accepted loans" — pinjaman yang sudah disetujui oleh Lending Club. Peminjam yang ditolak di tahap awal tidak ada dalam dataset. Ini berarti seleksi bias ada: populasi yang dianalisis sudah melewati filter awal Lending Club, sehingga distribusi FICO, DTI, dan variabel lain di sini tidak merepresentasikan seluruh populasi peminjam potensial.

---

## 8. Rekomendasi Lanjutan

Berikut adalah prioritas penelitian yang direkomendasikan untuk tim berikutnya:

1. **Jalankan DBSCAN di populasi penuh** menggunakan implementasi yang lebih efisien (mis. HDBSCAN dari library `hdbscan`, atau DBSCAN dengan ball-tree indexing) untuk mendapatkan jumlah noise point yang representatif dari 889,991 baris.

2. **Validasi association rules dengan data out-of-time:** Uji apakah 108 rules yang ditemukan dari data 2007–2018 masih berlaku untuk data 2019–2020 (jika tersedia). Rules yang tidak stabil lintas waktu sebaiknya tidak dijadikan dasar keputusan otomatis.

3. **Tambahkan variabel makroekonomi:** GDP growth, unemployment rate, dan federal funds rate sebagai fitur tambahan untuk mendeteksi apakah cluster membership dan rule validity berubah secara signifikan saat kondisi ekonomi berubah.

4. **Bangun model prediktif di atas temuan clustering:** Gunakan `kmeans_cluster` sebagai fitur tambahan untuk melatih Gradient Boosting Classifier yang memprediksi default — hipotesis: menambahkan cluster membership sebagai fitur akan meningkatkan AUC secara signifikan karena menangkap interaksi non-linear antar variabel.

5. **Investigasi 8,861 Unclassified Outlier secara manual:** Sampel 200–300 baris dari kelompok ini dan lakukan profiling kualitatif untuk mengidentifikasi pola yang bisa dikodifikasi menjadi tipologi kelima — mengurangi "abu-abu" di kategori Unclassified.

---

*Dokumen ini adalah bagian dari deliverable Phase 5 (Knowledge Presentation) dalam proyek Data Mining semester genap 2025/2026. Dashboard interaktif yang mengvisualisasikan semua temuan di atas tersedia di `app.py` (jalankan dengan `python app.py` dan buka `http://127.0.0.1:8050`).*
