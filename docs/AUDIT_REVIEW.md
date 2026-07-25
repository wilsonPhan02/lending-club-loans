# Audit Review — Phase 1–5 (Lending Club, Kelompok 2)

Audit menyeluruh terhadap question docs, PROJECT_BRIEF, pipeline Phase 1, notebook Phase 1–4, report, dan dashboard. Verdict singkat: **kualitas keseluruhan sudah tinggi** — metodologi benar, justifikasi parameter lengkap, hampir tidak ada hardcode tanpa dasar, dan angka antar-artefak konsisten (semua klaim markdown kucek terhadap output eksekusi aktual dan cocok). Namun ada **satu cacat metodologis serius di Phase 3** dan beberapa gap kepatuhan template yang perlu dibereskan sebelum bisa disebut excellent.

---

## A. Temuan KRITIS (perbaiki sebelum submit)

### A1. Phase 3 — "sample 250k" ternyata 100% pinjaman tahun 2015 (bias berat)
Cell load Phase 3 memakai `pd.read_csv(..., nrows=250_000)` — ini mengambil **250 ribu baris PERTAMA**, bukan sample acak. `RANDOM_SEED` dideklarasikan tapi tidak pernah dipakai. Kuverifikasi terhadap `issue_d` di file raw:

| | 2015 | 2016 | 2017 | 2018 |
|---|---|---|---|---|
| Sample Phase 3 (250k pertama) | **100,0%** | 0% | 0% | 0% |
| Data penuh | 18,6% | 19,2% | 19,6% | 21,9% |

Klaim di notebook ("sample memberi estimasi praktis identik dengan data penuh") tidak valid — support/confidence/lift semua rule dihitung dari satu kohort tahun origination saja. Ini persis jenis temuan yang bisa menjatuhkan nilai ARM (bobot rubrik 25%).

**Fix** (murah, data kecil karena hanya 12 + 9 kolom):
```python
nw_full  = pd.read_csv('data/processed/cleaned_lending_club_no_winsorization.csv')
arm_full = pd.read_csv('data/processed/lending_club_apriori_binned.csv')
idx = np.random.RandomState(RANDOM_SEED).choice(len(nw_full), SAMPLE_N, replace=False)
nw, arm = nw_full.iloc[idx].reset_index(drop=True), arm_full.iloc[idx].reset_index(drop=True)
```
Lalu **re-run Phase 3** dan update semua angka turunannya di report + dashboard (925 rule, lift 1,15–5,03, 1.980 itemset, tabel Appendix B, temuan "small_business = 0 rule", dan rule scatter/network di dashboard). Temuan struktural (grade↔int_rate) hampir pasti bertahan, tetapi angka pasti akan bergeser.

### A2. Phase 3 / Appendix D — "jumlah rule sebelum filtering" tidak pernah dihitung
Template report mewajibkan "total number of rules generated before filtering dan jumlah retained". Notebook langsung men-generate rule dengan `min_threshold=1.15` (lift), sehingga angka "sebelum filter" tidak ada; Appendix D menggantinya dengan jumlah frequent itemset (1.980) — itu metrik berbeda. **Fix**: generate dulu tanpa filter (`association_rules(frequent, metric='confidence', min_threshold=0.0)` → catat N), baru terapkan filter lift>1,15 & conf>0,5. Laporkan N sebelum → sesudah di Section 2 dan Appendix D.

### A3. Report — struktur tidak persis mengikuti template
Template eksplisit: *"Do not add sections that are not in this template and do not remove sections that are."*
1. Section 1 saat ini menggabung **Executive Summary + Informasi Dataset**. Dataset seharusnya berada di Section 2 (Dataset and Methodology → subbagian Dataset). Pisahkan.
2. Executive Summary tidak mengikuti format wajib **2–3 kalimat**: (i) pola terpenting yang ditemukan, (ii) apa yang bank belum ketahui/lakukan, (iii) aksi yang direkomendasikan. Paragraf sekarang menjelaskan proses, bukan temuan.
3. **Cover fields template hilang**: Group Name, tabel Student Name + Role (5 anggota), Submission Date, GitHub URL.

### A4. Report — Appendix B label korup
Semua item tertulis `int: rate: IntRate Very High`, `fico: range: low: ...` — underscore berubah jadi `": "` saat generate docx. Perbaiki jadi `int_rate = IntRate Very High` (atau format bersih lain yang terbaca).

### A5. GitHub — repo belum ada, dan .gitignore malah membuang kode
Folder belum di-`git init`, dan `.gitignore` berisi `*.py`, `*.csv`, `*.txt`, `*.pdf` — kalau dipakai apa adanya, **seluruh pipeline .py dan report .pdf tidak akan pernah masuk GitHub**. GitHub adalah bagian tech stack wajib dan template punya field GitHub URL. **Fix**: init repo; ganti .gitignore agar hanya meng-ignore data (`data/raw/`, `data/processed/`), bukan kode; tambahkan `requirements.txt` (pandas, scikit-learn, mlxtend, umap-learn, scipy, matplotlib, seaborn, plotly).

---

## B. Temuan SEDANG (dianjurkan kuat)

### B1. Phase 5 — tidak ada kode yang menghasilkan dashboard.html
`report/dashboard.html` ada dan lengkap (cluster map UMAP, radar, rule network, rule scatter, outlier plot, validasi risiko, distribusi, 6 KPI — semua visual wajib terpenuhi dan angkanya konsisten dengan output fase), tetapi **tidak ada notebook/script Phase 5** yang membuatnya. Brief internal: Phase 2–5 dikerjakan di notebook. Buat `data-mining-phase5-v2.ipynb` yang membaca CSV processed dan menulis dashboard.html — sekaligus menutup gap reproducibility.

### B2. Definisi charged-off tidak konsisten antar-temuan di report
Temuan 1 memakai *Charged Off saja* (14,4% vs 7,8%; implisit populasi ≈11,9%), Temuan 3 memakai definisi luas *Charged Off + Default + Late* (populasi 13,1%). Pembaca teliti akan bertanya kenapa "populasi" berbeda. **Fix**: satu kalimat definisi eksplisit di tiap temuan, atau seragamkan definisinya.

### B3. Report Section 2 — beberapa item template belum disebut
1. Phase 2: template minta **nilai Elbow di K final** — nyatakan eksplisit hasil elbow ("inertia menurun mulus tanpa siku; keputusan bertumpu pada Silhouette lintas 5 sample").
2. Phase 1: template minta keputusan **binning** — binning qcut untuk persiapan ARM tidak disebut; tambahkan satu kalimat. Sebutkan juga angka duplikat/baris dibuang (33) agar "cleaning steps" konkret.
3. Appendix C: template minta interpretasi bisnis **per record** — tambahkan satu kolom interpretasi singkat pada 18 record contoh (CSV lengkap sudah oke sebagai lampiran).

### B4. Dashboard — interaktivitas minimal & pilihan tool
Interaksi baru sebatas hover/zoom Plotly (memenuhi "<100ms" secara harfiah). Satu-dua kontrol (dropdown filter segmen pada cluster map, atau slider min-lift pada rule scatter) akan mengamankan kriteria "interactive" tanpa perlu server. Catatan kecil: soal menyebut Looker Studio / Dash / Tableau / Power BI (tech stack menyebut "Plotly Dash / Bokeh"); static Plotly HTML kemungkinan besar diterima karena self-contained, tapi konfirmasi singkat ke dosen menghilangkan risiko.

### B5. Bahasa laporan
Seluruh dokumen soal & template berbahasa Inggris; report/notebook berbahasa Indonesia. Kalau kelas mengizinkan Indonesian, aman; kalau tidak, ini biaya translate yang besar — pastikan sekarang, jangan di minggu akhir.

---

## C. Temuan MINOR

1. **`collections_12_mths_ex_med` diklasifikasi post-loan leakage** — sebenarnya atribut riwayat kredit yang diketahui saat origination (jumlah collections 12 bulan *sebelum* aplikasi). Dampak kecil (1 kolom); kalau tidak mau re-run, cukup koreksi justifikasinya di notebook/feature-fate.
2. **PROJECT_BRIEF.md §7 stale** — masih menyebut MI terhadap `loan_status`; implementasi final memakai `grade` (dengan justifikasi bagus di notebook). Update brief agar tidak menyesatkan anggota tim lain.
3. **Komentar output Phase 1** menyebut `cleaned_lending_club.csv` untuk "interpretasi profil", padahal Phase 2 (benar) memakai `no_winsorization` untuk profiling — nilai di `cleaned` sudah log-transformed sehingga tidak terbaca bisnis. Rapikan komentar/README.
4. **Urutan dedup** — duplikat dihapus *setelah* imputasi; urutan lazim sebaliknya. Dampak praktis nol (33 baris), cukup sadari.
5. **`int_rate_bin` ARM masih qcut** sementara fitur lain di-bin ambang domain — inkonsistensi kecil; bisa diganti ambang domain (mis. <10 / 10–15 / 15–20 / >20%) atau diberi satu kalimat justifikasi.
6. **Kopling tersembunyi sample UMAP** — Phase 2 mereproduksi indeks sample UMAP dengan `RandomState(42).choice(...)` yang identik dengan pipeline; valid (kuverifikasi call-nya sama persis), tapi rapuh terhadap perubahan panjang data. Lebih aman: simpan `orig_index` sebagai kolom di `lending_club_umap.csv`.
7. **Jumlah record vs dokumen soal** — soal menyebut ~890k record; kalian pakai 2,26 juta (justified & didokumentasikan di report). Siapkan jawaban singkat kalau ditanya saat presentasi.

---

## D. Yang sudah EXCELLENT (pertahankan)

1. **Phase 1**: verifikasi mekanisme missing (structural vs MAR, dengan uji empiris), anti-leakage, flag+sentinel, winsorization/log **data-driven dengan guard** (bukan daftar hardcode), feature selection korelasi + MI (dua metode sesuai rubrik), tabel justifikasi parameter lengkap, catatan etis fair-lending. Config terpusat di `config.py` — semua ambang punya alasan tertulis.
2. **Phase 2**: uji stabilitas K lintas 5 sample (bukan satu elbow rapuh), bukti chaining untuk pemilihan Ward (bukan sekadar klaim), Cophenetic + ARI + crosstab, penamaan cluster otomatis anti label-swap, radar + bar + interpretasi bisnis per profil, profil outlier "stretched high-earners", limitasi jujur. Semua angka markdown cocok dengan output eksekusi.
3. **Phase 3**: diskretisasi berbasis standar industri dengan rationale per fitur, treatment sirkularitas grade↔int_rate yang jujur dan cerdas, temuan negatif small_business dilaporkan apa adanya, 10 rule non-trivial dengan interpretasi actionable + bukti numerik.
4. **Phase 4**: 4 metode + korroborasi ≥2, desain anti-sirkularitas risk_signal (dibentuk dari atribut, divalidasi ke outcome: 19,2% vs 13,1%, lift 1,47×), klasifikasi error/rare/risk + tipologi point/contextual/collective, cross-reference DBSCAN.
5. **Report**: 3 temuan dari 3 fase berbeda, masing-masing lolos 4 uji translasi; Limitations lengkap 4 poin; total 10 halaman (body < limit); bahasa sudah bergaya humanizer.

---

## E. Urutan pengerjaan yang kusarankan

1. Fix sampling Phase 3 (A1) + hitung rule sebelum filter (A2) → re-run → update angka di report & dashboard.
2. Restrukturisasi report sesuai template + cover fields + exec summary 3 kalimat (A3) + perbaiki label Appendix B (A4) + item B2/B3.
3. Buat notebook Phase 5 pembuat dashboard (B1) + tambah 1–2 kontrol interaktif (B4).
4. Git init + perbaiki .gitignore + requirements.txt (A5).
5. Sapu minor (C1–C7) seperlunya.
