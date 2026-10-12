# TB1 JST: Klasifikasi Breast Cancer Wisconsin (Diagnostic)

Perceptron dan ADALINE mandiri dengan NumPy. Semua data dan hasil berada di repository, sehingga pelatihan tidak memerlukan akses Kaggle setelah dependensi terpasang.

## Identitas dan status

Mata kuliah: Jaringan Syaraf Tiruan IT6053, FTI UKDW. Nomor kelompok **XX**, isi sesuai kelompok final. Kelompok wajib terdiri dari 2-3 mahasiswa.

| Anggota final | NIM | Kontribusi aktual | Bukti verifikasi / bagian paparan |
|---|---|---|---|
| Nama anggota 1 (isi) | NIM (isi) | Isi sesuai pekerjaan nyata | Commit/catatan Run All/bagian paparan |
| Nama anggota 2 (isi) | NIM (isi) | Isi sesuai pekerjaan nyata | Commit/catatan Run All/bagian paparan |

Tambahkan anggota ketiga jika diperlukan. Daftar lima orang pada Tugas 1 tidak otomatis menjadi daftar kelompok TB1. Isi `identitas.json`, README serta identitas pada notebook/laporan/slide. Jangan membuat kontribusi fiktif. Ganti XX pada nama repository, notebook, laporan dan rujukannya setelah nomor diketahui.

**Konfirmasi penggantian:** pengguna mengonfirmasi konsultasi/persetujuan setelah pertanyaan tentang C.1 pada 12 Oktober 2026. Bukti tertulis, jika diperlukan, disediakan kelompok. Repository publik, identitas final, Run All di komputer kelompok, dan commit pengumpulan masih perlu dilengkapi. Lihat [STATUS_PENGUMPULAN.md](STATUS_PENGUMPULAN.md).

## Masalah dan kaitan dengan Tugas 1

Tugas 1 membahas segmentasi Mall Customers tanpa target dengan rencana SOM/Kohonen. TB1 ini beralih ke **klasifikasi dua kelas** diagnosis sampel payudara. Pergantian mengubah kasus/dataset, bukan eksperimen lanjutan pada observasi pelanggan yang sama. Kelas asli sudah berlabel, tidak dibentuk dari cluster atau threshold fitur.

Target: `B=-1` (benign/jinak), `M=+1` (malignant/ganas, positif). Model menggunakan empat karakteristik numerik inti sel. Ini latihan benchmark, bukan alat diagnosis klinis. Nilai z bukan probabilitas.

## Berkas wajib dan tambahan

| Berkas | Isi |
|---|---|
| [TB1_KelompokXX.ipynb](TB1_KelompokXX.ipynb) | Notebook lengkap, kode inti terlihat, keluaran eksekusi, manual, evaluasi dan pembahasan |
| [TB1_KelompokXX.pdf](TB1_KelompokXX.pdf) | 8 halaman isi utama, 1 sampul, 1 referensi, 2 lampiran manual |
| [dataset.csv](dataset.csv) | 569 baris, empat fitur skala asli, diagnosis asli dan label bipolar |
| [Presentasi_TB1.pdf](Presentasi_TB1.pdf) | 11 slide utama dan 2 cadangan |
| [Presentasi_TB1.pptx](Presentasi_TB1.pptx) | Slide editable, tabel dan grafik native |
| [requirements.txt](requirements.txt) | Dependensi dan versi untuk Python 3.12 |
| [identitas.json](identitas.json) | Tempat identitas final untuk membangun ulang laporan |
| [PANDUAN_PRESENTASI.md](PANDUAN_PRESENTASI.md) | Alokasi 8 menit, demo, dan jawaban tanya jawab |
| [STATUS_PENGUMPULAN.md](STATUS_PENGUMPULAN.md) | Pemetaan rubrik dan checklist administratif |
| [DATA_LICENSE.md](DATA_LICENSE.md) | Atribusi, perbedaan metadata lisensi, perubahan dataset |

### Kode, data, dan bukti

- [src/prepare_dataset.py](src/prepare_dataset.py): membangun ulang CSV dari file mentah lokal.
- [src/eksperimen.py](src/eksperimen.py): dua model NumPy, seluruh konfigurasi, prediksi dan grafik.
- [src/verify.py](src/verify.py): pemeriksaan aljabar/gradient/split/scaler/metrik/bobot/manual.
- [src/check_bundle.py](src/check_bundle.py): audit berkas, tautan README, notebook, halaman dan konsistensi angka antarformat.
- [src/execute_inprocess.py](src/execute_inprocess.py): menjalankan sel berurutan tanpa kernel socket dengan capture output.
- [src/run_notebook.py](src/run_notebook.py): runner kernel Jupyter standar setelah dependensi terpasang.
- [src/build_notebook.py](src/build_notebook.py): menyusun notebook dari implementasi dan hasil, lalu mengeksekusi semua sel.
- [src/build_report.py](src/build_report.py): membuat ulang laporan PDF.
- [src/content.py](src/content.py), [src/build_readme.py](src/build_readme.py): narasi/referensi dan README dari hasil aktual.
- [data/raw_breast_cancer.csv](data/raw_breast_cancer.csv): file sumber unggahan, byte asli dipertahankan.
- [data/provenance.json](data/provenance.json): sumber/lisensi/audit/transformasi/hash.
- [results/metrics.csv](results/metrics.csv): seluruh delapan konfigurasi dan metrik akhir.
- [results/history.csv](results/history.csv): delapan riwayat × 50 epoch (400 catatan).
- [results/checkpoints.csv](results/checkpoints.csv): 24 checkpoint train epoch 10/30/50.
- [results/weights.csv](results/weights.csv): semua bobot dan bias akhir.
- [results/predictions_test.csv](results/predictions_test.csv): 114 test × 8 konfigurasi, label/prediksi/z.
- [results/split.csv](results/split.csv): indeks dan urutan train/test, row_id mulai 0 pada dataset.csv.
- [results/scaler.csv](results/scaler.csv): mean/std train-only.
- [results/manual.json](results/manual.json): dua update Perceptron dan dua batch ADALINE, presisi penuh.
- [results/summary.json](results/summary.json): konfigurasi, lingkungan, stabilitas dan hash.
- [results/verification.json](results/verification.json): pemeriksaan numerik.
- [results/bundle_verification.json](results/bundle_verification.json): pemeriksaan kelengkapan dan konsistensi berkas.
- [figures/data_train.png](figures/data_train.png), [learning_scaled.png](figures/learning_scaled.png), [scaling_comparison.png](figures/scaling_comparison.png), [confusion_all.png](figures/confusion_all.png), [boundary_slice.png](figures/boundary_slice.png).

## Dataset, izin, dan transformasi

Sumber Kaggle: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data), organisasi UCI Machine Learning. Sumber asli: [UCI, DOI 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). Diakses 12 Oktober 2026.

Metadata Kaggle menyebut **CC BY-NC-SA 4.0**, sedangkan UCI asli **CC BY 4.0**. Paket mempertahankan atribusi dan mengikuti ketentuan Kaggle secara konservatif untuk salinan unggahan serta CSV turunannya. Lihat DATA_LICENSE.md sebelum publikasi.

Audit: 569 baris, 33 kolom, B=357 (62.74%), M=212 (37.26%). Kolom `Unnamed: 32` memiliki 569 nilai kosong dan dikeluarkan. Tidak ada missing pada 30 pengukuran atau diagnosis, tidak ada duplikasi baris maupun duplikasi 30 fitur. Semua ID unik. Tidak ada imputasi, deduplikasi, oversampling atau penghapusan outlier. Semua baris dipertahankan.

Empat fitur ditetapkan secara apriori sebelum melihat hasil test. Pemilihan memudahkan manual dan mencakup ukuran/tekstur/bentuk. Tidak sekaligus menggunakan radius/perimeter/area yang sama-sama terkait ukuran. Subset tidak diklaim optimal.

| Kolom dataset.csv | Makna / peran |
|---|---|
| radius_mean | Rerata jarak dari pusat inti sel ke titik pada perimeter. Mewakili ukuran. |
| texture_mean | Rerata karakteristik tekstur, dihitung dari simpangan baku intensitas keabuan. |
| smoothness_mean | Rerata variasi lokal panjang radius. Mewakili kehalusan batas inti sel. |
| concavity_mean | Rerata tingkat cekungan bagian kontur inti sel. |
| diagnosis | Label asli B/M, tidak menjadi fitur |
| label | B=-1, M=+1, target bipolar; M adalah kelas positif |

ID tidak menjadi fitur dan tidak terdapat pada CSV eksperimen. File mentah mempertahankan pengenal rekaman riset publik. CSV eksperimen tidak mengganti nilai dengan nilai standar: standardisasi dilakukan setelah split, memakai parameter train saja.

## Rancangan terkunci dan rumus

- Python **3.12.14**, seed **42**, split 80:20 terstratifikasi: 455 train (B=285,M=170), 114 test (B=72,M=42).
- X hanya empat fitur. Diagnosis dan label tidak masuk X.
- StandardScaler di-fit pada train, simpangan baku ddof=0. Tidak ada transformasi yang dipelajari dari test.
- w=(0,0,0,0), b=0. Urutan train tetap sepanjang epoch dan sama untuk semua konfigurasi.
- Perceptron/ADALINE masing-masing eta=0.001,0.01,0.1 pada fitur standar, 50 epoch.
- Kontrol masing-masing model eta=0.01 tanpa standardisasi, 50 epoch. Total 8 pelatihan.
- Checkpoint 10/30/50 berasal dari riwayat train yang sama. Semua model dilatih dahulu, baru dilaporkan test terakhir per konfigurasi. Tidak ada seleksi model/parameter. Fokus eta=0.01 ditetapkan apriori.

Neuron: `z=Xw+b`, prediksi +1 jika z>=0, selain itu -1. Perceptron per sampel: `delta=eta*(y-yhat)`, `w+=delta*x`, `b+=delta`. Catat update dalam epoch dan salah train setelah epoch terpisah.

ADALINE: `e=y-(Xw+b)`, `w+=eta*(X.T@e)/n`, `b+=eta*mean(e)`. J=`mean(e_after**2)/2`, dicatat sesudah update. MSE=2J. Threshold hanya untuk prediksi. Jika loss atau parameter menjadi inf/NaN, berhenti dengan parameter valid terakhir serta catat epoch gagal. Loss raw masih finite hingga 50, sehingga tidak dihentikan. Cabang non-finite diuji dengan contoh ekstrem terpisah.

## Instalasi dan menjalankan di Windows / VS Code

Ekstrak ZIP ke folder baru `TB1-JST-KelompokXX`, jangan gabungkan hasil lama dan jangan jalankan dari dalam ZIP. Buka terminal di direktori utama proyek. Gunakan Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name tb1-jst --display-name "Python (TB1 JST)"
.\.venv\Scripts\python.exe src\eksperimen.py
.\.venv\Scripts\python.exe src\verify.py
```

Di VS Code buka folder proyek, gunakan ekstensi Jupyter, buka notebook, pilih kernel virtual environment atau **Python (TB1 JST)**. Klik Restart Kernel lalu Run All. Setelah dependensi dipasang, eksperimen tidak membutuhkan internet.

Alternatif tanpa kernel Jupyter:

```powershell
.\.venv\Scripts\python.exe src\execute_inprocess.py
```

Runner kernel standar:

```powershell
.\.venv\Scripts\python.exe src\run_notebook.py
```

macOS/Linux: gunakan `python3 -m venv .venv` dan `.venv/bin/python`.

### Membangun ulang keluaran

```powershell
.\.venv\Scripts\python.exe src\prepare_dataset.py
.\.venv\Scripts\python.exe src\build_notebook.py
.\.venv\Scripts\python.exe src\build_report.py
.\.venv\Scripts\python.exe src\build_readme.py
```

Notebook builder juga menjalankan semua model dan verifikasi. Edit identitas.json sebelum membangun laporan. Slide PPTX dapat diedit di PowerPoint/LibreOffice dan diekspor menjadi PDF dengan nama wajib. Jika dataset/konfigurasi berubah, revisi narasi dan seluruh hasil secara konsisten. Paket ini untuk rancangan tetap yang tercatat.

## Ringkasan hasil aktual

| ID | Model | eta | Standar | Epoch | Acc. train | Acc. test | Salah train | U / J akhir |
|---|---|---:|:---:|---:|---:|---:|---:|---|
| P-S-0001 | Perceptron | 0.001 | Ya | 50 | 92.09% | 89.47% | 36 | U=43 |
| P-S-001 | Perceptron | 0.01 | Ya | 50 | 92.09% | 89.47% | 36 | U=43 |
| P-S-01 | Perceptron | 0.1 | Ya | 50 | 92.09% | 89.47% | 36 | U=43 |
| A-S-0001 | ADALINE | 0.001 | Ya | 50 | 93.41% | 93.86% | 30 | J=0.441136 |
| A-S-001 | ADALINE | 0.01 | Ya | 50 | 93.41% | 92.11% | 30 | J=0.218392 |
| A-S-01 | ADALINE | 0.1 | Ya | 50 | 92.53% | 90.35% | 34 | J=0.164894 |
| P-R-001 | Perceptron | 0.01 | Tidak | 50 | 40.88% | 39.47% | 269 | U=116 |
| A-R-001 | ADALINE | 0.01 | Tidak | 50 | 37.36% | 36.84% | 285 | J=6.659e+67 |

Confusion matrix baris aktual dan kolom prediksi berurutan [-1,+1], `[[TN,FP],[FN,TP]]`. Precision/recall/F1 M serta balanced accuracy seluruh konfigurasi tersedia di CSV dan notebook.

Tiga eta Perceptron standar memberi riwayat update dan prediksi yang sama. Dengan w=b=0, urutan tetap, eta positif konstan, dan ambang nol, bobot serta bias proporsional terhadap eta sehingga tanda z tidak berubah. Hasil ini sah dan telah diperiksa, bukan bukti eta selalu tidak berpengaruh pada setiap implementasi.

Kesalahan latih Perceptron standar pada epoch 10/30/50 adalah 34/42/36. Kesalahan tidak turun monoton. Pada ADALINE standar, eta 0.1 menghasilkan J akhir 0.164894, lebih rendah daripada eta 0.01 (0.218392), tetapi akurasi uji 90.35% justru lebih rendah daripada 92.11%. Objektif kuadrat dan kesalahan kelas mengukur hal berbeda. Epoch 50 adalah anggaran tetap, bukan epoch terbaik hasil seleksi data uji.

Pada eta=0.01, Perceptron standar mencapai akurasi uji 89.47%, sedangkan raw 39.47%. ADALINE standar mencapai 92.11%, sedangkan raw 36.84% dengan J=6.659e+67. Skala asli radius/texture jauh lebih besar daripada smoothness/concavity, sehingga kontribusi langkah tidak seimbang. ADALINE raw mengalami divergensi loss, bukan sekadar konvergensi lambat.

Untuk J kuadratik, H=A.T@A/n dengan A=[X,1]. Rentang stabil gradient descent adalah 0<eta<2/lambda_max(H). Dari train diperoleh batas 0.954260 pada fitur standar dan 0.003334 pada fitur raw. Eta 0.01 melampaui batas raw. J masih finite hingga epoch 50, sehingga pelatihan tidak dihentikan dan epoch non-finite dicatat sebagai tidak terjadi. Tidak ada perubahan eta setelah melihat hasil uji.

Kelas M (+1) berjumlah 42 dari 114 data uji. Perceptron standar eta=0.01 memiliki FN=9, FP=3, recall=0.7857. ADALINE standar eta=0.01 memiliki FN=6, FP=3, recall=0.8571. FN berarti sampel ganas diprediksi jinak, sehingga akurasi saja tidak cukup. Ini hanya evaluasi benchmark, bukan rekomendasi diagnosis klinis.

Satu split dengan 114 data uji membatasi generalisasi. Satu perubahan prediksi setara 0.88 poin persentase. Empat dari 30 fitur dipilih secara apriori, sehingga informasi lain tidak digunakan dan subset ini tidak diklaim optimal. Ketergantungan antarfitur dan asal satu benchmark membatasi transfer ke populasi lain. Tidak ada validasi eksternal atau kalibrasi probabilitas. Z bukan probabilitas.

`boundary_slice.png` menampilkan irisan model empat fitur saat smoothness_mean dan concavity_mean standar=0 (rerata train). Ini bukan model dua fitur atau plot prediksi semua titik proyeksi.

## Verifikasi dan batas pengujian

CLI dan semua sel kode notebook dieksekusi berurutan dengan Python satu proses. Keluaran stdout, tabel HTML dan PNG direkam dari eksekusi nyata. Paket tidak mengklaim telah menjalankan kernel Jupyter socket dalam lingkungan pengerjaan. Run All memakai kernel biasa tetap perlu dilakukan di komputer kelompok. Dependency core telah digunakan; dependency kernel/runner disertakan untuk pemasangan di komputer kelompok.

Pemeriksaan numerik: dua update aljabar Perceptron, dua batch ADALINE, finite difference gradient, tie z=0, cabang non-finite, mean/std train-only, split tanpa overlap, 400 riwayat dan 24 checkpoint, metrik dari prediksi, rekonstruksi parameter/J/kesalahan latih, invariansi skala Perceptron, dan dua langkah manual aktual konsisten dengan train.

## Bantuan AI dan kontribusi

ChatGPT membantu draf kode, penjelasan, perhitungan manual, grafik, laporan dan presentasi. Data diunggah pengguna dari Kaggle. Angka dihasilkan oleh program, bukan dibuat agar akurasi tinggi. Kelompok bertanggung jawab menjalankan ulang, memahami kode, mengecek perhitungan, serta mencatat kontribusi nyata pada tabel identitas. Verifikasi otomatis tidak menggantikan penguasaan individu.

## Referensi dan adaptasi

1. UCI Machine Learning. Breast Cancer Wisconsin (Diagnostic) Data Set. Kaggle. https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data. Lisensi pada metadata Kaggle: CC BY-NC-SA 4.0.

2. Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. Lisensi sumber asli: CC BY 4.0.

3. Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits. IRE WESCON Convention Record, Part 4, 96-104. https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf. Artikel fundamental asli ADALINE.

4. Raschka, S. (2015, 24 Maret). Single-Layer Neural Networks and Gradient Descent. https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html. Tutorial dan kode pendukung.

5. Raschka, S. (tanpa tahun). Adaline: Adaptive Linear Neuron Classifier. Dokumentasi mlxtend. https://rasbt.github.io/mlxtend/user_guide/classifier/Adaline/.

6. scikit-learn developers. StandardScaler; train_test_split. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html dan https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html. Versi eksekusi 1.8.0.

7. Panduan internal mata kuliah (2026). Tugas Besar 1: Analisis dan Implementasi Klasifikasi Data Menggunakan Perceptron dan ADALINE. IT6053, FTI UKDW. Sumber ketentuan tugas yang diberikan pengguna.

Artikel Widrow & Hoff dipakai sebagai landasan ilmiah. Raschka/mlxtend mendukung struktur neuron dan linear output vs threshold. Kode disusun ulang dengan bias terpisah, normalisasi /n, J setelah update, urutan tetap, history/checkpoint dan non-finite. Standardisasi sebelum split dari contoh tidak diikuti. Semua sumber daring diakses 12 Oktober 2026.

## Pengumpulan

Nama repository wajib **TB1-JST-KelompokXX** sesuai nomor final dan status **Public**. Unggah semua berkas langsung ke repository, bukan hanya ZIP. README berada di root dan menautkan file. Uji lewat browser privat tanpa login, catat URL dan commit hash, lalu kirim sesuai kanal/tenggat dosen. Paket ini belum menerbitkan repository atau mengirimkan tugas.
