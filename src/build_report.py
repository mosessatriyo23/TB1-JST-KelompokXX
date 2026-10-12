"""Laporan PDF: 8 halaman utama + sampul, referensi, 2 lampiran manual."""
from pathlib import Path
import html
import json
import pandas as pd
import matplotlib
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle, Image
from reportlab.lib.styles import ParagraphStyle
from content import FEATURE_INFO, REFERENCES, analysis, load_results, number

BASE = Path(__file__).resolve().parents[1]
FONT = str(Path(matplotlib.get_data_path()) / "fonts/ttf")
pdfmetrics.registerFont(TTFont("DV", f"{FONT}/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", f"{FONT}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")
INK = colors.HexColor("#183348")
TEAL = colors.HexColor("#117C83")
GRAY = colors.HexColor("#536675")
LIGHT = colors.HexColor("#E9F2F4")
M, W, C, S, MAN = load_results(BASE)
AN = analysis(BASE)
IDENT = json.loads((BASE / "identitas.json").read_text())
STYLES = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.4, leading=14, textColor=INK, spaceAfter=7),
    "small": ParagraphStyle("small", fontName="DV", fontSize=8.1, leading=11.5, textColor=GRAY, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=12, leading=16, textColor=TEAL, spaceAfter=7),
    "formula": ParagraphStyle("formula", fontName="DV", fontSize=10, leading=16, textColor=INK, backColor=LIGHT, borderPadding=7, spaceAfter=12),
}

class Report:
    def __init__(self, path):
        self.c = canvas.Canvas(str(path), pagesize=A4)
        self.c.setTitle("TB1 JST - Breast Cancer Wisconsin Diagnostic")
        self.page = 0
        self.width, self.height = A4
        self.left = 43
        self.usable = self.width - 86
    def start(self, title, label):
        if self.page: self.c.showPage()
        self.page += 1
        self.y = self.height - 75
        self.c.setFillColor(TEAL)
        self.c.setFont("DVB", 8)
        self.c.drawString(43, self.height-40, "JARINGAN SYARAF TIRUAN / IT6053 / TUGAS BESAR 1")
        self.c.setFillColor(INK)
        self.c.setFont("DVB", 16.5)
        self.c.drawString(43, self.y, title)
        self.y -= 27
        self.c.setFont("DV", 7.5)
        self.c.setFillColor(GRAY)
        self.c.drawString(43, 29, "Breast Cancer Wisconsin (Diagnostic) / Eksperimen NumPy")
        self.c.drawRightString(self.width-43, 29, label)
    def add(self, text, style="body"):
        p = Paragraph(text, STYLES[style])
        _, h = p.wrap(self.usable, 1000)
        if self.y-h < 48: raise RuntimeError(f"Overflow halaman {self.page}, y={self.y}, h={h}: {text[:60]}")
        p.drawOn(self.c, self.left, self.y-h)
        self.y -= h+STYLES[style].spaceAfter
    def gap(self, n=6): self.y -= n
    def table(self, headers, rows, widths=None, size=7.5):
        st = ParagraphStyle("cell", fontName="DV", fontSize=size, leading=size+3, textColor=INK)
        values = [[Paragraph(str(v), st) for v in row] for row in [headers]+rows]
        widths = widths or [self.usable/len(headers)]*len(headers)
        t = Table(values, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), LIGHT),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LINEBELOW", (0,0), (-1,0), .7, TEAL),
            ("LINEBELOW", (0,1), (-1,-1), .25, colors.HexColor("#D8E2E8")),
            ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        _, h = t.wrap(self.usable, 1000)
        if self.y-h < 48: raise RuntimeError(f"Tabel overflow halaman {self.page}, y={self.y}, h={h}")
        t.drawOn(self.c, self.left, self.y-h)
        self.y -= h+10
    def image(self, name, height=None):
        im = Image(str(BASE / f"figures/{name}.png"))
        w = self.usable
        h = w*im.imageHeight/im.imageWidth
        if height and h > height: w *= height/h; h = height
        if self.y-h < 48: raise RuntimeError(f"Gambar overflow halaman {self.page}")
        im.drawWidth = w; im.drawHeight = h
        im.drawOn(self.c, self.left+(self.usable-w)/2, self.y-h)
        self.y -= h+8
    def save(self): self.c.save()

def esc(s): return html.escape(s)
def vec(values): return "(" + ", ".join(f"{v:.6f}" for v in values) + ")"
def pct(v): return f"{100*v:.2f}%"

r = Report(BASE / "TB1_KelompokXX.pdf")
r.start("Klasifikasi Diagnosis Sampel Payudara", "Sampul")
r.gap(36)
r.add("Breast Cancer Wisconsin (Diagnostic)", "h2")
r.add("Perbandingan Perceptron dan ADALINE<br/>menggunakan empat fitur numerik inti sel", "h2")
r.add("Tugas Besar 1 Jaringan Syaraf Tiruan (IT6053)<br/>Fakultas Teknologi Informasi, Universitas Kristen Duta Wacana")
r.gap(25)
r.add(f"<b>Kelompok: {esc(IDENT['nomor_kelompok'])}</b><br/>Tanggal penyusunan: 12 Oktober 2026")
r.table(["Anggota final", "NIM"], [[esc(x['nama']),esc(x['nim'])] for x in IDENT['anggota']], widths=[365,r.usable-365], size=9)
r.add("Nomor kelompok serta nama/NIM pada tempat isian perlu dilengkapi. Kelompok mengikuti ketentuan 2-3 mahasiswa. Kontribusi anggota ditulis sesuai pekerjaan aktual, bukan diatribusikan oleh AI.", "small")
r.gap(30)
r.add("Ruang lingkup laporan", "h2")
r.add("Laporan memuat 8 halaman isi utama, di luar sampul, satu halaman daftar pustaka, dan dua halaman lampiran manual. Semua angka hasil berasal dari eksekusi pada dataset yang disertakan, bukan ilustrasi.")
r.add("Pergantian dari data segmentasi pelanggan ke dataset berlabel dilakukan sesuai konfirmasi konsultasi dari kelompok. File CSV lokal memungkinkan seluruh eksperimen dijalankan tanpa mengakses Kaggle lagi setelah dependensi tersedia.")
r.add("Studi ini adalah latihan klasifikasi benchmark. Model tidak dinyatakan sebagai alat diagnosis klinis atau pengganti pemeriksaan dokter.", "small")

r.start("1. Masalah dan tujuan", "Isi 1 / 8")
r.add("1.1 Perumusan kasus", "h2")
r.add("Dataset berisi pengukuran karakteristik inti sel pada sampel payudara yang telah memiliki label diagnosis. Masalah dirumuskan sebagai supervised binary classification: model menerima fitur numerik dan mempelajari keputusan antara sampel jinak (B) dan ganas (M). Kedua kelas memang sudah tersedia dari sumber. Label tidak dibuat dari threshold fitur atau hasil clustering.")
r.add("Pengukuran citra aspirasi jarum halus atau fine needle aspirate (FNA) menjadi konteks sumber data [1-2]. Eksperimen ini bekerja pada angka hasil ekstraksi, bukan memproses citra mikroskop dari awal. Hasil tidak ditafsirkan sebagai prediksi risiko pada populasi umum.")
r.add("1.2 Tujuan eksperimen", "h2")
r.add("Tujuan pertama ialah mengimplementasikan neuron linear, Perceptron, dan ADALINE secara mandiri dengan NumPy mengikuti persamaan panduan. Tujuan berikutnya ialah membandingkan mekanisme error, pengaruh learning rate, perkembangan epoch, serta pengaruh standardisasi dengan pengendalian split dan kondisi awal.")
r.add("Evaluasi mencakup akurasi latih/uji, jumlah kesalahan latih, confusion matrix, precision, recall dan F1 untuk kelas M. Riwayat pembaruan Perceptron dan loss ADALINE membantu menjelaskan proses pembelajaran. Akurasi sempurna tidak menjadi syarat keberhasilan studi.")
r.add("1.3 Hubungan dengan Tugas 1", "h2")
r.add("Presentasi Tugas 1 membahas segmentasi pelanggan Mall Customers, tanpa target segmen, dengan rencana SOM/Kohonen. Panduan TB1 menuntut Perceptron dan ADALINE untuk dua kelas. Karena itu, kasus diubah menjadi klasifikasi diagnosis pada dataset berlabel. Studi ini tidak mengklaim tetap memakai observasi pelanggan atau melanjutkan eksperimen clustering yang sama.")
r.add("Penggantian dataset telah dikonsultasikan sesuai bagian C.1 berdasarkan konfirmasi kelompok. Sumber Kaggle dipilih dan file data.csv diunggah langsung, sehingga masalah akses sumber sebelumnya tidak menjadi ketergantungan eksperimen. Bukti persetujuan tambahan, jika diminta dosen, tetap menjadi tanggung jawab kelompok.")
r.add("1.4 Keputusan ruang lingkup", "h2")
r.add("Empat fitur yang telah direkomendasikan sebelum evaluasi dipakai untuk menjaga keterbacaan perhitungan manual: radius_mean, texture_mean, smoothness_mean, dan concavity_mean. Pemilihan ini mencakup ukuran, tekstur dan bentuk, serta tidak sekaligus menggunakan radius, perimeter dan area yang sama-sama terkait ukuran. Tidak dilakukan pencarian subset berdasarkan hasil data uji.")
r.add("Semua 569 baris dipertahankan. Model tetap berdimensi empat. Visualisasi batas keputusan memperlihatkan irisan hiperbidang dengan dua fitur lain ditahan pada rerata train, bukan hasil pelatihan model dua fitur. Keputusan desain ini dicatat sebelum pelaporan data uji.")

r.start("2. Dataset dan label", "Isi 2 / 8")
r.add("2.1 Sumber dan izin", "h2")
r.add("Data berasal dari file data.csv yang diunggah kelompok dari halaman Kaggle UCI Machine Learning [1]. Sumber asli adalah Breast Cancer Wisconsin (Diagnostic), DOI 10.24432/C5DW2B [2]. Metadata Kaggle menyebut CC BY-NC-SA 4.0, sedangkan UCI menyebut CC BY 4.0. Untuk salinan Kaggle dan CSV turunannya, paket mempertahankan atribusi serta ketentuan nonkomersial dan berbagi-serupa Kaggle. Kedua lisensi dibedakan dalam DATA_LICENSE.md.")
r.table(["Kolom dataset.csv", "Makna dan peran"], [[f,esc(d)] for f,d in FEATURE_INFO] + [
    ["diagnosis", "Target asli B atau M, tidak menjadi fitur."],
    ["label", "Target bipolar: B=-1, M=+1. Kelas positif ialah M."],
], widths=[135,r.usable-135], size=8)
r.add("2.2 Audit dan distribusi", "h2")
r.add("File mentah memiliki 569 baris dan 33 kolom: 30 pengukuran, id, diagnosis, dan Unnamed: 32. Kolom terakhir berisi 569 nilai kosong dan dibuang dari data eksperimen. Tidak ada missing value pada fitur/target atau duplikasi baris. Semua ID unik dan 30 fitur tidak mempunyai baris duplikat. ID adalah pengenal rekaman publik dan tidak dimasukkan ke X. Tidak diperlukan imputasi, deduplikasi atau penghapusan outlier.")
r.table(["Subset", "Jumlah", "B (-1)", "M (+1)"], [["Keseluruhan",569,357,212],["Train",455,285,170],["Test",114,72,42]], size=8.5)
r.image("data_train", height=163)
r.add("Gambar 1. Proyeksi data latih pada dua pasangan dari empat fitur. Pola tumpang tindih pada proyeksi tidak menentukan separabilitas ruang empat dimensi.", "small")

r.start("3. Rujukan dan model neuron", "Isi 3 / 8")
r.add("3.1 Artikel ilmiah dan kode pendukung", "h2")
r.table(["Sumber", "Kontribusi dan adaptasi"], [
    ["Widrow &amp; Hoff (1960) [3]", "Artikel fundamental asli tentang pembelajaran adaptif dengan error kuadrat. Pembaruan batch rerata mengikuti panduan TB1, bukan klaim bahwa seluruh kode identik dengan perangkat asli."],
    ["Raschka (2015) [4]", "Struktur loop neuron dan aturan Perceptron/ADALINE. Kode disusun ulang dengan bias terpisah, w=b=0, urutan tetap, dan pencatatan tiap epoch."],
    ["mlxtend [5]", "Linear output vs threshold serta gradient descent. Estimator siap pakai tidak dipanggil. Gradient dibagi n dan J dicatat sesudah update sesuai rubrik."],
    ["scikit-learn [6]", "Hanya split, scaler dan metrik. Parameter scaler dipelajari dari train. Contoh standardisasi sebelum split pada rujukan tidak diikuti."],
], widths=[128,r.usable-128], size=8)
r.add("3.2 Neuron dan Perceptron", "h2")
r.add("z = wᵀx + b; ŷ = +1 jika z ≥ 0, selain itu -1<br/>Δ = η(y - ŷ); w baru = w + Δx; b baru = b + Δ", "formula")
r.add("Empat bobot memberi kontribusi linear fitur, bias menggeser hiperbidang, dan z bukan probabilitas. Perceptron memakai error label. Jika prediksi benar, Δ=0. Jika salah, Δ=±2η. Update berlangsung segera per sampel. Jumlah update dalam epoch berbeda dari kesalahan seluruh train sesudah epoch karena parameter terus berubah saat loop berjalan.")
r.add("3.3 ADALINE dan Delta Rule", "h2")
r.add("e = y - (Xw + b); J = Σeᵢ²/(2n); MSE = 2J<br/>w baru = w + ηXᵀe/n; b baru = b + η mean(e)", "formula")
r.add("Error berasal dari output linear. Semua komponen gradient memakai bobot lama yang sama. Ambang dipakai hanya untuk prediksi/metrik. J dicatat sesudah pembaruan batch. Jika parameter atau J menjadi inf/NaN, pelatihan berhenti dan state valid terakhir dipertahankan. Dua sampel Perceptron serta dua batch aktual ADALINE pada η=0,01 ditelusuri di Lampiran A-B.")

r.start("4. Penyiapan dan rancangan eksperimen", "Isi 4 / 8")
r.add("4.1 Pengendalian data", "h2")
r.add("Split 80:20 menggunakan train_test_split(test_size=0.2, random_state=42, stratify=y), sehingga diperoleh 455 train dan 114 test. Split, inisialisasi w=(0,0,0,0), b=0 dan urutan sampel dipertahankan untuk seluruh konfigurasi. Semua epoch menggunakan urutan hasil split yang sama, tanpa pengacakan ulang.")
r.add("x standar = (x - μ train) / σ train, dengan ddof=0", "formula")
sc = pd.read_csv(BASE / "results/scaler.csv")
r.table(["Fitur", "μ train", "σ train"], [[q.feature,f"{q.train_mean:.9f}",f"{q.train_std_ddof0:.9f}"] for q in sc.itertuples()], widths=[173,167,r.usable-340], size=8)
r.add("CSV tetap pada skala asli. Mean/std dihitung hanya dari train, lalu transformasi yang sama diterapkan pada test. Data uji tidak memberi kontribusi pada imputasi, scaler, atau parameter model [6].")
r.add("4.2 Delapan konfigurasi apriori", "h2")
r.table(["ID", "Model", "η", "Skala", "Epoch"], [[q.id,q.model,f"{q.eta:g}","Standar" if q.standardized else "Asli",50] for q in M.itertuples()], widths=[90,110,55,105,r.usable-360], size=8)
r.add("Enam pelatihan utama mengikuti learning rate 0,001; 0,01; 0,1. Dua kontrol tanpa standardisasi memakai η=0,01. Anggaran 50 epoch ditetapkan tanpa early stopping pada nol kesalahan. Semua konfigurasi selesai 50 epoch. Loss raw yang besar tetapi finite tidak dihentikan atau diganti dengan eksperimen yang tampak lebih baik.")
r.add("4.3 Penggunaan data uji", "h2")
r.add("Konfigurasi dan empat fitur ditentukan sebelum evaluasi test. Seluruh model dilatih terlebih dahulu, lalu hasil uji akhir tiap konfigurasi dilaporkan. Tidak dilakukan seleksi model/parameter. Fokus η=0,01 dipilih sebelum evaluasi. Checkpoint 10/30/50 diambil dari riwayat train yang sama. Seleksi lanjutan harus memakai validasi di dalam train.")

r.start("5. Hasil akhir dan parameter", "Isi 5 / 8")
r.add("5.1 Tabel seluruh konfigurasi", "h2")
rows=[]
for q in M.itertuples():
    end=f"U={int(q.final_updates)}" if q.model=="Perceptron" else f"J={number(q.final_J)}"
    rows.append([q.id,q.model,f"{q.eta:g}","Ya" if q.standardized else "Tidak",int(q.completed_epochs),pct(q.train_accuracy),pct(q.test_accuracy),int(q.train_errors),end])
r.table(["ID", "Model", "η", "Std", "E", "Acc.<br/>train", "Acc.<br/>test", "Salah<br/>train", "Update / loss akhir"], rows,
        widths=[70,66,35,31,25,57,57,44,r.usable-385], size=7.1)
r.add("E=epoch selesai; U=update salah klasifikasi pada epoch terakhir. J dan MSE=2J hanya berlaku untuk ADALINE. J dihitung sesudah update. Semua pelatihan masih finite. Parameter dan loss raw yang sangat besar menunjukkan divergensi, bukan konvergensi.", "small")
r.add("5.2 Bobot dan bias akhir", "h2")
r.table(["ID", "w radius", "w texture", "w smooth", "w concavity", "bias"], [[q.config]+[number(v) for v in q[2:]] for q in W.itertuples()], widths=[83]+[(r.usable-83)/5]*5, size=7.5)
r.add("Urutan bobot sama dengan FEATURES. Baris S berada di koordinat standar, R pada skala asli. Angka dibulatkan untuk tampilan. CSV menyimpan presisi lebih tinggi. Besarnya bobot antar-skala tidak dapat dibandingkan langsung sebagai ukuran kepentingan fitur.", "small")
r.add("5.3 Hasil klasifikasi dan objektif", "h2")
r.add(esc(AN["epochs"]))
r.add("ADALINE standar η=0,001 memberi 107 prediksi uji benar dari 114, η=0,01 memberi 105, dan η=0,1 memberi 103. Hasil ini merupakan pelaporan konfigurasi tetap, bukan pemilihan model terbaik untuk aplikasi. Baseline selalu memprediksi B memberi akurasi 72/114=63,16% pada test. Kontrol raw menunjukkan bahwa tinggi atau rendahnya akurasi perlu dibaca bersama dinamika loss.")

r.start("6. Kurva dan perkembangan epoch", "Isi 6 / 8")
r.image("learning_scaled", height=155)
r.add("Gambar 2. Kurva Perceptron memakai update selama epoch, ADALINE memakai J sesudah batch pada sumbu log. Tiga kurva Perceptron bertumpuk karena penskalaan positif bobot.", "small")
rows=[]
for cid in M.id:
    sub=C[C.config==cid].set_index("epoch")
    vals=[]
    for ep in [10,30,50]:
        q=sub.loc[ep]
        vals += [str(int(q['updates'])) if cid.startswith("P") else (f"{q['J']:.3e}" if q['J']>=1e6 else f"{q['J']:.4f}"),str(int(q.train_errors))]
    rows.append([cid]+vals)
r.table(["ID", "E10<br/>U / J", "E10<br/>Salah", "E30<br/>U / J", "E30<br/>Salah", "E50<br/>U / J", "E50<br/>Salah"],rows,widths=[80]+[(r.usable-80)/6]*6,size=7.3)
r.add("U=jumlah update Perceptron. Salah=kesalahan seluruh train memakai bobot akhir epoch. Checkpoint berasal dari satu riwayat, bukan pelatihan terpisah.", "small")
r.image("scaling_comparison", height=155)
r.add("Gambar 3. Kontrol skala pada η=0,01. Sumbu J logaritmik mempertahankan keterbacaan loss raw yang membesar hingga 6,659×10⁶⁷. Semua 50 epoch tetap ditampilkan.", "small")
r.add(esc(AN["stability"]), "small")

r.start("7. Confusion matrix dan kelas positif", "Isi 7 / 8")
r.image("confusion_all", height=253)
r.add("Gambar 4. Delapan matriks test. Baris=aktual dan kolom=prediksi, keduanya [-1,+1]. Susunan [[TN,FP],[FN,TP]]. B=-1, M=+1. Angka di setiap matriks berjumlah 114.", "small")
r.table(["ID", "Precision M", "Recall M", "F1 M", "Balanced acc."], [[q.id]+[f"{v:.4f}" for v in [q.test_precision,q.test_recall,q.test_f1,q.test_balanced_accuracy]] for q in M.itertuples()],widths=[89,105,99,99,r.usable-392],size=8)
r.add(esc(AN["positive"]))
r.add("Pada Perceptron standar: [[69,3],[9,33]], akurasi=(69+33)/114=89,47%. Pada ADALINE standar η=0,01: [[69,3],[6,36]], akurasi=92,11%. Precision keduanya mirip, tetapi jumlah sampel M yang terlewat berbeda. Penamaan kelas sudah jelas dari diagnosis asli, sehingga FN dapat dijelaskan secara kontekstual.")
r.add("Recall raw ADALINE sebesar 1,0 tidak menandakan model berguna: model akhir memprediksi semua test sebagai M. Akurasinya 42/114=36,84% dan FP=72. Metrik tunggal, termasuk recall, tidak cukup untuk menilai performa.", "small")

r.start("8. Pembahasan dan kesimpulan", "Isi 8 / 8")
r.add("8.1 Indikasi linear dan perbedaan error", "h2")
r.add(esc(AN["linear"]))
r.add(esc(AN["error"]))
r.add("8.2 Learning rate dan standardisasi", "h2")
r.add(esc(AN["eta"]))
r.add(esc(AN["scale"]))
r.image("boundary_slice", height=132)
r.add("Gambar 5. Irisan model 4D pada smoothness_mean dan concavity_mean standar=0. Garis z=0 memperlihatkan pengaruh radius/texture pada dua model fokus η=0,01. Titik proyeksi tidak dianggap sebagai titik pada irisan ini.", "small")
r.add("8.3 Keterbatasan dan pengembangan", "h2")
r.add(esc(AN["limits"]), "small")
r.add(esc(AN["development"]), "small")
r.add("8.4 Kesimpulan", "h2")
r.add(esc(AN["conclusion"]))

r.start("Daftar pustaka", "Referensi")
for i, ref in enumerate(REFERENCES,1):
    r.add(f"[{i}] {esc(ref)}")
    r.gap(5)
r.add("Tanggal akses dan penggunaan sumber", "h2")
r.add("Semua sumber daring diakses pada 12 Oktober 2026. Artikel 1960 digunakan sebagai sumber fundamental asli metode, bukan artikel tentang akurasi kasus terkini. Dokumentasi teknis tidak menggantikan kewajiban rujukan ilmiah. Hasil empiris laporan dihitung dari CSV yang disertakan.")
r.add("Bantuan AI dan verifikasi", "h2")
r.add("ChatGPT membantu menyusun draf kode, narasi, perhitungan manual, laporan, dan presentasi. Verifikasi meliputi update aljabar, gradient finite difference, ambang z=0, cabang non-finite pada contoh ekstrem buatan, scaler train-only, split tanpa overlap, rekonstruksi metrik/bobot dan dua langkah manual aktual. Kelompok tetap bertanggung jawab menjalankan ulang serta memahami kode dan mencatat kontribusi nyata.")

r.start("Lampiran A. Dua update Perceptron", "Lampiran A")
r.add("Konfigurasi P-S-001, η=0,01", "h2")
r.add("Urutan fitur: radius_mean, texture_mean, smoothness_mean, concavity_mean. Row_id dimulai dari 0 pada dataset.csv. Sampel posisi train 0 berlabel +1 dan diprediksi +1 saat z=0, sehingga tidak memperbarui parameter. Dua pembaruan pertama terjadi pada posisi 1 dan 2. Angka di bawah dibulatkan enam desimal, komputasi memakai presisi penuh.")
for i,t in enumerate(MAN["perceptron"],1):
    r.add(f"Langkah {i}: row_id={t['dataset_row_id']}, posisi train={t['position']}", "h2")
    r.add(f"x={vec(t['x'])}<br/>w lama={vec(t['w_before'])}; b lama={t['b_before']:.6f}<br/>y={t['y']}")
    r.add(f"z=wᵀx+b={t['z']:.9f} ≥ 0, sehingga ŷ=+1<br/>Δ=0,01(-1-1)=-0,02", "formula")
    r.table(["Komponen", "w lama", "Δx", "w baru"], [[f"w{j+1}",f"{t['w_before'][j]:.6f}",f"{t['delta']*t['x'][j]:.6f}",f"{t['w_after'][j]:.6f}"] for j in range(4)],size=8)
    r.add(f"b baru={t['b_before']:.6f}+({t['delta']:.6f})={t['b_after']:.6f}")
r.add("Langkah kedua memakai bobot baru langkah pertama. Kedua contoh berasal dari urutan latihan aktual. Tidak ada label atau bobot ilustratif yang berbeda dari eksperimen. File results/manual.json menyimpan semua komponen presisi penuh.", "small")

r.start("Lampiran B. Dua batch ADALINE", "Lampiran B")
r.add("Konfigurasi A-S-001, η=0,01, n=455", "h2")
r.add("Batch memakai seluruh data train terstandardisasi, urutan fitur sama dengan Lampiran A. Error e=y-(Xw+b), J awal=0,5. Seluruh gradient satu batch dihitung dari parameter sebelum update.")
for t in MAN["adaline"]:
    r.add(f"Iterasi batch {t['epoch']}", "h2")
    r.add(f"w lama={vec(t['w_before'])}<br/>b lama={t['b_before']:.9f}")
    r.table(["Fitur", "Σxᵢeᵢ", "Σxᵢeᵢ/n", "w baru"], [[FEATURE_INFO[j][0],f"{t['sum_Xe'][j]:.6f}",f"{t['direction_w'][j]:.6f}",f"{t['w_after'][j]:.6f}"] for j in range(4)], widths=[136,124,124,r.usable-384],size=7.8)
    r.add(f"w baru=w lama+0,01(Xᵀe/455)<br/>mean(e)={t['mean_e']:.9f}<br/>b baru={t['b_before']:.9f}+0,01({t['mean_e']:.9f})={t['b_after']:.9f}<br/>J sesudah update={t['J_after']:.9f}", "formula")
t=MAN["adaline"][1]
r.add(f"Contoh radius pada batch 2: {t['w_before'][0]:.9f}+0,01({t['direction_w'][0]:.9f})={t['w_after'][0]:.9f}. Gradient tidak memakai label threshold. Selisih pembulatan akhir dapat muncul jika memakai angka tabel untuk hitung ulang.", "small")
r.save()
print("Laporan selesai:", r.page, "halaman (8 isi utama).")
