"""Bangun notebook v4, tampilkan inti model, lalu eksekusi semua cell."""
from pathlib import Path
import ast
import hashlib
import json
import sys
from content import TITLE, FEATURE_INFO, REFERENCES, analysis
from execute_inprocess import execute_notebook

BASE = Path(__file__).resolve().parents[1]
source = (BASE / "src/eksperimen.py").read_text()
tree = ast.parse(source)
lines = source.splitlines()
def node_source(name):
    n = next(x for x in tree.body if getattr(x, "name", None) == name)
    return "\n".join(lines[n.lineno-1:n.end_lineno])
first_def = next(x.lineno for x in tree.body if isinstance(x, ast.FunctionDef))
setup = "\n".join(lines[:first_def-1])
cells = []
def cell(kind, s):
    value = {"cell_type": kind, "id": hashlib.sha256((str(len(cells))+s).encode()).hexdigest()[:12], "metadata": {}, "source": s}
    if kind == "code": value.update(execution_count=None, outputs=[])
    cells.append(value)
def md(s): cell("markdown", s)
def code(s): cell("code", s)
md(f"""# TB1 JST: {TITLE}
## Perceptron dan ADALINE menggunakan NumPy

Kelompok **XX**. Isi identitas final 2-3 anggota sebelum pengumpulan. Notebook ini memakai data lokal yang benar-benar diunggah pengguna, bukan tautan unduhan.

**Kasus:** klasifikasi diagnosis sampel jinak (`B=-1`) dan ganas (`M=+1`, kelas positif) dari karakteristik inti sel. Eksperimen memakai empat fitur yang ditetapkan sebelum evaluasi: `radius_mean`, `texture_mean`, `smoothness_mean`, `concavity_mean`.

**Kaitan Tugas 1:** kasus sebelumnya adalah segmentasi pelanggan Mall Customers tanpa label dengan rencana SOM. Penggantian ke klasifikasi dua kelas mengubah kasus dan dataset. Pengguna telah mengonfirmasi bahwa penggantian dikonsultasikan/diperbolehkan setelah penjelasan ketentuan C.1. Konfirmasi ini tidak diperlakukan sebagai surat persetujuan yang dibuat oleh AI.

Ini latihan benchmark akademik. Model dan keluaran `z` tidak digunakan sebagai alat diagnosis klinis atau probabilitas keganasan.""")
md("""## 1. Data, izin, dan fitur

[Halaman Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) adalah sumber unduhan pengguna. Metadata Kaggle menyebut CC BY-NC-SA 4.0. [Sumber asli UCI](https://doi.org/10.24432/C5DW2B) menyebut CC BY 4.0. Paket mempertahankan atribusi dan secara konservatif mengikuti ketentuan Kaggle untuk CSV sumber dan turunannya. Lihat `DATA_LICENSE.md`.

Audit data.csv: 569 observasi, 33 kolom, diagnosis B=357 dan M=212. Terdapat 30 pengukuran numerik, satu ID, satu target, dan `Unnamed: 32` yang kosong seluruhnya. Tidak ada nilai kosong pada 30 pengukuran atau diagnosis, tidak ada duplikasi baris maupun duplikasi 30 fitur. Kolom ID dan kolom kosong tidak menjadi fitur. Semua baris dipertahankan, tanpa imputasi/oversampling/penghapusan outlier.

`dataset.csv` berisi empat fitur pada skala asli, `diagnosis`, dan `label`. Dua kolom target tidak masuk X. Pemilihan empat fitur menggambarkan ukuran, tekstur dan bentuk. Tidak sekaligus memakai radius/perimeter/area yang semuanya terkait ukuran. Subset ini tidak diklaim optimal atau dipilih dari test.

""" + "\n".join(f"- `{name}`: {meaning}" for name, meaning in FEATURE_INFO))
md("""## 2. Rujukan ilmiah dan adaptasi kode

**Widrow & Hoff (1960), Adaptive switching circuits:** artikel fundamental pembelajaran adaptif dengan error kuadrat. Batch rerata mengikuti rumus rubrik. Artikel lama ini dipakai sebagai sumber asli metode, bukan bukti kinerja klinis terkini.

**Raschka (2015) dan dokumentasi mlxtend:** referensi struktur loop Perceptron serta pembedaan linear output dan threshold pada ADALINE. Struktur konseptual disusun ulang dengan bias terpisah, label -1/+1, zero initialization, urutan tetap, normalisasi gradient `/n`, loss J setelah update, checkpoint dan penjagaan non-finite. Contoh estimator siap pakai dan standardisasi sebelum split tidak diikuti.

**scikit-learn:** hanya untuk split, scaler dan metrik. Implementasi fit kedua model memakai NumPy, bukan model siap pakai. Semua URL lengkap tersedia pada daftar pustaka di bagian akhir.""")
md("## 3. Rancangan terkunci sebelum evaluasi uji")
code(setup + '''
BASE = Path.cwd()
assert (BASE / "dataset.csv").is_file(), "Buka notebook dari direktori utama proyek."
display(pd.DataFrame(CONFIGS))
print("Python:", platform.python_version(), "seed:", SEED)
''')
md(r"""## 4. Neuron dan implementasi Perceptron

$z=w^Tx+b$; $\hat y=+1$ jika $z\ge0$, selain itu $-1$. Empat bobot mengatur kontribusi fitur. Bias menggeser hiperbidang. Tidak ada hidden layer.

Untuk tiap sampel, $\Delta=\eta(y-\hat y)$, $w\leftarrow w+\Delta x$, $b\leftarrow b+\Delta$. Dengan label bipolar, salah klasifikasi menghasilkan $\Delta=\pm2\eta$.

**Update selama epoch** dan **kesalahan setelah epoch** dicatat terpisah. Semua model dimulai w=0 dan b=0, urutan train tetap, 50 epoch tanpa early stopping akibat nol kesalahan.""")
code(node_source("validate_xy") + "\n\n" + node_source("predict_linear") + "\n\n" + node_source("PerceptronNumPy"))
md(r"""## 5. ADALINE dengan batch Delta Rule

$$e=y-(Xw+b),\quad J=\frac{1}{2n}\sum_i e_i^2,\quad MSE=2J$$
$$w\leftarrow w+\eta X^Te/n,\quad b\leftarrow b+\eta\operatorname{mean}(e)$$

Semua komponen gradient satu iterasi memakai bobot lama yang sama. Error berasal dari linear output, bukan label hasil threshold. J dicatat **sesudah** pembaruan. Threshold hanya dipakai saat prediksi/metrik. Bila parameter atau loss menjadi inf/NaN, simpan parameter valid terakhir dan catat epoch gagal. Loss yang sangat besar tetapi masih finite tidak dihentikan diam-diam.""")
code(node_source("AdalineNumPy"))
md("## 6. Audit, split 80:20, dan standardisasi train-only")
code(node_source("prepare_data") + "\n\n" + node_source("metrics"))
code('''df, tr, te, arrays, yt, ye, scaler = prepare_data(BASE)
raw = pd.read_csv(BASE / "data/raw_breast_cancer.csv")
print("Raw:", raw.shape, "CSV eksperimen:", df.shape)
print("Missing raw:", raw.isna().sum()[raw.isna().sum()>0].to_dict())
print("Duplikasi baris raw:", raw.duplicated().sum())
display(df.head())
display(pd.crosstab(df.diagnosis, df.label))
print("Train:", len(tr), "Test:", len(te))
display(pd.DataFrame({"feature": FEATURES, "train_mean": scaler.mean_, "train_std_ddof0": scaler.scale_}))
print("Target train:", dict(zip(*np.unique(yt, return_counts=True))))
print("Target test:", dict(zip(*np.unique(ye, return_counts=True))))
''')
md("""`train_test_split(test_size=0.2, random_state=42, stratify=y)` menghasilkan 455 train dan 114 test. Mean/std (ddof=0) berasal hanya dari train. Enam konfigurasi memakai data standar. Dua kontrol memakai skala asli, eta=0.01, 50 epoch. Tidak ada seleksi/tuning lewat test. Fokus pembahasan eta=0.01 ditentukan sebelum evaluasi. Checkpoint 10/30/50 hanya merangkum riwayat latih yang sama.""")
md("## 7. Eksekusi delapan konfigurasi dan ekspor hasil")
code(node_source("run_all") + "\n\n" + node_source("plot_results"))
code('''tables, models, summary = run_all(BASE)
display(tables["metrics"])
print(json.dumps(summary, indent=2))
''')
md("## 8. Dua langkah manual Perceptron")
md("""P-S-001, eta=0.01. Contoh adalah dua pembaruan aktual pertama pada epoch 1, posisi train 1 dan 2 (indeks mulai 0). Sampel pertama pada posisi 0 berlabel +1 dan diprediksi +1 saat z=0, sehingga tidak memperbarui parameter. Urutan fitur mengikuti FEATURES. Lampiran laporan memuat perhitungan setiap komponen.""")
code('''manual = json.loads((BASE / "results/manual.json").read_text())
for i, t in enumerate(manual["perceptron"], 1):
    print(f"\\nLangkah {i}, row_id={t['dataset_row_id']}, posisi train={t['position']}")
    print("x =", np.array(t["x"]))
    print("w lama =", t["w_before"], "b lama =", t["b_before"])
    print(f"z = {t['z']:.9f}; y = {t['y']}; yhat = {t['prediction']}")
    print(f"Delta = {manual['eta']} * ({t['y']} - {t['prediction']}) = {t['delta']}")
    display(pd.DataFrame({"feature": FEATURES, "w_before": t["w_before"],
                          "delta_x": np.array(t["x"])*t["delta"], "w_after": t["w_after"]}))
    print("b baru =", t["b_after"])
''')
md("## 9. Dua iterasi batch manual ADALINE")
md("""A-S-001, eta=0.01, n=455. Tabel memakai seluruh train yang sama dengan eksperimen, bukan batch ilustrasi lain. Pada batch 1: z=0, e=y, J awal=0.5. Batch 2 memakai bobot hasil batch 1. Hasil dan angka presisi penuh disimpan dalam `results/manual.json`.""")
code('''for t in manual["adaline"]:
    print("\\nBatch", t["epoch"], "n=", t["n"])
    print("w lama =", t["w_before"], "b lama =", t["b_before"])
    display(pd.DataFrame({"feature": FEATURES, "sum_Xe": t["sum_Xe"],
                          "XTe_div_n": t["direction_w"], "w_after": t["w_after"]}))
    print("mean(e) =", t["mean_e"], "b baru =", t["b_after"], "J sesudah update =", t["J_after"])
''')
md("## 10. Kurva pembelajaran dan checkpoint latih")
code('''display(tables["checkpoints"])
import base64
from IPython.display import Image as NotebookImage
display(NotebookImage(filename=str(BASE / "figures/learning_scaled.png")))
display(NotebookImage(filename=str(BASE / "figures/scaling_comparison.png")))
'''.replace('from IPython.display import Image as NotebookImage\n','').replace('display(NotebookImage(filename=str(BASE / "figures/learning_scaled.png")))','display_png(BASE / "figures/learning_scaled.png")').replace('display(NotebookImage(filename=str(BASE / "figures/scaling_comparison.png")))','display_png(BASE / "figures/scaling_comparison.png")'))
md("## 11. Confusion matrix dan parameter akhir")
code('''display(tables["metrics"][["id", "tn", "fp", "fn", "tp", "test_precision", "test_recall", "test_f1", "test_balanced_accuracy"]])
display(tables["weights"])
display_png(BASE / "figures/confusion_all.png")
display_png(BASE / "figures/boundary_slice.png")
''')
md("""Confusion matrix memakai baris aktual dan kolom prediksi, keduanya urutan `[-1,+1]`, sehingga `[[TN,FP],[FN,TP]]`. Kelas positif M. Gambar batas keputusan adalah **irisan hiperbidang empat fitur**, dengan smoothness dan concavity ditahan pada rerata train (nol setelah standardisasi), bukan model dua fitur atau proyeksi titik semua sampel. Karena itu titik sampel asli tidak ditempelkan pada irisan seolah semua fitur lainnya nol.""")
md("## 12. Pembahasan berdasarkan hasil aktual\n\n" + "\n\n".join(f"**{label}.** {analysis(BASE)[key]}" for key,label in [
    ("linear","Pemisahan linear"),("error","Perbedaan error"),("eta","Learning rate Perceptron"),
    ("epochs","Epoch dan loss vs akurasi"),("scale","Standardisasi"),("stability","Kestabilan"),
    ("positive","Kelas positif"),("limits","Keterbatasan"),("development","Pengembangan")]))
md("## 13. Verifikasi numerik dan konsistensi")
code('''import sys
sys.path.insert(0, str(BASE / "src"))
from verify import verify
print(json.dumps(verify(BASE), indent=2))
''')
md("## 14. Kesimpulan\n\n" + analysis(BASE)["conclusion"] + "\n\n**Bantuan AI:** ChatGPT membantu draf kode, analisis dan dokumen. Angka berasal dari eksekusi. Setiap anggota perlu Run All, memeriksa penjelasan, dan mencatat kontribusi nyata. Nomor kelompok, anggota final, repository publik dan hash commit masih perlu dilengkapi. Konfirmasi konsultasi penggantian berasal dari pengguna.")
md("## Daftar pustaka\n\n" + "\n\n".join(f"{i}. {s}" for i,s in enumerate(REFERENCES,1)) + "\n\nSumber daring diakses 12 Oktober 2026. Rujukan fundamental asli tidak diperlakukan sebagai bukti klinis terkini.")

# Fungsi PNG berperilaku sama di Jupyter dan runner satu proses. Runner
# menangkap dict MIME, Jupyter menerima raw=True untuk rich output.
png_function = '''
def display_png(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    try:
        display({"image/png": encoded}, raw=True)
    except TypeError:
        display({"image/png": encoded})
'''
# Temukan setup cell secara eksplisit, bukan mengandalkan indeks narasi.
setup_cell = next(c for c in cells if c["cell_type"]=="code")
if "def display_png" not in setup_cell["source"]:
    setup_cell["source"] += "\nimport base64\n" + png_function
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.14"}}}
execute_notebook(nb, BASE)
(BASE / "TB1_KelompokXX.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook dibuat dan seluruh sel kode dieksekusi:", len(cells), "sel")
