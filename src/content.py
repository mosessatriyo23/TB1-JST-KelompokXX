"""Narasi dan referensi bersama agar laporan, README dan notebook konsisten."""
from pathlib import Path
import json
import pandas as pd

TITLE = "Klasifikasi Breast Cancer Wisconsin (Diagnostic)"
FEATURE_INFO = [
    ("radius_mean", "Rerata jarak dari pusat inti sel ke titik pada perimeter. Mewakili ukuran."),
    ("texture_mean", "Rerata karakteristik tekstur, dihitung dari simpangan baku intensitas keabuan."),
    ("smoothness_mean", "Rerata variasi lokal panjang radius. Mewakili kehalusan batas inti sel."),
    ("concavity_mean", "Rerata tingkat cekungan bagian kontur inti sel."),
]
REFERENCES = [
    "UCI Machine Learning. Breast Cancer Wisconsin (Diagnostic) Data Set. Kaggle. https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data. Lisensi pada metadata Kaggle: CC BY-NC-SA 4.0.",
    "Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. Lisensi sumber asli: CC BY 4.0.",
    "Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits. IRE WESCON Convention Record, Part 4, 96-104. https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf. Artikel fundamental asli ADALINE.",
    "Raschka, S. (2015, 24 Maret). Single-Layer Neural Networks and Gradient Descent. https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html. Tutorial dan kode pendukung.",
    "Raschka, S. (tanpa tahun). Adaline: Adaptive Linear Neuron Classifier. Dokumentasi mlxtend. https://rasbt.github.io/mlxtend/user_guide/classifier/Adaline/.",
    "scikit-learn developers. StandardScaler; train_test_split. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html dan https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html. Versi eksekusi 1.8.0.",
    "Panduan internal mata kuliah (2026). Tugas Besar 1: Analisis dan Implementasi Klasifikasi Data Menggunakan Perceptron dan ADALINE. IT6053, FTI UKDW. Sumber ketentuan tugas yang diberikan pengguna.",
]

def load_results(base):
    base = Path(base)
    return (
        pd.read_csv(base / "results/metrics.csv"),
        pd.read_csv(base / "results/weights.csv"),
        pd.read_csv(base / "results/checkpoints.csv"),
        json.loads((base / "results/summary.json").read_text()),
        json.loads((base / "results/manual.json").read_text()),
    )

def number(x):
    return f"{x:.3e}" if abs(x) >= 1e6 else f"{x:.6f}"

def analysis(base):
    m, w, c, s, manual = load_results(base)
    mm = m.set_index("id")
    p, a = mm.loc["P-S-001"], mm.loc["A-S-001"]
    ar, pr = mm.loc["A-R-001"], mm.loc["P-R-001"]
    ap = mm.loc["A-S-01"]
    cps = c[c.config == "P-S-001"].sort_values("epoch")
    return {
        "linear": f"Perceptron standar masih melakukan {int(p.final_updates)} pembaruan pada epoch 50 dan menyisakan {int(p.train_errors)} kesalahan dari {s['train_n']} data latih. Pola proyeksi dan akurasi latih menunjukkan struktur yang cukup cocok untuk keputusan linear, tetapi 50 epoch belum membuktikan separabilitas sempurna maupun ketidakmungkinannya. Irisan/proyeksi dua fitur tidak menentukan sifat ruang empat dimensi.",
        "error": "Perceptron memperbarui parameter dari error label diskret y-yhat. ADALINE memakai error kontinu y-z, termasuk sampel yang kelasnya sudah benar. Karena itu, akurasi yang sama tidak berarti bobot, margin, loss, atau proses pembelajaran sama. Jumlah update Perceptron tidak sebanding langsung dengan J ADALINE.",
        "eta": "Tiga eta Perceptron standar memberi riwayat update dan prediksi yang sama. Dengan w=b=0, urutan tetap, eta positif konstan, dan ambang nol, bobot serta bias proporsional terhadap eta sehingga tanda z tidak berubah. Hasil ini sah dan telah diperiksa, bukan bukti eta selalu tidak berpengaruh pada setiap implementasi.",
        "epochs": f"Kesalahan latih Perceptron standar pada epoch 10/30/50 adalah {'/'.join(str(int(v)) for v in cps.train_errors)}. Kesalahan tidak turun monoton. Pada ADALINE standar, eta 0.1 menghasilkan J akhir {ap.final_J:.6f}, lebih rendah daripada eta 0.01 ({a.final_J:.6f}), tetapi akurasi uji {100*ap.test_accuracy:.2f}% justru lebih rendah daripada {100*a.test_accuracy:.2f}%. Objektif kuadrat dan kesalahan kelas mengukur hal berbeda. Epoch 50 adalah anggaran tetap, bukan epoch terbaik hasil seleksi data uji.",
        "scale": f"Pada eta=0.01, Perceptron standar mencapai akurasi uji {100*p.test_accuracy:.2f}%, sedangkan raw {100*pr.test_accuracy:.2f}%. ADALINE standar mencapai {100*a.test_accuracy:.2f}%, sedangkan raw {100*ar.test_accuracy:.2f}% dengan J={ar.final_J:.3e}. Skala asli radius/texture jauh lebih besar daripada smoothness/concavity, sehingga kontribusi langkah tidak seimbang. ADALINE raw mengalami divergensi loss, bukan sekadar konvergensi lambat.",
        "stability": f"Untuk J kuadratik, H=A.T@A/n dengan A=[X,1]. Rentang stabil gradient descent adalah 0<eta<2/lambda_max(H). Dari train diperoleh batas {s['stability']['True']['eta_upper_bound']:.6f} pada fitur standar dan {s['stability']['False']['eta_upper_bound']:.6f} pada fitur raw. Eta 0.01 melampaui batas raw. J masih finite hingga epoch 50, sehingga pelatihan tidak dihentikan dan epoch non-finite dicatat sebagai tidak terjadi. Tidak ada perubahan eta setelah melihat hasil uji.",
        "positive": f"Kelas M (+1) berjumlah {s['test_counts']['1']} dari {s['test_n']} data uji. Perceptron standar eta=0.01 memiliki FN={int(p.fn)}, FP={int(p.fp)}, recall={p.test_recall:.4f}. ADALINE standar eta=0.01 memiliki FN={int(a.fn)}, FP={int(a.fp)}, recall={a.test_recall:.4f}. FN berarti sampel ganas diprediksi jinak, sehingga akurasi saja tidak cukup. Ini hanya evaluasi benchmark, bukan rekomendasi diagnosis klinis.",
        "limits": f"Satu split dengan {s['test_n']} data uji membatasi generalisasi. Satu perubahan prediksi setara {100/s['test_n']:.2f} poin persentase. Empat dari 30 fitur dipilih secara apriori, sehingga informasi lain tidak digunakan dan subset ini tidak diklaim optimal. Ketergantungan antarfitur dan asal satu benchmark membatasi transfer ke populasi lain. Tidak ada validasi eksternal atau kalibrasi probabilitas. Z bukan probabilitas.",
        "development": "Pengembangan dapat membandingkan semua fitur atau subset lain menggunakan validasi terstratifikasi di dalam train, kemudian memilih eta/epoch tanpa melihat test. Gunakan beberapa seed dan validasi eksternal. Data uji akhir harus tetap terpisah sampai seleksi selesai.",
        "conclusion": f"Implementasi NumPy kedua model, delapan pelatihan 50 epoch, serta pemeriksaan numerik telah selesai. Pada rancangan ini Perceptron standar menghasilkan akurasi uji {100*p.test_accuracy:.2f}% untuk tiga eta, sedangkan ADALINE standar menghasilkan {100*mm.loc['A-S-0001'].test_accuracy:.2f}%, {100*a.test_accuracy:.2f}%, dan {100*ap.test_accuracy:.2f}%. Standardisasi menjaga ADALINE stabil pada eta yang diuji. Penurunan J tidak menjamin kenaikan akurasi kelas. Tidak dilakukan seleksi model berdasarkan data uji.",
    }
