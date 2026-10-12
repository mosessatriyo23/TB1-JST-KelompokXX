"""Transformasi deterministik CSV unggahan, tanpa unduhan dan tanpa data uji."""
from pathlib import Path
import hashlib
import json
import pandas as pd

FEATURES = ["radius_mean", "texture_mean", "smoothness_mean", "concavity_mean"]

def prepare(base):
    base = Path(base)
    raw_path = base / "data/raw_breast_cancer.csv"
    raw = pd.read_csv(raw_path)
    numeric = raw.drop(columns=["id", "diagnosis", "Unnamed: 32"])
    assert raw.shape == (569, 33)
    assert raw["Unnamed: 32"].isna().all()
    assert not numeric.isna().any().any()
    assert not raw["diagnosis"].isna().any()
    assert numeric.shape[1] == 30 and not numeric.duplicated().any()
    assert not raw.duplicated().any() and raw["id"].is_unique
    df = raw[FEATURES + ["diagnosis"]].copy()
    df["label"] = df["diagnosis"].map({"B": -1, "M": 1})
    assert not df.isna().any().any() and not df.duplicated().any()
    df.to_csv(base / "dataset.csv", index=False)
    info = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "File data.csv yang diunggah pengguna dari Kaggle",
        "kaggle_url": "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data",
        "kaggle_license": "CC BY-NC-SA 4.0",
        "kaggle_license_verification": "Kaggle API datasets/list, uciml/breast-cancer-wisconsin-data",
        "original_url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "original_doi": "10.24432/C5DW2B",
        "original_license": "CC BY 4.0 (UCI)",
        "access_date": "2026-10-12",
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256((base / "dataset.csv").read_bytes()).hexdigest(),
        "raw_shape": list(raw.shape),
        "raw_counts": raw["diagnosis"].value_counts().to_dict(),
        "raw_duplicate_rows": int(raw.duplicated().sum()),
        "numeric_duplicate_rows": int(numeric.duplicated().sum()),
        "missing_predictors": int(numeric.isna().sum().sum()),
        "empty_column": "Unnamed: 32, 569 nilai kosong, dihapus",
        "excluded_id": "id adalah pengenal rekaman riset publik, tidak menjadi prediktor",
        "selected_features": FEATURES,
        "selection_reason": "Ukuran, tekstur, kehalusan, dan cekungan inti sel. Dipilih sebelum evaluasi, bukan hasil tuning/test. Tidak sekaligus memakai radius, perimeter, dan area yang merepresentasikan ukuran.",
        "label_map": {"B": -1, "M": 1},
        "positive_class": "M (malignant/ganas)",
        "rows_removed": 0,
        "transforms": "Buang ID dan kolom kosong dari data eksperimen; pilih empat fitur; pertahankan diagnosis dan tambah label bipolar. Tidak ada imputasi, deduplikasi, oversampling atau penghapusan outlier. CSV tetap dalam skala asli.",
        "approval": "Pengguna mengonfirmasi bahwa penggantian telah dikonsultasikan/diperbolehkan setelah pertanyaan tentang C.1. Tidak ada surat persetujuan di dalam paket."
    }
    (base / "data/provenance.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
