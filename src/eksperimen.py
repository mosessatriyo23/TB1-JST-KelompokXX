"""Delapan eksperimen TB1 JST. Inti kedua model ditulis dengan NumPy.

Rujukan konsep: Widrow & Hoff (1960), Raschka (2015), dokumentasi mlxtend.
Adaptasi: bias terpisah, batch dibagi n, loss sesudah update, pencatatan
checkpoint dan pemeriksaan non-finite. Tidak memakai estimator siap pakai.
"""
from pathlib import Path
import json
import hashlib
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score

# Empat fitur ditetapkan sebelum evaluasi, tanpa pencarian memakai data uji.
FEATURES = ["radius_mean", "texture_mean", "smoothness_mean", "concavity_mean"]
SEED = 42
EPOCHS = 50
# Daftar ini ditetapkan sebelum pelatihan dan evaluasi data uji.
CONFIGS = [
    {"id": f"{prefix}-S-{str(eta).replace('.', '')}", "model": name, "eta": eta, "standardized": True}
    for name, prefix in [("Perceptron", "P"), ("ADALINE", "A")]
    for eta in [0.001, 0.01, 0.1]
] + [
    {"id": "P-R-001", "model": "Perceptron", "eta": 0.01, "standardized": False},
    {"id": "A-R-001", "model": "ADALINE", "eta": 0.01, "standardized": False},
]


def validate_xy(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError("X harus matriks n x p dan y vektor sepanjang n > 0.")
    if not np.isfinite(X).all() or not np.isin(y, [-1, 1]).all():
        raise ValueError("Fitur harus finite dan label harus -1/+1.")
    return X, y


def predict_linear(X, w, b):
    """Ambang nol, termasuk tie z=0 yang menghasilkan +1."""
    return np.where(np.asarray(X) @ w + b >= 0, 1, -1)


class PerceptronNumPy:
    def __init__(self, eta=0.01, epochs=50):
        if not np.isfinite(eta) or eta <= 0 or epochs < 1:
            raise ValueError("eta harus positif finite dan epochs >= 1.")
        self.eta, self.epochs = eta, epochs

    def fit(self, X, y):
        X, y = validate_xy(X, y)
        self.w_, self.b_ = np.zeros(X.shape[1]), 0.0
        self.history_, self.snapshots_, self.manual_trace_ = [], {}, []
        self.stop_epoch_ = None
        for epoch in range(1, self.epochs + 1):
            updates = 0
            # Urutan hasil split digunakan tetap pada setiap epoch.
            for position, (xi, target) in enumerate(zip(X, y)):
                z = float(xi @ self.w_ + self.b_)
                prediction = 1 if z >= 0 else -1
                delta = self.eta * (target - prediction)
                old_w, old_b = self.w_.copy(), self.b_
                self.w_ = self.w_ + delta * xi
                self.b_ = self.b_ + delta
                if delta != 0:
                    updates += 1
                    if len(self.manual_trace_) < 2:
                        self.manual_trace_.append({
                            "epoch": epoch, "position": position, "x": xi.tolist(),
                            "y": int(target), "z": z, "prediction": prediction,
                            "delta": float(delta), "w_before": old_w.tolist(),
                            "b_before": float(old_b), "w_after": self.w_.tolist(),
                            "b_after": float(self.b_),
                        })
            errors = int(np.count_nonzero(self.predict(X) != y))
            self.history_.append({"epoch": epoch, "updates": updates,
                                  "train_errors": errors, "train_accuracy": 1-errors/len(y)})
            # Semua epoch disimpan agar pemeriksaan riwayat dapat direproduksi.
            self.snapshots_[epoch] = (self.w_.copy(), float(self.b_))
        return self

    def predict(self, X):
        return predict_linear(X, self.w_, self.b_)


class AdalineNumPy:
    def __init__(self, eta=0.01, epochs=50):
        if not np.isfinite(eta) or eta <= 0 or epochs < 1:
            raise ValueError("eta harus positif finite dan epochs >= 1.")
        self.eta, self.epochs = eta, epochs

    def fit(self, X, y):
        X, y = validate_xy(X, y)
        n = len(y)
        self.w_, self.b_ = np.zeros(X.shape[1]), 0.0
        self.history_, self.snapshots_, self.manual_trace_ = [], {}, []
        self.stop_epoch_ = None
        self.initial_loss_ = 0.5 * float(np.mean(y.astype(float)**2))
        for epoch in range(1, self.epochs + 1):
            # Error LINEAR. Prediksi hasil threshold tidak masuk gradient.
            with np.errstate(over="ignore", invalid="ignore"):
                z = X @ self.w_ + self.b_
                e = y - z
                direction_w, direction_b = X.T @ e / n, float(np.mean(e))
                new_w = self.w_ + self.eta * direction_w
                new_b = self.b_ + self.eta * direction_b
                new_e = y - (X @ new_w + new_b)
                J = 0.5 * float(np.mean(new_e**2))
            # Hentikan bila update/loss menjadi NaN atau inf. Pertahankan state
            # terakhir yang valid; epoch gagal dicatat secara terpisah.
            if not (np.isfinite(new_w).all() and np.isfinite(new_b) and np.isfinite(J)):
                self.stop_epoch_ = epoch
                break
            if epoch <= 2:
                self.manual_trace_.append({
                    "epoch": epoch, "n": n, "w_before": self.w_.tolist(),
                    "b_before": float(self.b_), "sum_Xe": (direction_w*n).tolist(),
                    "mean_e": direction_b, "direction_w": direction_w.tolist(),
                    "w_after": new_w.tolist(), "b_after": float(new_b), "J_after": J,
                })
            self.w_, self.b_ = new_w, float(new_b)
            errors = int(np.count_nonzero(self.predict(X) != y))
            self.history_.append({"epoch": epoch, "J": J, "MSE": 2*J,
                                  "train_errors": errors, "train_accuracy": 1-errors/n})
            self.snapshots_[epoch] = (self.w_.copy(), self.b_)
        return self

    def predict(self, X):
        return predict_linear(X, self.w_, self.b_)


def prepare_data(base):
    base = Path(base)
    raw = pd.read_csv(base / "data/raw_breast_cancer.csv")
    expected = raw[FEATURES + ["diagnosis"]].copy()
    expected["label"] = expected["diagnosis"].map({"B": -1, "M": 1})
    df = pd.read_csv(base / "dataset.csv")
    pd.testing.assert_frame_equal(df, expected)
    assert not df.isna().any().any() and not df.duplicated().any()
    assert len(df) == 569 and set(df["diagnosis"]) == {"B", "M"}
    assert df.groupby(FEATURES)["label"].nunique().max() == 1
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=df["label"].to_numpy())
    X, y = df[FEATURES].to_numpy(float), df["label"].to_numpy(int)
    scaler = StandardScaler().fit(X[train_idx])
    arrays = {
        True: (scaler.transform(X[train_idx]), scaler.transform(X[test_idx])),
        False: (X[train_idx].copy(), X[test_idx].copy()),
    }
    assert set(train_idx).isdisjoint(test_idx)
    assert set(map(tuple, X[train_idx])).isdisjoint(set(map(tuple, X[test_idx])))
    return df, train_idx, test_idx, arrays, y[train_idx], y[test_idx], scaler


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[-1, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def run_all(base, make_figures=True):
    base = Path(base)
    for folder in ["results", "figures"]:
        (base/folder).mkdir(exist_ok=True)
    df, train_idx, test_idx, arrays, y_train, y_test, scaler = prepare_data(base)
    # Tahap 1: latih kedelapan konfigurasi TANPA evaluasi data uji.
    models = {}
    for cfg in CONFIGS:
        cls = PerceptronNumPy if cfg["model"] == "Perceptron" else AdalineNumPy
        models[cfg["id"]] = cls(cfg["eta"], EPOCHS).fit(arrays[cfg["standardized"]][0], y_train)
    # Tahap 2: satu pelaporan uji akhir untuk tiap konfigurasi yang telah dikunci.
    rows, history, checkpoints, weights, predictions = [], [], [], [], []
    for cfg in CONFIGS:
        model = models[cfg["id"]]
        Xtr, Xte = arrays[cfg["standardized"]]
        mt, me = metrics(y_train, model.predict(Xtr)), metrics(y_test, model.predict(Xte))
        last = model.history_[-1]
        row = dict(cfg, requested_epochs=EPOCHS, completed_epochs=len(model.history_),
                   nonfinite_epoch=model.stop_epoch_, train_accuracy=mt["accuracy"],
                   train_errors=last["train_errors"], test_accuracy=me["accuracy"],
                   test_balanced_accuracy=me["balanced_accuracy"], test_precision=me["precision"],
                   test_recall=me["recall"], test_f1=me["f1"],
                   tn=me["tn"], fp=me["fp"], fn=me["fn"], tp=me["tp"],
                   final_updates=last.get("updates"), final_J=last.get("J"), final_MSE=last.get("MSE"))
        rows.append(row)
        history.extend(dict(config=cfg["id"], **h) for h in model.history_)
        checkpoints.extend(dict(config=cfg["id"], **h) for h in model.history_ if h["epoch"] in [10,30,50])
        wr = dict(config=cfg["id"], **dict(zip(FEATURES,model.w_.tolist())), bias=model.b_)
        weights.append(wr)
        for idx, actual, pred, z in zip(test_idx, y_test, model.predict(Xte), Xte@model.w_+model.b_):
            predictions.append({"config":cfg["id"], "row_id":int(idx), "actual":int(actual), "prediction":int(pred), "z":float(z)})
    tables = {"metrics":pd.DataFrame(rows),"history":pd.DataFrame(history),
              "checkpoints":pd.DataFrame(checkpoints),"weights":pd.DataFrame(weights),
              "predictions_test":pd.DataFrame(predictions)}
    for name, table in tables.items():
        table.to_csv(base/f"results/{name}.csv",index=False)
    split = pd.DataFrame({"row_id":np.r_[train_idx,test_idx],
                          "split":["train"]*len(train_idx)+["test"]*len(test_idx),
                          "order":np.r_[np.arange(len(train_idx)),np.arange(len(test_idx))]})
    split.to_csv(base/"results/split.csv", index=False)
    pd.DataFrame({"feature":FEATURES,"train_mean":scaler.mean_,"train_std_ddof0":scaler.scale_}).to_csv(base/"results/scaler.csv",index=False)
    stability = {}
    for scaled in [True,False]:
        Xtr = arrays[scaled][0]
        augmented = np.column_stack([Xtr,np.ones(len(Xtr))])
        eigmax = float(np.linalg.eigvalsh(augmented.T@augmented/len(Xtr)).max())
        stability[str(scaled)] = {"lambda_max":eigmax,"eta_upper_bound":2/eigmax}
    p = models["P-S-001"]
    manual = {"eta":0.01,"feature_order":FEATURES,"perceptron":p.manual_trace_,
              "adaline":models["A-S-001"].manual_trace_}
    for trace in manual["perceptron"]:
        trace["dataset_row_id"] = int(train_idx[trace["position"]])
    (base/"results/manual.json").write_text(json.dumps(manual,indent=2),encoding="utf-8")
    summary = {"seed":SEED,"epochs":EPOCHS,"train_n":len(train_idx),"test_n":len(test_idx),
               "train_counts":{str(k):int(v) for k,v in zip(*np.unique(y_train,return_counts=True))},
               "test_counts":{str(k):int(v) for k,v in zip(*np.unique(y_test,return_counts=True))},
               "stability":stability,"python":platform.python_version(),
               "dataset_sha256":hashlib.sha256((base/"dataset.csv").read_bytes()).hexdigest(),
               "features":FEATURES,"label_map":{"B":-1,"M":1},
               "raw_sha256":hashlib.sha256((base/"data/raw_breast_cancer.csv").read_bytes()).hexdigest(),
               "selection":"Tidak ada seleksi model/parameter. Empat fitur dan fokus eta=0.01 ditetapkan sebelum evaluasi uji."}
    (base/"results/summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if make_figures:
        plot_results(base,df,train_idx,test_idx,arrays,y_train,y_test,models,tables)
    return tables, models, summary


def plot_results(base,df,train_idx,test_idx,arrays,y_train,y_test,models,tables):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "figure.dpi":120,"savefig.dpi":180})
    colors = ["#117C83", "#E09037", "#6159A5"]
    def save(fig,name):
        fig.savefig(base/f"figures/{name}.png",bbox_inches="tight",facecolor="white")
        plt.close(fig)
    # Hanya data latih digunakan untuk eksplorasi pola.
    fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    for ax,(a,b) in zip(axs,[(0,1),(0,2)]):
        for label,col in [(-1,colors[0]),(1,colors[1])]:
            part=df.iloc[train_idx].loc[lambda t:t.label==label]
            ax.scatter(part[FEATURES[a]],part[FEATURES[b]],s=11,alpha=.55,c=col,label=f"kelas {label:+d}")
        ax.set(xlabel=FEATURES[a],ylabel=FEATURES[b]);ax.legend(fontsize=9)
    fig.suptitle("Proyeksi data latih: dua pasangan dari empat fitur")
    fig.tight_layout();save(fig,"data_train")
    fig,axs=plt.subplots(1,2,figsize=(10,3.4))
    for eta,col in zip([.001,.01,.1],colors):
        key=str(eta).replace('.','')
        ph=pd.DataFrame(models[f"P-S-{key}"].history_)
        ah=pd.DataFrame(models[f"A-S-{key}"].history_)
        axs[0].plot(ph.epoch,ph.updates,color=col,label=f"eta={eta}",alpha=.8,linestyle={.001:'-',.01:'--',.1:':'}[eta])
        axs[1].plot(ah.epoch,ah.J,color=col,label=f"eta={eta}")
    axs[0].set(title="Perceptron",xlabel="Epoch",ylabel="Jumlah pembaruan salah klasifikasi")
    axs[1].set(title="ADALINE",xlabel="Epoch",ylabel="J setelah pembaruan batch")
    axs[1].set_yscale("log")
    for ax in axs: ax.legend();ax.grid(alpha=.15)
    fig.tight_layout();save(fig,"learning_scaled")
    fig,axs=plt.subplots(1,2,figsize=(10,3.4))
    for tag,label,col in [("S","Standardisasi",colors[0]),("R","Tanpa standardisasi",colors[1])]:
        ph=pd.DataFrame(models[f"P-{tag}-001"].history_)
        ah=pd.DataFrame(models[f"A-{tag}-001"].history_)
        axs[0].plot(ph.epoch,ph.updates,label=label,color=col)
        axs[1].plot(ah.epoch,ah.J,label=label,color=col)
    axs[0].set(title="Perceptron, eta=0.01",xlabel="Epoch",ylabel="Jumlah pembaruan")
    axs[1].set(title="ADALINE, eta=0.01",xlabel="Epoch",ylabel="J setelah pembaruan batch")
    axs[1].set_yscale("log")
    for ax in axs: ax.legend();ax.grid(alpha=.15)
    fig.tight_layout();save(fig,"scaling_comparison")
    fig,axs=plt.subplots(2,4,figsize=(11,5.5))
    for ax,row in zip(axs.flat,tables['metrics'].to_dict('records')):
        cm=np.array([[row['tn'],row['fp']],[row['fn'],row['tp']]])
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(np.bincount((y_test+1)//2)))
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=12,color='white' if cm[i,j]>max(np.bincount((y_test+1)//2))/2 else '#163044')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['-1','+1'],yticklabels=['-1','+1'],title=row['id'],xlabel='Prediksi',ylabel='Aktual')
    fig.tight_layout();save(fig,"confusion_all")
    # Irisan hiperbidang, bukan model dua fitur. Fitur lain dipasang pada
    # mean data latih (0 dalam koordinat standardisasi); tanpa titik proyeksi
    # yang dapat disalahartikan sebagai prediksi irisan.
    fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    xx,yy=np.meshgrid(np.linspace(-3,3,180),np.linspace(-3,3,180))
    grid=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size),np.zeros(xx.size)])
    for ax,cid in zip(axs,["P-S-001","A-S-001"]):
        model=models[cid];zz=(grid@model.w_+model.b_).reshape(xx.shape)
        ax.contourf(xx,yy,np.where(zz>=0,1,-1),levels=[-2,0,2],colors=["#D3EEEA","#FCE6CA"])
        contour=ax.contour(xx,yy,zz,levels=[0],colors=['#173B4C'],linewidths=1.5)
        ax.clabel(contour,fmt={0:'z=0'});ax.set(title=cid,xlabel='radius_mean (standar)',ylabel='texture_mean (standar)')
    fig.suptitle("Irisan 4D: smoothness_mean=0 dan concavity_mean=0 (standar)")
    fig.tight_layout();save(fig,"boundary_slice")


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    result, _, _ = run_all(base)
    print(result["metrics"].to_string(index=False))
