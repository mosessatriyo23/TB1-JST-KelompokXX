"""Pemeriksaan bermakna: rumus, gradient, penghentian non-finite, dan hasil."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from eksperimen import PerceptronNumPy, AdalineNumPy, prepare_data, metrics, FEATURES, predict_linear

def verify(base):
    # Contoh independen, hasil dihitung secara aljabar.
    X=np.array([[1.,2.],[-1.,1.]])
    y=np.array([-1,1])
    p=PerceptronNumPy(.1,1).fit(X,y)
    np.testing.assert_allclose(p.w_,[-.4,-.2]);np.testing.assert_allclose(p.b_,0.)
    assert p.history_[0]['updates']==2
    a=AdalineNumPy(.1,2).fit(X,y)
    np.testing.assert_allclose(a.manual_trace_[0]['w_after'],[-.1,-.05])
    np.testing.assert_allclose(a.w_,[-.1875,-.0825]);np.testing.assert_allclose(a.b_,.0075)
    # Gradient analitik dibanding finite difference pada bobot NONNOL.
    theta=np.array([.2,-.3,.1]);eps=1e-6;Xa=np.column_stack([X,np.ones(2)])
    f=lambda t:np.mean((y-Xa@t)**2)/2
    numeric=np.array([(f(theta+eps*np.eye(3)[i])-f(theta-eps*np.eye(3)[i]))/(2*eps) for i in range(3)])
    analytic=Xa.T@(Xa@theta-y)/2
    np.testing.assert_allclose(analytic,numeric,atol=1e-9)
    bad=AdalineNumPy(.1,50).fit(np.array([[1e200],[1e200]]),np.array([1,1]))
    assert bad.stop_epoch_==1 and len(bad.history_)==0
    df,tr,te,arr,yt,ye,sc=prepare_data(base)
    np.testing.assert_array_equal(predict_linear(np.zeros((1,2)),np.zeros(2),0),[1])
    np.testing.assert_allclose(arr[True][0].mean(axis=0),0,atol=1e-12)
    np.testing.assert_allclose(sc.mean_,df.iloc[tr][FEATURES].mean().to_numpy())
    np.testing.assert_allclose(sc.scale_,df.iloc[tr][FEATURES].std(ddof=0).to_numpy())
    m=pd.read_csv(base/'results/metrics.csv');h=pd.read_csv(base/'results/history.csv')
    w=pd.read_csv(base/'results/weights.csv').set_index('config')
    preds=pd.read_csv(base/'results/predictions_test.csv')
    assert len(m)==8 and len(h)==400 and len(preds)==8*len(te)
    assert set(h.groupby('config').size())=={50}
    checkpoints=pd.read_csv(base/'results/checkpoints.csv')
    assert len(checkpoints)==24 and set(checkpoints.epoch)=={10,30,50}
    for row in m.to_dict('records'):
        p=preds.loc[preds.config==row['id']]
        met=metrics(p.actual,p.prediction)
        for key in ['accuracy','precision','recall','f1']:
            np.testing.assert_allclose(row['test_'+key],met[key])
        assert p.row_id.tolist()==te.tolist()
        assert row['tn']+row['fp']+row['fn']+row['tp']==len(te)
        assert int(row['completed_epochs'])==50
        Xtr,Xte=arr[bool(row['standardized'])]
        wr=w.loc[row['id']]
        coef=wr[FEATURES].to_numpy(float);bias=float(wr.bias)
        np.testing.assert_array_equal(p.prediction.to_numpy(),predict_linear(Xte,coef,bias))
        train_errors=int(np.count_nonzero(predict_linear(Xtr,coef,bias)!=yt))
        assert train_errors==int(row['train_errors'])
        if row['model']=='ADALINE':
            np.testing.assert_allclose(row['final_MSE'],2*row['final_J'])
            np.testing.assert_allclose(row['final_J'],np.mean((yt-Xtr@coef-bias)**2)/2)
    # Invariansi skala Perceptron untuk inisialisasi nol dan urutan tetap.
    ref=w.loc['P-S-001'].to_numpy(float)/.01
    for cid,eta in [('P-S-0001',.001),('P-S-01',.1)]:
        np.testing.assert_allclose(w.loc[cid].to_numpy(float)/eta,ref,atol=1e-8)
        np.testing.assert_array_equal(h[h.config==cid].updates.to_numpy(),h[h.config=='P-S-001'].updates.to_numpy())
    manual=json.loads((base/'results/manual.json').read_text())
    eta=float(manual['eta'])
    for step in manual['perceptron']:
        xi=np.array(step['x']);wb=np.array(step['w_before'])
        z=float(xi@wb+step['b_before']);yp=1 if z>=0 else -1
        delta=eta*(step['y']-yp)
        np.testing.assert_allclose(step['z'],z)
        np.testing.assert_allclose(step['w_after'],wb+delta*xi)
        np.testing.assert_allclose(step['b_after'],step['b_before']+delta)
        np.testing.assert_allclose(xi,arr[True][0][step['position']])
        assert step['dataset_row_id']==int(tr[step['position']])
    for step in manual['adaline']:
        wb=np.array(step['w_before']);bb=float(step['b_before'])
        e=yt-(arr[True][0]@wb+bb)
        np.testing.assert_allclose(step['sum_Xe'],arr[True][0].T@e)
        np.testing.assert_allclose(step['w_after'],wb+eta*(arr[True][0].T@e)/len(yt))
        np.testing.assert_allclose(step['b_after'],bb+eta*e.mean())
    result={'passed':True,'checks':['pembaruan Perceptron','dua batch ADALINE','gradient finite difference','tie z=0 menjadi +1','penghentian non-finite pada data ekstrem buatan','split disjoint tanpa duplikasi','standardisasi mean/std train-only','8 pelatihan x 50 epoch','checkpoint 10/30/50 dari riwayat','metrik dari prediksi uji','rekonstruksi prediksi/kesalahan latih dari bobot','MSE=2J dan J akhir dari parameter','invariansi skala Perceptron','dua update aktual Perceptron konsisten dengan dataset','dua batch aktual ADALINE konsisten dengan seluruh train']}
    (base/'results/verification.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]),indent=2))
