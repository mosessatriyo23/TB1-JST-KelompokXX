"""Audit kelengkapan artefak, konsistensi dan tautan internal sebelum ZIP."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from pypdf import PdfReader
from verify import verify

def check(base):
    base=Path(base)
    verify(base)
    required=['TB1_KelompokXX.ipynb','TB1_KelompokXX.pdf','dataset.csv','README.md','requirements.txt','Presentasi_TB1.pdf','Presentasi_TB1.pptx']
    for name in required:
        assert (base/name).is_file(), name
    readme=(base/'README.md').read_text()
    links=re.findall(r'\]\(([^)]+)\)',readme)
    missing=[x for x in links if not x.startswith(('http:','https:','#')) and not (base/x).exists()]
    assert not missing,missing
    assert not any(ord(c)<32 and c not in '\n\r\t' for c in readme)
    nb=json.loads((base/'TB1_KelompokXX.ipynb').read_text())
    assert nb['nbformat']==4 and nb['nbformat_minor']==5
    ids=[c['id'] for c in nb['cells']]
    assert len(ids)==len(set(ids)) and all(re.fullmatch('[A-Za-z0-9_-]{1,64}',x) for x in ids)
    code=[c for c in nb['cells'] if c['cell_type']=='code']
    assert [c['execution_count'] for c in code]==list(range(1,len(code)+1))
    assert all(not any(o['output_type']=='error' for o in c['outputs']) for c in code)
    assert sum('image/png' in o.get('data',{}) for c in code for o in c['outputs'])==4
    report=PdfReader(base/'TB1_KelompokXX.pdf')
    slides=PdfReader(base/'Presentasi_TB1.pdf')
    assert len(report.pages)==12 and len(slides.pages)==13
    for i in range(8):
        assert f'Isi {i+1} / 8' in report.pages[i+1].extract_text()
    reporttext='\n'.join(p.extract_text() for p in report.pages)
    slidestext='\n'.join(p.extract_text() for p in slides.pages)
    all_text=readme+reporttext+slidestext+json.dumps(nb)
    for word in ['Banknote','raw_banknote','C55P57','skewness','curtosis']:
        assert word.lower() not in all_text.lower(),word
    m=pd.read_csv(base/'results/metrics.csv')
    for row in m.itertuples():
        assert f'{100*row.test_accuracy:.2f}%' in reporttext
        assert f'{100*row.test_accuracy:.2f}%' in slidestext
    with zipfile.ZipFile(base/'Presentasi_TB1.pptx') as z:
        parts=z.namelist()
        slideparts=[n for n in parts if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
        chartparts=[n for n in parts if re.fullmatch(r'ppt/slides/charts/chart\d+\.xml',n)]
        assert len(slideparts)==13 and len(chartparts)==6
        ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        table_text=[x.text for x in ET.fromstring(z.read('ppt/slides/slide6.xml')).findall('.//a:t',ns)]
        for row in m.itertuples():
            assert row.id in table_text and f'{100*row.test_accuracy:.2f}%' in table_text
    info=json.loads((base/'data/provenance.json').read_text())
    assert hashlib.sha256((base/'dataset.csv').read_bytes()).hexdigest()==info['dataset_sha256']
    assert hashlib.sha256((base/'data/raw_breast_cancer.csv').read_bytes()).hexdigest()==info['raw_sha256']
    result={'passed':True,'required_files':required,'report_pages':12,'main_report_pages':8,'slide_count':13,'notebook_cells':len(nb['cells']),'executed_code_cells':len(code),'native_chart_count':6,'missing_readme_links':missing,'legacy_dataset_references':False,'checks':['keberadaan berkas wajib','sel notebook berurutan dan output nyata','tautan README','jumlah/label halaman isi','angka akurasi CSV/PDF/PPTX','referensi dataset lama tidak tertinggal','hash CSV sumber/eksperimen']}
    (base/'results/bundle_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]),indent=2))
