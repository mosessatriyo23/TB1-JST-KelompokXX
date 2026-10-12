"""Buat paket GitHub-ready dengan manifest; tidak memasukkan cache/venv."""
from pathlib import Path
import hashlib
import json
import zipfile
from check_bundle import check

BASE=Path(__file__).resolve().parents[1]
check(BASE)
files=[p for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.venv' not in p.parts and not p.name.endswith('.pyc') and p.name!='MANIFEST_SHA256.json']
manifest={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(BASE/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
target=BASE.parent/'TB1-JST-BreastCancer-Paket-Lengkap.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
    for p in files+[BASE/'MANIFEST_SHA256.json']:
        z.write(p,str(Path('TB1-JST-KelompokXX')/p.relative_to(BASE)))
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    for name,digest in manifest.items():
        assert hashlib.sha256(z.read('TB1-JST-KelompokXX/'+name)).hexdigest()==digest
print('Paket selesai:',target.name,'files:',len(files)+1,'bytes:',target.stat().st_size)
