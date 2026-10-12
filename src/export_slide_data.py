"""Ekspor literal angka untuk authoring slide, bukan dependensi eksperimen."""
from pathlib import Path
import json
import pandas as pd
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("destination", type=Path)
args=parser.parse_args()
base=Path(__file__).resolve().parents[1]
data={}
for name in ["metrics","history","checkpoints","weights"]:
    table=pd.read_csv(base/f"results/{name}.csv")
    data[name]=json.loads(table.to_json(orient="records",double_precision=15))
for name in ["summary","manual"]:
    data[name]=json.loads((base/f"results/{name}.json").read_text())
args.destination.parent.mkdir(parents=True,exist_ok=True)
args.destination.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
