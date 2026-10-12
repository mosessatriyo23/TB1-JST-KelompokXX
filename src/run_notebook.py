"""Eksekusi seluruh notebook menggunakan kernel baru dari root repository."""
from pathlib import Path
import nbformat
import sys
from jupyter_client.kernelspec import KernelSpecManager
from ipykernel.kernelspec import install
from nbclient import NotebookClient

base=Path(__file__).resolve().parents[1]
path=base/'TB1_KelompokXX.ipynb'
install(user=True,kernel_name='tb1-jst',display_name='Python (TB1 JST)')
nb=nbformat.read(path,as_version=4)
NotebookClient(nb,timeout=300,kernel_name='tb1-jst',resources={'metadata':{'path':str(base)}}).execute()
nbformat.write(nb,path)
print('Selesai:',path.name)
