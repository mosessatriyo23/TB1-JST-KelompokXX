"""Eksekusi cell Python tanpa kernel socket atau dependensi notebook.

Setiap cell benar-benar di-exec. stdout, DataFrame HTML, dan gambar lokal
direkam sebagai output notebook v4. Run All pada Jupyter tetap disarankan.
"""
from pathlib import Path
import base64
import contextlib
import io
import json
import os

def execute_notebook(nb, base):
    base = Path(base).resolve()
    old_cwd = Path.cwd()
    os.chdir(base)
    ns = {"__name__": "__notebook__"}
    counter = 0
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            counter += 1
            outputs = []
            def display(*objects):
                for obj in objects:
                    if hasattr(obj, "to_html"):
                        data = {"text/plain": obj.to_string(), "text/html": obj.to_html(index=True)}
                    elif isinstance(obj, dict) and "image/png" in obj:
                        data = obj
                    else:
                        data = {"text/plain": str(obj)}
                    outputs.append({"output_type": "display_data", "data": data, "metadata": {}})
            ns.update(display=display)
            stdout, stderr = io.StringIO(), io.StringIO()
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(source, f"<notebook-cell-{counter}>", "exec"), ns)
            if stdout.getvalue():
                outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": stdout.getvalue()})
            if stderr.getvalue():
                outputs.append({"output_type": "stream", "name": "stderr", "text": stderr.getvalue()})
            cell.update(execution_count=counter, outputs=outputs)
        nb["metadata"]["execution_method"] = "All code cells executed sequentially with Python exec; stdout, DataFrame HTML and PNG displays captured. Not a socket-based Jupyter kernel test."
        return nb
    finally:
        os.chdir(old_cwd)

if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    path = base / 'TB1_KelompokXX.ipynb'
    nb = json.loads(path.read_text(encoding="utf-8"))
    execute_notebook(nb,base)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print('Seluruh sel berhasil dijalankan:',path.name)
