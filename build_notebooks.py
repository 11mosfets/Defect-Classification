import nbformat as nbf
import os

def py_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r') as f:
        content = f.read()
    
    nb = nbf.v4.new_notebook()
    cells = content.split('# %%')
    
    for cell in cells:
        cell = cell.strip()
        if not cell: continue
        if cell.startswith('##'):
            # clean markdown line markers
            md = "\n".join(line.lstrip("##").strip() for line in cell.split("\n"))
            nb.cells.append(nbf.v4.new_markdown_cell(md))
        else:
            nb.cells.append(nbf.v4.new_code_cell(cell))
            
    with open(ipynb_file, 'w') as f:
        nbf.write(nb, f)
    print(f"Created {ipynb_file}")

py_to_ipynb('rf.py', 'rf_pipeline.ipynb')
py_to_ipynb('knn.py', 'knn_pipeline.ipynb')
py_to_ipynb('patchcore.py', 'patchcore_ensemble_pipeline.ipynb')

os.remove('rf.py')
os.remove('knn.py')
os.remove('patchcore.py')
print("Complete. Temporary files removed.")
