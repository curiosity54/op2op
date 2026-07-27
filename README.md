Machine learning of atomic properties and electronic structure from the external potential

Associated manuscript: https://doi.org/10.1063/5.0332678

Data accompanying the manuscript can be found at <https://doi.org/10.5281/zenodo.21265749>.

## Environment

Python 3.11 is recommended

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install 'cmake<4'
uv pip install e3nn==0.5 metatensor==0.2.0 metatensor-core==0.1.12 metatensor-learn==0.3.1 metatensor-operations==0.3.2 metatensor-torch==0.7.3 pyscfad==0.1.11 torch==2.3.1 featomic==0.6.1 featomic-torch==0.6.1 opt-einsum==3.3.0 scikit-learn==1.5.0
uv pip install --no-binary pyscf pyscf==2.3.0
```

## Running DFT/LDA calculations

Use `pyscf_run.py` to run the corresponding DFT/LDA calculations for water, qm7, and anthracene. Update the `filename` and `basis` set in the script to match the system of interest. The XYZ structures are available in the Zenodo dataset linked above.

## Training models

Training scripts for each system are in:

- `experiments/water`
- `experiments/anthracene`
- `experiments/qm7`

Scripts to verify the degeneracy of matrix elements of the external potential, and compare it to SOAP are also provided.