import numpy as np 
from ase.io import read, write
import ase

#----- Create water dimers -----
def move_molecule_wrt_origin(frame, origin=np.zeros(3), R=np.eye(3)):
    coords = frame.positions
    coords = (R @ coords.T).T + origin
    return ase.Atoms(symbols = ["O", "H", "H"], positions =coords)

def create_water_dimers(monomers, distances, npairs = 20, seed = 2414):
    nmonomers = len(monomers)
    rng = np.random.default_rng(seed)
    nmonomers = len(monomers)
    frames = []
    pairs = []
    
    for ipair in range(npairs):
        pair_idx = rng.choice(nmonomers, size=2, replace=False)
        pairs.append(tuple(pair_idx))
    
        for d in distances:
            mols = []
    
            monA = monomers[pair_idx[0]]
            monB = monomers[pair_idx[1]]
            mols.append(move_molecule_wrt_origin(monA, origin=np.array([0.0, 0.0, -d/2])))
            mols.append(move_molecule_wrt_origin(monB, origin=np.array([0.0, 0.0, d/2])))#, R=R))
    
            dimer = mols[0] + mols[1]
            frames.append(dimer)

    return frames, pairs

monomers = read('data/water_1000/water_1000.xyz', ':')
distances = np.linspace(3.5, 12.0, 20)
dimers, pairs = create_water_dimers(monomers, distances, npairs = 50, seed = 2414)
write('data/water_dimers/water_dimers.xyz', dimers)

#----- Compute energy and dipole for each dimer -----
from pyscf import gto, scf
from pyscf.scf import hf
import warnings
import torch
import hickle
def compute_energy_and_dipole(ase_atoms, verbose = 3, basis="sto-3g", compute_vext = False, return_fockdm = False):
    results = {}
    
    atom_spec = []
    for sym, pos in zip(ase_atoms.symbols, ase_atoms.positions):
        atom_spec.append([sym, pos])

    mol = gto.Mole(
        atom=atom_spec,
        basis=basis,
        verbose=0
    )
    mol.build()
    vext = mol.intor('int1e_nuc') if compute_vext else None
            
    mf = scf.RKS(mol)
    mf.max_cycle = 200
    mf.diis = 6
    mf.xc = 'pbe'
    
    E = mf.kernel()
    if not mf.converged:
        warnings.warn('Not converged')
    mu = mf.dip_moment(unit="Debye")
    if return_fockdm:
        fock = mf.get_fock()
        dm = mf.make_rdm1()
    else:
        fock, dm = None, None

    results['vext'] = vext
    results['E'] = E
    results['mu'] = mu
    results['fock'] = fock
    results['dm'] = dm
    return results


BASIS = 'def2-tzvp'
results = []
frames = read('data/water_dimers/water_dimers.xyz',':') 
energies, dipoles  = [], []
for f in frames[:]:
    res = compute_energy_and_dipole(f, basis = BASIS, compute_vext = True, return_fockdm = False)
    results.append(res)
    energies.append(res['E'])
    dipoles.append(res['mu'])

vext = [torch.from_numpy(results[i]['vext']) for i in range(len(results))]
energies = torch.tensor(energies)

hickle.dump(vext, f'data/water_dimers/{BASIS}/vext.hickle')
hickle.dump(energies, f'data/water_dimers/{BASIS}/energy.hickle')
hickle.dump(torch.tensor(dipoles), f'data/water_dimers/{BASIS}/dipole.hickle')
