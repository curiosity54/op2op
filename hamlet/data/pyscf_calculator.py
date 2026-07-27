from typing import List, Optional, Union
from ase.io import read
import ase
import numpy as np
import pyscf  
import os
from pathlib import Path
import hickle
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import torch
from collections import defaultdict
from ase.data import atomic_numbers
import warnings

import re


def convert_str_to_nlm(x: str):
    """x : string of the form 'nlm' where n is the principal quantum number, l is the azimuthal quantum number and m is the magnetic quantum number
    example: '2px' -> [2,1,1]
    """
    orb_map = {
        "s": [0, 0],
        "px": [1, 1],
        "py": [1, -1],
        "pz": [1, 0],
        "dxy": [2, -2],
        "dyz": [2, -1],
        "dz^2": [2, 0],
        "dxz": [2, 1],
        "dx2-y2": [2, 2],
        "f-3": [3, -3],
        
        "f-2": [3, -2],
        "f-1": [3, -1],
        "f+0": [3, 0],
        "f+1": [3, 1],
        "f+2": [3, 2],
        "f+3": [3, 3],

        "g-4": [4, -4],
        "g-3": [4, -3],
        "g-2": [4, -2],
        "g-1": [4, -1],
        "g+0": [4, 0],
        "g+1": [4, 1],
        "g+2": [4, 2],
        "g+3": [4, 3],
        "g+4": [4, 4],

        "h-5": [5, -5],
        "h-4": [5, -4],
        "h-3": [5, -3],
        "h-2": [5, -2],
        "h-1": [5, -1],
        "h+0": [5, 0],
        "h+1": [5, 1],
        "h+2": [5, 2],
        "h+3": [5, 3],
        "h+4": [5, 4],
        "h+5": [5, 5],
    }
    match = re.match(r"([0-9]+)(.+)", x, re.I)
    try:
        n, lm = match.groups()
    except:
        raise NotImplementedError("Only upto f orbitals supported. Are you sure you are using the correct basis set?")  
    return [int(n)] + orb_map[lm]


def _instantiate_pyscf_mol(frame, basis="sto-3g"):
    mol = pyscf.gto.Mole()
    mol.atom = pyscf_ase.ase_atoms_to_pyscf(frame)
    mol.basis = basis
    if mol.nelectron % 2 != 0:
        mol.spin = 1
    mol.build()
    return mol

def _instantiate_pyscf_mf(calculator, frames, matrices=None, dtype = torch.float64, **kwargs):
    """
        calculator:pyscf_calculator instance
    """

    mfs = calculator.calculate(return_just_mf=True, **kwargs)
    if matrices is not None:
        matrixvar = []
        for i in range(len(frames)):
            mat = torch.autograd.Variable(matrices[i].type(dtype), requires_grad=True)
            matrixvar.append(mat)
        return mfs, matrixvar
    else:
        return mfs


def _instantiate_pyscf_mf_ad(frames, matrices, basis="sto-3g", dtype = torch.float64, **kwargs):
    from pyscfad.ml.scf import hf
    mfs = []
    matrixvars = []
    for frame, matrix in zip(frames, matrices):
        mol = _instantiate_pyscf_mol(frame, basis=basis)
        mat = torch.autograd.Variable(matrix.type(dtype), requires_grad=True)
        matrixvars.append(mat)
        mfs.append(hf.SCF(mol))
    return mfs, matrixvars

    
def frame_to_orbital_dict(frame, basis):
    """
    Converts a PySCF mol object for a given atomic frame using STO-3G basis 
    to a dictionary of unique atomic numbers and their orbitals.

    Parameters:
    atom_frame (str): String representation of atomic positions and elements 
                      (e.g., 'O 0 0 0; H 0 1 0; H 0 0 1')

    Returns:
    dict: A dictionary where keys are atomic numbers, and values are lists of orbital quantum numbers.
    """
    mol = _instantiate_pyscf_mol(frame, basis=basis)
    ao_labels = mol.ao_labels()
    orbs = {}
    for label in ao_labels:
        parts = label.split()
        atom_idx = int(parts[0]) 
        atom_symbol = mol.atom_symbol(atom_idx)
        atom_number = ase.data.atomic_numbers[atom_symbol]
        orbital_type = parts[2]  # Orbital type, e.g., '1s', '2px', '3d'
        nlm = convert_str_to_nlm(orbital_type)       
        if atom_number not in orbs:
            orbs[atom_number] = []
        if nlm not in orbs[atom_number]:
            orbs[atom_number].append(nlm)
    
    return orbs

class calculator:
    def __init__(
        self,
        path: str,
        structures: Optional[List[ase.Atoms]] = None,
        mol_name: str = "water",
        frame_slice=":",
        dft: bool = False,
        target: Union[str, List[str]] = "fock",
    ):  
        self.path = path
        self.structures = structures
        self.mol_name = mol_name
        self.slice = frame_slice
        self.dft = dft
        self.load_structures()
        self.pbc = False
        if np.any(self.structures[0].cell):
            self.pbc = True
        self.nframes = len(self.structures)
        print("Number of frames: ", self.nframes)

        if isinstance(target, str):
            target = [target]
        self.target = target
        if "fock" in self.target:
            self.target.append("overlap")
        self.results = {t: [] for t in self.target}
        self.ao_labels = defaultdict(list)

    def load_structures(self):
        if self.structures is None:
            try:
                print("Loading")
                self.structures = read(
                    self.path + "/{}.xyz".format(self.mol_name), index=self.slice
                )
            except:
                raise FileNotFoundError("No structures found at the given path")

    def calculate(self, basis_set: str = "sto-3g", xc:str = None, smearing:callable = None, return_just_mf= False, **kwargs: Optional[dict]):
        """
        provide smearing function from (pyscf.scf.addons import smearing_) as a fn of frame
        kwargs -
        dft: run dft
        pbc: bool = False,
        spin: int = 0,
        charge: int = 0,
        symmetry: bool = False,
        kpts: Optional[List] = None,
	unrestricted: Optional[boolean] = False

        """

        self.basis = basis_set
        verbose = kwargs.get("verbose", 5)
        spin = kwargs.get("spin", 0)
        charge = kwargs.get("charge", 0)
        symmetry = kwargs.get("symmetry", False)
        self.max_cycle = kwargs.get("max_cycle", 100)
        self.diis_space = kwargs.get("diis_space", 10)
        self.conv_tol = kwargs.get("conv_tol", 1e-10)
        self.conv_tol_grad = kwargs.get("conv_tol_grad", 1e-10)
        self.unrestricted = kwargs.get("unrestricted", False)

        self.damp = kwargs.get("damp", 0)
        self.diis_start_cycle = kwargs.get("diis_start_cycle", 1)
        
        self.xc = xc
        dm = kwargs.get("dm", None)
        init_guess = kwargs.get("init_guess", 'minao')

        if self.pbc:
            if self.unrestricted: 
                raise NotImplementedError('Unrestricted calc not supported with PBC')
            self.kpts = kwargs.get("kpts", [0, 0, 0])
            mol = pyscf.pbc.gto.Cell()
            if self.dft:
                self.calc = getattr(pyscf.pbc.dft, "KRKS")
            else:
                self.calc = getattr(pyscf.pbc.scf, "KRHF")

        else:
            mol = pyscf.gto.Mole()
            caltype = 'R' if not self.unrestricted else 'U'
            if self.dft:
                self.calc = getattr(pyscf.dft, f"{caltype}KS")
            else:
                self.calc = getattr(pyscf.scf, f"{caltype}HF")

        mol.basis = basis_set
        mol.verbose = verbose
        mol.charge = charge
        mol.spin = spin
        mol.symmetry = symmetry
        
        if return_just_mf:
            mfs = []
        for i, frame in enumerate(self.structures):
            
            mf = self.single_calc(mol, 
                frame,
                dm = dm,
                init_guess = init_guess, 
                xc = self.xc, 
                return_just_mf = return_just_mf,
                # smearing = smearing[i] 
            )

            if return_just_mf:
                mfs.append(mf)
        if return_just_mf:
            return mfs
           

    def single_calc(self,mol, frame, dm=None, init_guess=None, xc=None, smearing=None, return_just_mf= False):
        mol.atom = pyscf_ase.ase_atoms_to_pyscf(frame)
        mol.build()
        if self.pbc:
            mf = self.calc(mol, kpts=self.kpts)
            mf = mf.density_fit()
        else:
            mf = self.calc(mol)

        mf.conv_tol = self.conv_tol
        mf.conv_tol_grad = self.conv_tol_grad
        mf.max_cycle = self.max_cycle
        mf.diis_space = self.diis_space
        mf.init_guess = init_guess
        if xc is not None: # ensure xc is ONLY set for DFT calculations
            mf.xc = xc

        if smearing is not None:
            mf = smearing(mf)
        
        if return_just_mf:  # return mf and exit 
            return mf
        
        if dm is None:
            mf.kernel()
        else:
            mf.kernel(dm)
        for label in mol.ao_labels():
            _, elem, bas = label.split(" ")[:3]
            if bas not in self.ao_labels[atomic_numbers[elem]]:
                self.ao_labels[atomic_numbers[elem]].append(bas)

        print("converged:", mf.converged)
        if not mf.converged and self.max_cycle!=0:
            warnings.warn("PYSCF Calculation did not converge")

        dm = mf.make_rdm1()
        fock = mf.get_fock()
        overlap = mf.get_ovlp()
        hcore = mf.get_hcore()
        vext = mol.intor_symmetric('int1e_nuc')
        if "vext" in self.target:
            self.results["vext"].append(vext)
        if "fock" in self.target:
            self.results["fock"].append(fock)
            self.results["overlap"].append(overlap)
        if "energy" in self.target:
            self.results["energy"].append(mf.e_tot)
        if "density" in self.target:
            self.results["density"].append(dm)
        if "hcore" in self.target:
            self.results["hcore"].append(hcore)
        if "dipole_moment" in self.target:
            mo_energy, mo_coeff = mf.eig(fock, overlap)
            mo_occ = mf.get_occ(mo_energy)  # get_occ returns a numpy array
            dm1 = mf.make_rdm1(mo_coeff, mo_occ)
            self.results["dipole_moment"].append(mf.dip_moment(dm=dm1))
        del mf 
    def save_results(self, path: str = None):
        if path is None:
            path = os.path.join(self.path, self.basis)
            p = Path(path).mkdir(parents=True, exist_ok=True)
        else:
            # check if path exists
            if not os.path.exists(path):
                print("Creating path", path)
                p = Path(path).mkdir(parents=True, exist_ok=True)

        for k in self.results.keys():
            assert len(self.results[k]) == self.nframes
            self.results[k] = [torch.tensor(self.results[k][i]) for i in range(self.nframes)]
            hickle.dump(self.results[k], os.path.join(path, k + ".hickle"))

        ao_nlm = {i: [] for i in self.ao_labels.keys()}

        for k in self.ao_labels.keys():
            for v in self.ao_labels[k]:
                ao_nlm[k].append(convert_str_to_nlm(v))

        hickle.dump(ao_nlm, os.path.join(path, "orbitals.hickle"))
        print("All done, results saved at: ", path)
