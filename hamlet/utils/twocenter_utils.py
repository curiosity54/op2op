from typing import Optional, List, Union, Tuple, Dict
import warnings
from collections import defaultdict

import torch
import ase
import numpy as np
import scipy

from e3nn.o3._reduce import ReducedTensorProducts
from e3nn import o3

import metatensor.torch as mts
from metatensor.torch import TensorMap, Labels, TensorBlock
from hamlet.utils.metatensor_utils import TensorBuilder
from hamlet.utils.symmetry import ClebschGordanReal, TensorProd
import copy 
SQRT_2 = 2 ** (0.5)
ISQRT_2 = 1 / SQRT_2

def is_psd(mat):
    try :
        _ = torch.linalg.cholesky_ex(mat, check_errors = True) 
    except:
        return False 
    return True

def fix_orbital_order(
    matrix: Union[List, torch.tensor, np.ndarray],
    frames: Union[List, ase.Atoms],
    orbital: dict,
):
    """Fix the l=1 matrix components from [x,y,z] to Condon-Shortley convention of SPH with m=[-1, 0,1], handles single and multiple frames, 
    """
    def fix_one_matrix(
        matrix: Union[torch.tensor, np.ndarray], frame: ase.Atoms, orbital: dict
    ):
        idx = []
        iorb = 0
        atoms = list(frame.numbers)
        for atom_type in atoms:
            cur = ()
            for _, a in enumerate(orbital[atom_type]):
                n, l, _ = a
                if (n, l) != cur:
                    if l == 1:
                        idx += [iorb + 1, iorb + 2, iorb]
                    else:
                        idx += range(iorb, iorb + 2 * l + 1)
                    iorb += 2 * l + 1
                    cur = (n, l)
        return matrix[idx][:, idx]

    if isinstance(frames, list):
        assert len(matrix[-1].shape) == 2  # [(nframe, nao, nao)] where nao is a function of frame
        fixed_matrices = []
        for i, f in enumerate(frames):
            fixed_matrices.append(fix_one_matrix(copy.copy(matrix[i]), f, orbital))
        if isinstance(matrix, np.ndarray):
            return np.asarray(fixed_matrices)
        try: 
            return torch.stack(fixed_matrices)
        except:
            return fixed_matrices
    else:
        return fix_one_matrix(copy.copy(matrix), frames, orbital)


def unfix_orbital_order(
    matrix: Union[torch.tensor, np.ndarray],
    frames: Union[List, ase.Atoms],
    orbital: dict,
):
    """Fix the l=1 matrix components from Condon-Shortley convention of SPH with m=[-1,0,1] to [x,y,z], handles single and multiple frames"""

    def unfix_one_matrix(
        matrix: Union[torch.tensor, np.ndarray], frame: ase.Atoms, orbital: dict
    ):
        idx = []
        iorb = 0
        atoms = list(frame.numbers)
        for atom_type in atoms:
            cur = ()
            for _, a in enumerate(orbital[atom_type]):
                n, l, _ = a
                if (n, l) != cur:
                    if l == 1:
                        idx += [iorb + 2, iorb, iorb + 1]
                    else:
                        idx += range(iorb, iorb + 2 * l + 1)
                    iorb += 2 * l + 1
                    cur = (n, l)
        return matrix[idx][:, idx]

    if isinstance(frames, list):
        if len(frames) == 1:
            matrix = matrix.reshape(1, *matrix.shape)
        assert len(matrix[-1].shape) == 2,  matrix.shape  # (nframe, nao, nao)
        fixed_matrices = []
        for i, f in enumerate(frames):
            fixed_matrices.append(unfix_one_matrix(matrix[i], f, orbital))
        if isinstance(matrix, np.ndarray):
            return np.asarray(fixed_matrices)
        try:
            return torch.stack(fixed_matrices)
        except:
            return fixed_matrices
    else:
        return unfix_one_matrix(matrix, frames, orbital)


def lowdin_orthogonalize(fock: torch.tensor, overlap: torch.tensor, overlap_power = -0.5):
    """
    lowdin orthogonalization of a fock matrix computing the square root of the overlap matrix, i.e. H = S^(-1/2) F S^(-1/2)
    """
    eva, eve = torch.linalg.eigh(overlap)
    sm12 = eve @ torch.diag(torch.pow(eva, overlap_power)).to(eve) @ eve.T
    return sm12.conj() @ fock @ sm12

def orthogonalize_dm(qmdata, dm=None, overlap=None):
    """
    P = S^(1/2) P S^(1/2), call lowdin_orthogonalize with overlap_power = +0.5
    """
    dm_ = qmdata.data['dm'] if dm is None else dm
    overlap = qmdata.data['overlap'] if overlap is None else overlap
    
    if dm is not None:
        if isinstance(dm_, list):
            assert len(dm_[0].shape) == 2
        elif isinstance(dm_, torch.tensor) and len(dm_.shape)==2:
            dm_ = dm_.unsqueeze(-1)
        assert len(dm_) == len(overlap)

    dm_ortho = [0.5*lowdin_orthogonalize(p,s, overlap_power = 0.5) for p,s in zip(dm_,overlap)]

    return dm_ortho

def commutator_matrices(qmdata, matrix_type1 = 'fock', matrix_type2 = 'dm', matrix1 = None, matrix2 = None):
    """
    Compute the commutator of two matrices, i.e. [A,B] = AB - BA
    """
    matrix1 = qmdata.data[matrix_type1] if matrix1 is None else matrix1
    matrix2 = qmdata.data[matrix_type2] if matrix2 is None else matrix2
 
    commutator = [m1@m2 - m2@m1 for m1,m2 in zip(matrix1,matrix2)]
    return commutator

def _components_idx(l):
    """Returns the m \in {-l,...,l} indices"""
    return torch.arange(-l, l + 1, dtype = torch.int32).reshape(2 * l + 1, 1)


def _components_idx_2d(li, lj):
    """Returns the 2D outerproduct of m_i \in {-l_i,... , l_i} and m_j \in {-l_j,... , l_j} to index the (l_i, l_j) block of the hamiltonian
    in the uncoupled basis"""

    return torch.cartesian_prod(_components_idx(li).flatten(), _components_idx(lj).flatten()).to(dtype = torch.int32)


def _orbs_offsets(orbs):
    """offsets for the orbital subblocks within an atom block of the Hamiltonian matrix"""
    orbs_tot = {}
    orbs_offset = {}
    for k in orbs:
        ko = 0
        for n, l, m in orbs[k]:
            if m != -l:
                continue
            orbs_offset[(k, n, l)] = ko
            ko += 2 * l + 1
        orbs_tot[k] = ko
    return orbs_tot, orbs_offset


def _atom_blocks_idx(frames, orbs_tot):
    """position of the hamiltonian subblocks for each atom in each frame"""
    if isinstance(frames, ase.Atoms):
        frames = [frames]
    atom_blocks_idx = {}
    for A, f in enumerate(frames):
        ki = 0
        for i, ai in enumerate(f.numbers):
            kj = 0
            for j, aj in enumerate(f.numbers):
                atom_blocks_idx[(A, i, j)] = (ki, kj)
                kj += orbs_tot[aj]
            ki += orbs_tot[ai]
    return atom_blocks_idx

def matrix_to_blocks(matrix, orbitals, frames, device=None, all_pairs = False, cutoff = None, sort_orbs = True, high_rank = False):
    key_names = ["block_type", "species_i", "n_1", "l_1", "species_j", "n_2", "l_2"]
    sample_names = ["structure", "center", "neighbor", "cell_shift_a", "cell_shift_b", "cell_shift_c"]
    property_names = ["dummy"]
    property_values = np.asarray([[0]])
    component_names = [["m_1"], ["m_2"]]
    if high_rank:
        component_names += [["m_3"]]
        key_names += ["l_3"]
    # multiplicity
    orbs_mult = {}
    for species in orbitals:
        _, orbidx, count = np.unique(
            np.asarray(orbitals[species])[:, :2],
            axis=0,
            return_counts=True,
            return_index=True,
        )
        idx = np.argsort(orbidx)
        unique_orbs = np.asarray(orbitals[species])[orbidx[idx]][:, :2]
        orbs_mult[species] = {tuple(k): v for k, v in zip(unique_orbs, count[idx])}

    block_builder = TensorBuilder(
        key_names,
        sample_names,
        component_names,
        property_names,
        device = device
    )

    orbs_tot, _ = _orbs_offsets(orbitals)  

    for A, frame in enumerate(frames): 
        if not isinstance(matrix[A], dict):
            matrices = {(0, 0, 0): matrix[A]}
        else: matrices = matrix[A] 
      
        if len(matrix[A].shape)==3:
                high_rank = True
                warnings.warn("high_rank must be True if matrix is a 3D tensor, setting to True")
                l3 = matrix[A].shape[-1]//2 # assuming the last dimension is 2*l+1

        for _,T in enumerate(matrices):
            mT = tuple(-t for t in T)
            assert mT in matrices, f"{mT} not in the real space matrix keys"

            matrixT = matrices[T]
            matrixmT = matrices[mT]
            i_start = 0
            # Loop over the all the atoms in the structure, by atomic number
            for i, ai in enumerate(frame.numbers):
                orbs_i = orbs_mult[ai]
                j_start = 0

                # Loop over the all the atoms in the structure, by atomic number
                for j, aj in enumerate(frame.numbers):

                    # Handle the case only the upper triangle is learnt
                    if not all_pairs: # not all orbital pairs
                        if i > j and ai == aj: # skip block type 1 if i>j 
                            j_start += orbs_tot[aj]
                            continue
                        elif ai > aj: # keep only sorted species 
                            j_start += orbs_tot[aj]
                            continue
                       
                    # Skip the pair if their distance exceeds the cutoff
                    if cutoff is not None:
                        ij_distance = np.linalg.norm(frame.cell.array.T @ np.array(T) + frame.get_distance(i,j,mic=False,vector=True))
                        if ij_distance > cutoff:
                            j_start += orbs_tot[aj]
                            continue
                        
                    orbs_j = orbs_mult[aj]

                    # add what kind of blocks we expect in the tensormap
                    # n1l1n2l2 = list(sum([tuple(k2 + k1 for k1 in orbs_i) for k2 in orbs_j], ()))
                    n1l1n2l2 = list(sum([tuple(k2 + k1 for k1 in orbs_j) for k2 in orbs_i], ()))

                    block_ij = matrixT[i_start:i_start + orbs_tot[ai], j_start:j_start + orbs_tot[aj]]

                    block_split = [torch.split(blocki, list(orbs_j.values()), dim = 1) for blocki in torch.split(block_ij, list(orbs_i.values()), dim = 0)]
                    block_split = [y for x in block_split for y in x]  # flattening the list of lists above

                    for iorbital, (ni, li, nj, lj) in enumerate(n1l1n2l2):
                        value = block_split[iorbital]
                        components_idx=[_components_idx(li), _components_idx(lj)]
                        data_shape =  (1, 2 * li + 1, 2 * lj + 1, 2*l3 + 1,1) if high_rank else (1, 2 * li + 1, 2 * lj + 1,1)
                        if i == j and np.linalg.norm(T) == 0:
                            if sort_orbs:
                                if ni > nj or (ni == nj and li > lj):
                                    continue
                            # On-site
                            block_type = 0
                            key = (block_type, ai, ni, li, aj, nj, lj)

                        elif (ai == aj) or (i == j and T != [0, 0, 0]):
                            # Same species interaction
                           #----sorting ni,li,nj,lj---  
                            if sort_orbs:
                                if ni > nj or (ni == nj and li > lj):
                                    continue
                            #-------
                            block_type = 1
                            key = (block_type, ai, ni, li, aj, nj, lj)
                            block_jimT = matrixmT[j_start : j_start + orbs_tot[aj], i_start : i_start + orbs_tot[ai]]
                            block_jimT_split = [torch.split(blocki, list(orbs_i.values()), dim=1) for blocki in torch.split(block_jimT, list(orbs_j.values()), dim = 0)]
                            block_jimT_split = [y for x in block_jimT_split for y in x]  # flattening the list of lists above
                            # value_ji \equiv H_{ji}(-T)[\phi, \psi]
                            value_ji = block_jimT_split[iorbital]  # same orbital in the ji subblock
                            
                        else:
                            # Different species interaction
                            # skip ai>aj if not all_pairs
                            block_type = 2
                            key = (block_type, ai, ni, li, aj, nj, lj)
                        
                        if high_rank: 
                            key += (l3,)
                            components_idx += [_components_idx(l3)]
                            
                        if key not in block_builder.blocks:
                            # add blocks if not already present
                            block = block_builder.add_block(key=key, properties=property_values, components = components_idx)
                            if block_type == 1:
                                block = block_builder.add_block(
                                    key=(-1,) + key[1:],
                                    properties=property_values,
                                    components=components_idx,
                                )

                        # add samples to the blocks when present
                        block = block_builder.blocks[key]
                        if block_type == 1:
                            block_asym = block_builder.blocks[(-1,) + key[1:]]

                        if block_type == 1:
                            # if i > j:  # keep only (i,j) and not (j,i)
                                # continue

                            # block_(+1)ijT = <i \phi| H(T)|j \psi> + <j \phi| H(-T)|i \psi>
                            bplus = (value + value_ji) * ISQRT_2
                            # block_(-1)ijT = <i \phi| H(T)|j \psi> - <j \phi| H(-T)|i \psi>
                            bminus = (value - value_ji) * ISQRT_2

                            block.add_samples(     labels = [(A, i, j, *T)], data =  bplus.reshape(data_shape))
                            block_asym.add_samples(labels = [(A, i, j, *T)], data = bminus.reshape(data_shape))

                        elif block_type == 0 or block_type == 2:
                            block.add_samples(labels = [(A, i, j, *T)], data = value.reshape(data_shape))
                        
                        else:
                            raise ValueError("Block type not implemented")
                    j_start += orbs_tot[aj]

                i_start += orbs_tot[ai]
    return block_builder.build()

def blocks_to_matrix(blocks, orbitals, frames, device=None, cg = None, all_pairs = False, sort_orbs = True, batchid = None):
    if "L" in blocks.keys.names:
        blocks = _to_uncoupled_basis(blocks, cg = cg, device = device)

    orbs_tot, orbs_offset = _orbs_offsets(orbitals)
    atom_blocks_idx = _atom_blocks_idx(frames, orbs_tot)
    orbs_mult = {
        species: 
                {tuple(k): v
            for k, v in zip(
                *np.unique(
                    np.asarray(orbitals[species])[:, :2],
                    axis=0,
                    return_counts=True,
                )
            )
        }
        for species in orbitals
    }

    bt1factor = ISQRT_2
    if all_pairs:
        bt1factor /= 2

    reconstructed_matrices = [{} for _ in range(len(frames))]
    if batchid is not None:
        reconstructed_matrices = [{} for _ in range(len(batchid))]
        mapping = {val: idx for idx, val in enumerate(batchid)}
    
    for key, block in blocks.items():
        block_type = key["block_type"]
        ai, ni, li = key["species_i"], key["n_1"], key["l_1"]
        aj, nj, lj = key["species_j"], key["n_2"], key["l_2"]
        
        #----sorting ni,li,nj,lj---
        if sort_orbs:
            fac=1 # sorted orbs - we only count everything once
            if ai == aj and (ni == nj and li == lj): #except these diag blocks
                fac=2 #so we need to divide by 2 to avoic double count
        else: 
            # no sorting -->  we count everything twice
            fac=2
        #----sorting ni,li,nj,lj---
        # What's the multiplicity of the orbital type, ex. 2p_x, 2p_y, 2p_z makes the multiplicity 
        # of a p block = 3
        orbs_i = orbs_mult[ai]
        orbs_j = orbs_mult[aj]
        
        # The shape of the block corresponding to the orbital pair
        shapes = {
            (k1 + k2): (orbs_i[tuple(k1)], orbs_j[tuple(k2)])
            for k1 in orbs_i
            for k2 in orbs_j
        }
        # where does orbital PHI = (ni, li) start within a block of atom i
        phioffset = orbs_offset[(ai, ni, li)] 
        # where does orbital PSI = (nj,lj) start within a block of atom j
        psioffset = orbs_offset[(aj, nj, lj)]

        # loops over samples (structure, i, j)
        for sample, blockval in zip(block.samples.values, block.values):

            if blockval.numel() == 0:
                # Empty block
                continue        
            try:
                A1, i, j, Tx, Ty, Tz = sample.tolist() # NOTE use A, i,j .. if you want to use sample.structure index to update the reconstructed_matrices
                T = Tx, Ty, Tz
            except:
                A1, i, j = sample.tolist() 
                T = 0,0,0
            
            
            A = mapping[A1] if batchid is not None else A1
            mT = tuple(-t for t in T)

            other_fac = 1
            if i == j and T != (0,0,0) and not all_pairs:
                other_fac = 0.5

            # bt 0
            if not sort_orbs:
                bt0_factor_p = 0.5
            else: 
                if not(ni==nj and li==lj):
                    bt0_factor_p = 1
                else:
                    bt0_factor_p = 0.5
            bt0_factor_m = bt0_factor_p*other_fac

            # bt 2 
            bt2_factor_p=0.5
            if not all_pairs:
                bt2_factor_p=1
            bt2_factor_m = bt2_factor_p*other_fac

            # bt 1
            bt1_fact_fin = bt1factor/fac*other_fac
            if T not in reconstructed_matrices[A]:
                assert mT not in reconstructed_matrices[A], "why is mT present but not T?"
                norbs = np.sum([orbs_tot[ai] for ai in frames[A].numbers])
                reconstructed_matrices[A][T] = torch.zeros(norbs, norbs, device = device, dtype=torch.float64)
                reconstructed_matrices[A][mT] = torch.zeros(norbs, norbs, device = device, dtype=torch.float64)
            
            matrix_T  = reconstructed_matrices[A][T]
            matrix_mT = reconstructed_matrices[A][mT]
            # beginning of the block corresponding to the atom i-j pair
            i_start, j_start = atom_blocks_idx[(A, i, j)]
            # where does orbital (ni, li) end (or how large is it)
            phi_end = shapes[(ni, li, nj, lj)][0]  # orb end
            # where does orbital (nj, lj) end (or how large is it)
            psi_end = shapes[(ni, li, nj, lj)][1]  

            iphi_jpsi_slice = slice(i_start + phioffset , i_start + phioffset + phi_end),\
                              slice(j_start + psioffset , j_start + psioffset + psi_end)
            ipsi_jphi_slice = slice(i_start + psioffset , i_start + psioffset + psi_end),\
                              slice(j_start + phioffset , j_start + phioffset + phi_end),
                            
            jphi_ipsi_slice = slice(j_start + phioffset , j_start + phioffset + phi_end),\
                              slice(i_start + psioffset , i_start + psioffset + psi_end)
            
            jpsi_iphi_slice = slice(j_start + psioffset , j_start + psioffset + psi_end),\
                              slice(i_start + phioffset , i_start + phioffset + phi_end)

            bv = blockval[:, :, 0]
            # position of the orbital within this block
            if block_type == 0:
                matrix_T[iphi_jpsi_slice] += bv*bt0_factor_p
                matrix_mT[jpsi_iphi_slice] += bv.T*bt0_factor_m
                
            elif block_type == 2:

                
                matrix_T[iphi_jpsi_slice] += bv*bt2_factor_p
                matrix_mT[jpsi_iphi_slice] += bv.T*bt2_factor_m
                
            elif abs(block_type) == 1:
                # Eq (1) <i \phi| H(T)|j \psi> = # block_(+1)ijT + block_(-1)ijT 
                # Eq (2) <j \phi| H(-T)|i \psi> = # block_(+1)ijT - block_(-1)ijT 
                # Eq (3) <j \psi| H(-T)|i \phi> = # block_(+1)ijT^\dagger + block_(-1)ijT^\dagger (Transpose of Eq1) 
                # Eq (4) <i \psi| H(T)|j \phi> = # block_(+1)ijT^\dagger - block_(-1)ijT^\dagger (Transpose of Eq2)
                bv = bv*bt1_fact_fin

                if block_type == 1:
                    # first half of Eq (1) 
                    matrix_T[iphi_jpsi_slice] += bv
                    # first half of Eq (2)
                    matrix_mT[jphi_ipsi_slice] += bv
                    # first half of Eq (3)
                    matrix_mT[jpsi_iphi_slice] += bv.T
                    # first half of Eq (4)
                    matrix_T[ ipsi_jphi_slice] += bv.T
        
                else:
                    # second half of Eq (1)
                    matrix_T[iphi_jpsi_slice] += bv
                    # second half of Eq (2)
                    matrix_mT[jphi_ipsi_slice] -= bv
                    # second half of Eq (3)
                    matrix_mT[jpsi_iphi_slice] += bv.T
                    # second half of Eq (4)
                    matrix_T[ipsi_jphi_slice ] -= bv.T    
    
    for A, matrix in enumerate(reconstructed_matrices):
        Ts = list(matrix.keys())
        for T in Ts:
            mT = tuple(-t for t in T)
         
            assert torch.all(torch.isclose(matrix[T] - reconstructed_matrices[A][mT].T, torch.zeros_like(matrix[T]))), torch.norm(matrix[T] - reconstructed_matrices[A][mT].T).item()

    if all(d.keys() == {(0, 0, 0)} for d in reconstructed_matrices):
        try: 
            return torch.stack([d[(0, 0, 0)] for d in reconstructed_matrices])
        except:
            return [d[(0, 0, 0)] for d in reconstructed_matrices]
    
    return reconstructed_matrices




def move_orbitals_to_keys(in_blocks, dummy_property = None):

    device = in_blocks.device

    if dummy_property is None: 
        dummy_property = Labels(["dummy"], torch.tensor([[0]], device = device))
    else:
        dummy_property.to(device = device)

    blocks = []
    keys = []
    for k,b in in_blocks.items():
        n1l1n2l2 = torch.unique(b.properties.values[:,:4], dim=0)
        block_view = b.properties.view(['n_1', 'l_1', 'n_2', 'l_2']).values
        
        for nlinlj in n1l1n2l2:
            idx = torch.where(torch.all(torch.isclose(block_view, nlinlj), dim = 1))[0]
            
            keys.append(torch.hstack((k.values, nlinlj.clone().detach())))
            if len(idx):
                blocks.append(TensorBlock(
                            samples = b.samples,
                            values = b.values[...,idx],
                            components = b.components,
                            properties = dummy_property
                        )
                )
    keys = Labels(in_blocks.keys.names+['n_1', 'l_1', 'n_2', 'l_2'], torch.stack(keys).to(device = device))
    # keys = block_type, species_i, species_j, L, sigma, n_i, l_i, n_j, l_j
    tmap = TensorMap(keys, blocks)
    return mts.permute_dimensions(tmap, axis='keys', dimensions_indexes = [0,1,5,6,2,7,8,3,4])


def _to_coupled_basis(
    blocks: Union[torch.tensor, TensorMap],
    orbitals: Optional[dict] = None,
    cg: Optional[ClebschGordanReal] = None,
    device: str = "cpu",
    skip_symmetry: bool = False,
    translations: bool = None,
):
    if torch.is_tensor(blocks):
        print("Converting matrix to blocks before coupling")
        assert orbitals is not None, "Need orbitals to convert matrix to blocks"
        blocks = matrix_to_blocks(blocks, orbitals)
    if cg is None:
        lmax = max(blocks.keys["l_1"] + blocks.keys["l_2"])
        cg = ClebschGordanReal(lmax, device=device)
    if translations is None:
        block_builder = TensorBuilder(
            ["block_type", "species_i", "n_1", "l_1", "species_j", "n_2", "l_2", "L"],
            ["structure", "center", "neighbor"],
            [["M"]],
            ["dummy"],
        )
    else:
        if translations:
            block_builder = TensorBuilder(
                ["block_type", "species_i", "n_1", "l_1", "species_j", "n_2", "l_2", "L"],
                ["structure", "center", "neighbor", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
                [["M"]],
                ["dummy"],
            )
        else:
            block_builder = TensorBuilder(
                ["block_type", "species_i", "n_1", "l_1", "species_j", "n_2", "l_2", "L"],
                ["structure", "center", "neighbor", "kpoint"],
                [["M"]],
                ["dummy"],
            )
        
        
    for idx, block in blocks.items():
        block_type = idx["block_type"]
        ai = idx["species_i"]
        ni = idx["n_1"]
        li = idx["l_1"]
        aj = idx["species_j"]
        nj = idx["n_2"]
        lj = idx["l_2"]

        # Moves the components at the end as cg.couple assumes so
        decoupled = torch.moveaxis(block.values, -1, -2).reshape(
            (len(block.samples), len(block.properties), 2 * li + 1, 2 * lj + 1)
        )

        # selects the (only) key in the coupled dictionary (l1 and l2
        # that contribute to the coupled terms L, with L going from
        # |l1 - l2| up to |l1 + l2|
        coupled = cg.couple(decoupled)[(li, lj)]

        for L in coupled:
            block_idx = tuple(idx) + (L,)
            # skip blocks that are zero because of symmetry
            if ai == aj and ni == nj and li == lj:
                parity = (-1) ** (li + lj + L)
                if ((parity == -1 and block_type in (0, 1)) or (parity == 1 and block_type == -1)) and not skip_symmetry:
                    continue

            new_block = block_builder.add_block(
                key=block_idx,
                properties=torch.tensor([[0]], dtype=torch.int32),
                components=[_components_idx(L).reshape(-1, 1)],
            )

            new_block.add_samples(
                labels = block.samples.values.reshape(block.samples.values.shape[0], -1),
                data = torch.moveaxis(coupled[L], -1, -2),
            )

    return block_builder.build()

def _to_uncoupled_basis(
    blocks: TensorMap,
    cg: Optional[ClebschGordanReal] = None,
    device: str = "cpu",
    translations: bool = False,
):
    if cg is None:
        lmax = max(blocks.keys["L"])
        cg = ClebschGordanReal(lmax, device = device)

    dummy_property = Labels(['dummy'], torch.tensor([[0]], device = device))

    uncoupled_blocks = {}
    samples = {}
    for key, block in blocks.items():

        if block.values.numel() == 0:
            # Empty block
            continue
        values = block.values
        dtype = values.dtype
        block_type, ai, aj,L  = key['block_type'], key['species_i'], key['species_j'], key['L']
        # block_type, ai, aj, L  = key.values[:4].tolist()

        for ip, (ni, li, nj, lj) in enumerate(block.properties.values[:, :4].tolist()):
            k = block_type, ai, ni, li, aj, nj, lj
            
            if k not in uncoupled_blocks:
                 uncoupled_blocks[k] = torch.zeros((values.shape[0], 2*li+1, 2*lj+1, 1), device = device, dtype = dtype)
                 samples[k] = block.samples

            uncoupled_blocks[k].add_(torch.tensordot(values[:,:,ip:ip+1], cg._cg[(li, lj, L)].to(dtype = dtype), dims=([1], [2])).permute(0, 2, 3, 1))

    new_blocks = []
    new_keys = []
    for k in uncoupled_blocks:
        _, _, _, li, _, _, lj = k
        new_keys.append(k)
        new_blocks.append(
            TensorBlock(
                values = uncoupled_blocks[k],
                samples = samples[k],
                properties = dummy_property,
                components = [Labels(['m_1'], torch.arange(-li, li+1, device = device).reshape(-1, 1)), Labels(['m_2'], torch.arange(-lj, lj+1, device = device).reshape(-1, 1))]
                )
        )
    return mts.sort(TensorMap(Labels(['block_type', 'species_i', 'n_1', 'l_1', 'species_j', 'n_2', 'l_2'], torch.tensor(new_keys, device = device)), new_blocks))


def tensor_product_coupled(
    blocks_left: TensorMap,
    blocks_right: Optional[TensorMap] = None,
    *,
    cg: Optional[ClebschGordanReal] = None,
    device: str = "cpu",
    invariants_only: bool = False,
):
    """Build H\otimesH (or, more generally, tensor product of two coupled two-center operators) in the coupled basis.

    Inputs must be the coupled representation produced by `_to_coupled_basis`, i.e. keys include
    `['block_type','species_i','n_1','l_1','species_j','n_2','l_2','L']` and values have shape
    `(num_samples, 2*L+1, num_properties)`.

    If `blocks_right` is None, the right operand is taken to be `blocks_left` (self tensor product).

    The output keys enumerate the four centers/orbitals and the coupled irrep `G`:
    `['block_type_left','species_i_1','n_1','l_1','species_j_1','n_2','l_2',
      'block_type_right','species_i_2','n_1p','l_1p','species_j_2','n_2p','l_2p','G']`.
    Components are `['M']` for the target irrep `G` unless `invariants_only=True`, in which case
    only `G=0` is kept and the result has no components.
    """

    if blocks_right is None:
        blocks_right = blocks_left

    if cg is None:
        lmax_left = int(torch.max(blocks_left.keys["L"]).item()) if "L" in blocks_left.keys.names else 0
        lmax_right = int(torch.max(blocks_right.keys["L"]).item()) if "L" in blocks_right.keys.names else 0
        cg = ClebschGordanReal(max(lmax_left, lmax_right), device=device)

    # Determine if translations are present in samples; preserve sample semantics
    if "cell_shift_a" in blocks_left.sample_names:
        sample_names = [
            "structure",
            "center_1",
            "neighbor_1",
            "cell_shift_a_1",
            "cell_shift_b_1",
            "cell_shift_c_1",
            "center_2",
            "neighbor_2",
            "cell_shift_a_2",
            "cell_shift_b_2",
            "cell_shift_c_2",
        ]
    else:
        # default (no translations)
        # if k-point encoded, keep a separate index from each operand
        if "kpoint" in blocks_left.sample_names:
            sample_names = ["structure", "center_1", "neighbor_1", "kpoint_1", "center_2", "neighbor_2", "kpoint_2"]
        else:
            sample_names = ["structure", "center_1", "neighbor_1", "center_2", "neighbor_2"]

    key_names = [
        "block_type_left",
        "species_i_1",
        "n_1",
        "l_1",
        "species_j_1",
        "n_2",
        "l_2",
        "block_type_right",
        "species_i_2",
        "n_1p",
        "l_1p",
        "species_j_2",
        "n_2p",
        "l_2p",
        "G",
    ]

    component_names = [] if invariants_only else [["M"]]

    block_builder = TensorBuilder(
        key_names=key_names,
        sample_names=sample_names,
        component_names=component_names,
        property_names=["dummy"],
        device=device,
    )

    # Index left blocks by their orbital/species/L key for faster matching
    def _pack_left_key(k):
        return (
            int(k["block_type"]),
            int(k["species_i"]),
            int(k["n_1"]),
            int(k["l_1"]),
            int(k["species_j"]),
            int(k["n_2"]),
            int(k["l_2"]),
            int(k["L"]),
        )

    left_grouped: Dict[Tuple[int, int, int, int, int, int, int, int], Tuple[Labels, torch.Tensor]] = {}
    for k_left, b_left in blocks_left.items():
        key_tuple = _pack_left_key(k_left)
        left_grouped[key_tuple] = (b_left.samples, b_left.values)

    # Iterate over all combinations of left/right keys
    for k_left, (samples_left, vals_left) in left_grouped.items():
        btl, ai1, n1, l1, aj1, n2, l2, L1 = k_left

        for k_right, b_right in blocks_right.items():
            btr = int(k_right["block_type"]) if "block_type" in k_right.names else int(k_right[0])
            ai2 = int(k_right["species_i"]) if "species_i" in k_right.names else int(k_right[1])
            n1p = int(k_right["n_1"]) if "n_1" in k_right.names else int(k_right[2])
            l1p = int(k_right["l_1"]) if "l_1" in k_right.names else int(k_right[3])
            aj2 = int(k_right["species_j"]) if "species_j" in k_right.names else int(k_right[4])
            n2p = int(k_right["n_2"]) if "n_2" in k_right.names else int(k_right[5])
            l2p = int(k_right["l_2"]) if "l_2" in k_right.names else int(k_right[6])
            L2 = int(k_right["L"]) if "L" in k_right.names else int(k_right[7])

            vals_right = b_right.values  # (N_sr, 2L2+1, P)
            samples_right = b_right.samples

            # Build Cartesian product of samples from left/right
            Ns_l = samples_left.values.shape[0]
            Ns_r = samples_right.values.shape[0]

            if Ns_l == 0 or Ns_r == 0:
                continue

            # Prepare CG table for (L1,L2)->G
            possible_G = range(abs(L1 - L2), L1 + L2 + 1)
            for G in possible_G:
                if invariants_only and G != 0:
                    continue

                # Prepare output block
                out_key = (
                    btl,
                    ai1,
                    n1,
                    l1,
                    aj1,
                    n2,
                    l2,
                    btr,
                    ai2,
                    n1p,
                    l1p,
                    aj2,
                    n2p,
                    l2p,
                    G,
                )
                if out_key not in block_builder.blocks:
                    block_builder.add_block(
                        key=out_key,
                        properties=torch.tensor([[0]], dtype=torch.int32, device=device),
                        components=([] if invariants_only else [Labels(["M"], _components_idx(G).to(device=device))]),
                    )

                # Compute coupled tensor product for all sample pairs
                # Expand to Cartesian product using broadcasting
                # Left: (Ns_l, 2L1+1, P) -> (Ns_l, 1, 2L1+1)
                # Right: (Ns_r, 2L2+1, P) -> (1, Ns_r, 2L2+1)
                vL = vals_left[:, :, 0].to(device)
                vR = vals_right[:, :, 0].to(device)

                vL_exp = vL.unsqueeze(1)
                vR_exp = vR.unsqueeze(0)

                # CG tensor: (2L1+1, 2L2+1, 2G+1)
                CG = cg._cg[(L1, L2, G)].to(device=vL.device, dtype=vL.dtype)

                # Result: (Ns_l, Ns_r, 2G+1) or (Ns_l, Ns_r) if invariants
                if invariants_only:
                    # For G=0, 2G+1=1; still perform contraction and squeeze
                    prod = torch.einsum("l r m1 m2, m1 m2 g -> l r g", vL_exp.unsqueeze(-1) * vR_exp.unsqueeze(-2), CG)
                    prod = prod.squeeze(-1)
                else:
                    prod = torch.einsum("l r m1 m2, m1 m2 g -> l r g", vL_exp.unsqueeze(-1) * vR_exp.unsqueeze(-2), CG)

                # Build samples for the Cartesian product
                sL = samples_left.values
                sR = samples_right.values

                if "cell_shift_a" in blocks_left.sample_names:
                    # Map to the names order
                    new_samples = torch.cartesian_prod(
                        torch.arange(Ns_l, device=device), torch.arange(Ns_r, device=device)
                    )
                    iL = new_samples[:, 0]
                    iR = new_samples[:, 1]
                    sl = sL[iL]
                    sr = sR[iR]
                    # structure indices must match; filter to pairs within same structure
                    same_struct = sl[:, 0] == sr[:, 0]
                    if not torch.any(same_struct):
                        continue
                    sl = sl[same_struct]
                    sr = sr[same_struct]
                    prod_flat = prod[iL[same_struct], iR[same_struct]]
                    # [A, i, j, Ta, Tb, Tc] + [k, l, Ta', Tb', Tc']
                    samples_concat = torch.cat(
                        [
                            sl[:, [0, 1, 2, 3, 4, 5]],
                            sr[:, [1, 2, 3, 4, 5]],
                        ],
                        dim=1,
                    )
                else:
                    new_samples = torch.cartesian_prod(
                        torch.arange(Ns_l, device=device), torch.arange(Ns_r, device=device)
                    )
                    iL = new_samples[:, 0]
                    iR = new_samples[:, 1]
                    sl = sL[iL]
                    sr = sR[iR]
                    # require same structure index in column 0
                    same_struct = sl[:, 0] == sr[:, 0]
                    if not torch.any(same_struct):
                        continue
                    sl = sl[same_struct]
                    sr = sr[same_struct]
                    prod_flat = prod[iL[same_struct], iR[same_struct]]

                    if "kpoint" in blocks_left.sample_names:
                        samples_concat = torch.cat(
                            [sl[:, [0, 1, 2, 3]], sr[:, [1, 2, 3]]], dim=1
                        )
                    else:
                        samples_concat = torch.cat(
                            [sl[:, [0, 1, 2]], sr[:, [1, 2]]], dim=1
                        )

                # Finally, add to block
                if invariants_only:
                    data = prod_flat.reshape(prod_flat.shape[0], 1, 1)
                else:
                    data = prod_flat.unsqueeze(-1)
                block_builder.blocks[out_key].add_samples(
                    labels=[tuple(s.tolist()) for s in samples_concat],
                    data=data,
                )

    return block_builder.build()


def invariants_from_tensor_product(blocks_left: TensorMap, blocks_right: Optional[TensorMap] = None, *, cg: Optional[ClebschGordanReal] = None, device: str = "cpu") -> TensorMap:
    """Convenience wrapper to compute scalar invariants (G=0) from H⊗H.

    Equivalent to calling `tensor_product_coupled(..., invariants_only=True)`.
    """
    return tensor_product_coupled(blocks_left, blocks_right, cg=cg, device=device, invariants_only=True)


def H_power_from_blocks(
    qmdata, 
    matrix_type: str = "fock",
    matrices=None, 
    power: int = 2,
    couple: bool = True,
    cg: Optional[ClebschGordanReal] = None,
    device: str = "cpu",
    orbitals = None,
    **kwargs,
):
    """Compute H^power from AO matrices for molecules (no translations, variable sizes allowed).
    - Get matrices from qmdata or provided matrices (uses provided matrices if both matrices and matrix_type are set)
    - Compute H^power independently per structure
    - convert to blocks 

    Returns uncoupled blocks unless `couple=True`.
    """
    assert matrix_type in ['fock0', 'fock', 'overlap', 'orthogonalH', 'dm', 'dm0', 'orthoH', 'vext']
    assert power >= 1 and isinstance(power, int), "power must be a positive integer"

    matrices = matrices if matrices is not None else qmdata.data[matrix_type]
    if isinstance(matrices[0], dict):
        matrices = [m[(0, 0, 0)] for m in matrices]

    # Compute power per frame
    Hp_list: List[torch.Tensor] = []
    for H in matrices:
        Hp = H.clone()
        for _ in range(1, power):
            Hp = Hp @ H
        Hp_list.append(Hp)

    from hamlet.utils.target_utils import get_blocks
    Hp_blocks = get_blocks(
        qmdata, 
        target=matrix_type,
        matrix=Hp_list,
        device=device,
        return_uncoupled=not couple,  # Return uncoupled if not coupling
        orbitals = orbitals,
        **kwargs
    )

    return Hp_blocks


def recover_H_from_Hp(Hp_matrices, power=1):
    """ recover H from H^p """
    H_recovered = []
    for Hp in Hp_matrices:
        # Eigendecomposition: H² = QΛ²Q^T
        evalsp, evecs = torch.linalg.eigh(Hp)
        # Take positive square root: H = Q√Λ²Q^T
        sign = torch.sign(evalsp)
        evals = sign*torch.pow(torch.abs(evalsp), 1/power) 
        H = evecs @ torch.diag(evals) @ evecs.T
        H_recovered.append(H)
    return H_recovered

def get_power_blocks(qmdata, matrix_type, powers, device = 'cpu', orbitals_to_properties = True, skip_symmetry = True, orbitals = None):
    """ Get concatenated blocks of matrix powers """
    power_blocks = []
    for power in powers:
        power_blocks.append(H_power_from_blocks(qmdata, 
                                                matrix_type = matrix_type, 
                                                device=device, 
                                                orbitals_to_properties = orbitals_to_properties, 
                                                skip_symmetry = skip_symmetry, 
                                                power = power, 
                                                orbitals = orbitals))
    
    joined_power_blocks = mts.join(power_blocks, axis = 'properties')
    return joined_power_blocks

def apply_cutoff(qmdata, matrix_type = None, matrix = None, frames=None, orbitals=None, cutoff=100):
    # set eleemnts of matrix beyond r_ij>cutoff to zero

    matrices = matrix if matrix is not None else qmdata.data[matrix_type]
    frames = frames if frames is not None else qmdata.structures
    orbitals = orbitals if orbitals is not None else qmdata.orbitals
    assert len(matrices) == len(frames)
    cutoff_mat = []
    orbs_tot, _ = _orbs_offsets(orbitals)
    atom_idx = _atom_blocks_idx(frames, orbs_tot) 
    for A, (mat, frame) in enumerate(zip(matrices, frames)): 
        out = mat.clone()
        for i in range(len(frame)):
            for j in range(i, len(frame)):
                dij = frame.get_distance(i,j)
                if dij>cutoff and (A, i, j) in atom_idx:
                    i_start, j_start = atom_idx[(A, i, j)]
                    atom_i = frame.numbers[i]
                    atom_j = frame.numbers[j]
                    
                    i_slice = slice(i_start, i_start + orbs_tot[atom_i])
                    j_slice = slice(j_start, j_start + orbs_tot[atom_j])
                    out[i_slice, j_slice] = 0
                    # symmetric counterpart 
                    if i != j and (A, j, i) in atom_idx:
                        out[j_slice, i_slice] = 0
        
   
        cutoff_mat.append(out)
   
    return cutoff_mat

def get_nonzero_atom_slices(qmdata, frames=None, orbitals=None, cutoff=100):
    """ Get the slices of the matrix for each atom pair """
    frames = frames if frames is not None else qmdata.structures
    orbitals = orbitals if orbitals is not None else qmdata.orbitals
    orbs_tot, _ = _orbs_offsets(orbitals)
    atom_idx = _atom_blocks_idx(frames, orbs_tot) 
    atom_slices = {}
    for A, frame in enumerate(frames): 
        for i in range(len(frame)):
            for j in range(i, len(frame)):
                dij = frame.get_distance(i,j)
                if dij<=cutoff and (A, i, j) in atom_idx:
                    i_start, j_start = atom_idx[(A, i, j)]
                    atom_i = frame.numbers[i]
                    atom_j = frame.numbers[j]
                    
                    i_slice = slice(i_start, i_start + orbs_tot[atom_i])
                    j_slice = slice(j_start, j_start + orbs_tot[atom_j])
                    atom_slices[(A, i, j)] = (i_slice, j_slice)
    return atom_slices