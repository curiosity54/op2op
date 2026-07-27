# ACDC style 1,2 centered features from rascaline
# depending on the target decide what kind of features must be computed
from typing import List, Optional, Union, Tuple, Dict
import torch
import numpy as np
import warnings
import ase

import featomic.torch
from featomic.torch import SphericalExpansion
from featomic.torch import SphericalExpansionByPair as PairExpansion


import metatensor.torch as mts
from metatensor.torch import TensorMap, TensorBlock, Labels

from hamlet.features.acdc_utils import (
    acdc_standardize_keys,
    cg_combine,
    _pca,
    relabel_keys,
    fix_gij,
    drop_blocks_L,
)

use_native = True  # True for featomic by default

def single_center_features(
    frames, hypers, order_nu, lcut=None, cg=None, device="cpu", **kwargs
):  
    print(device, 'single center features')
    calculator = SphericalExpansion(**hypers)
    rhoi = calculator.compute(featomic.torch.systems_to_torch(frames), use_native_system = use_native)
    rho1i = acdc_standardize_keys(rhoi)
    rho1i = rho1i.keys_to_properties(["species_neighbor"])
    if order_nu == 1:
        return drop_blocks_L(rho1i, lcut)
    if lcut is None:
        lcut = 10
    if cg is None:
        from hamlet.utils.symmetry import ClebschGordanReal

        L = max(lcut, hypers['basis']["max_angular"])
        cg = ClebschGordanReal(lmax = L, device = device)
    rho_prev = rho1i
    for _ in range(order_nu - 2):
        rho_x = cg_combine(
            rho_prev,
            rho1i,
            clebsch_gordan=cg,
            lcut=lcut,
            other_keys_match=["species_center"],
            device=device,
        )
        rho_prev = _pca(
            rho_x, kwargs.get("npca", None), kwargs.get("slice_samples", None)
        )

    rho_x = cg_combine(
        rho_prev,
        rho1i,
        clebsch_gordan=cg,
        lcut=lcut,
        other_keys_match=["species_center"],
        feature_names=kwargs.get("feature_names", None),
        device=device,
    )
    if kwargs.get("pca_final", False):
        warnings.warn("PCA final features")
        rho_x = _pca(rho_x, kwargs.get("npca", None), kwargs.get("slice_samples", None))
    return rho_x


def pair_features(
    frames: List[ase.Atoms],
    hypers: Dict,
    hypers_pair: Dict = None,
    cg=None,
    rhonu_i: TensorMap = None,
    order_nu: Union[
        List[int], int
    ] = None,  # List - useful when combining nu on i and nu' on j
    all_pairs: bool = False,
    both_centers: bool = False,
    lcut: int = 3,
    device="cpu",
    kmesh=None,
    return_rho0ij=False,
    backend='torch',
    overwrite_cutoff = False,
    **kwargs,
):
    print(device, 'pair features')
    """
    hypers: dictionary of hyperparameters for the pair expansion as in Rascaline
    cg: object of utils:symmetry:ClebschGordanReal
    rhonu_i: TensorMap of single center features
    order_nu: int or list of int, order of the spherical expansion
    """
    
    if not isinstance(frames, list):
        frames = [frames]

    if lcut is None:
        lcut = 10

    if cg is None:
        from hamlet.utils.symmetry import ClebschGordanReal

        L = max(lcut, hypers['basis']["max_angular"]+1)
        cg = ClebschGordanReal(lmax=L, device=device)

    if hypers_pair is None:
        hypers_pair = hypers

    if all_pairs:    
        repframes = [f.repeat(kmesh[ifr]) for ifr, f in enumerate(frames)]
        min_cutoff = np.max([np.max(f.get_all_distances(mic = False)) for f in repframes])
        if hypers_pair["cutoff"]['radius'] < min_cutoff:
            if overwrite_cutoff:
                hypers_pair["cutoff"]['radius'] = np.ceil(min_cutoff)
                warnings.warn(f"Overwriting hyperparameter 'cutoff' to new value {hypers_pair['cutoff']['radius']} for all pair feature.")
            else:
                warnings.warn(f"The selected cutoff is less than the maximum distance as repeated for kmesh ({np.ceil(min_cutoff)}) among atoms in the system!")

    calculator = PairExpansion(**hypers_pair)
    rho0_ij = calculator.compute(featomic.torch.systems_to_torch(frames), use_native_system = use_native)
    rho0_ij = fix_gij(rho0_ij)
    rho0_ij = acdc_standardize_keys(rho0_ij)

    if return_rho0ij:
        return rho0_ij

    blocks = []
    keys = []
    for key, block in rho0_ij.items():
        same_species = key['species_center'] == key['species_neighbor']
        sample_labels = []
        value_indices = []

        for isample, sample in enumerate(block.samples):
            ifr, i, j, x, y, z = sample.values[:6].tolist()
            same_atoms = i == j
            is_central_cell = x == 0 and y == 0 and z == 0
            
            if not ((same_atoms and is_central_cell)):
                
                value_indices.append(isample)
                sample_labels.append([ifr, i, j, x, y, z, 1])

                if not same_species:
                    continue
                
                sample_labels.append([ifr, j, i, x, y, z, -1])
                
                neg_label = torch.tensor([ifr, j, i, -x, -y, -z])
                mappedidx = block.samples.position(neg_label)
                
                assert isinstance(mappedidx, int), (mappedidx, neg_label, key)
                value_indices.append(mappedidx)

        if not len(sample_labels):
            continue

        keys.append(key.values)
        sample_labels = torch.tensor(sample_labels)
        
        torch_block = TensorBlock(
                values = block.values[value_indices],
                samples = Labels(
                    block.samples.names + ['sign'],
                    sample_labels,
                ),
                components = block.components,
                properties = block.properties,
            )
        
        blocks.append(mts.sort_block(torch_block))
    

    rho0_ij = TensorMap(keys = Labels(rho0_ij.keys.names, torch.stack(keys)), blocks = blocks)
    
    if isinstance(order_nu, list):
        assert (
            len(order_nu) == 2
        ), "specify order_nu as [nu_i, nu_j] for correlation orders for i and j respectively"
        order_nu_i, order_nu_j = order_nu
    else:
        assert isinstance(order_nu, int), f"order_nu = {order_nu}. Specify order_nu as int or list of 2 ints"
        order_nu_i = order_nu

    if rhonu_i is None:
        lmax = hypers['basis']["max_angular"]
        rhonu_i = single_center_features(
            frames, order_nu=order_nu_i, hypers=hypers, lcut=lmax, cg=cg, device = device, kwargs=kwargs
        )
    rhonu_ij = cg_combine(
        rhonu_i,
        rho0_ij,
        clebsch_gordan=cg,
        other_keys_match=["species_center"],
        lcut=lcut,
        feature_names=kwargs.get("feature_names", None),
        device=device,
    )
    if not both_centers:
        return rhonu_ij

    else:
        if "order_nu_j" not in locals():
            warnings.warn("nu_j not defined, using nu_i for nu_j as well")
            order_nu_j = order_nu_i
        if order_nu_j != order_nu_i:
            rhonup_j = single_center_features(
                frames,
                order_nu=order_nu_j,
                hypers=hypers,
                lcut=lcut,
                cg=cg,
                device=device,
                kwargs=kwargs,
            )
        else:
            rhonup_j = rhonu_i.copy()

        rhoj = relabel_keys(rhonup_j, "species_neighbor")

        # build rhoj x gij
        rhonu_nupij = cg_combine(
            rhoj,
            rhonu_ij,
            lcut=lcut,
            other_keys_match=["species_neighbor"],
            clebsch_gordan=cg,
            mp=True,  
            feature_names=kwargs.get("feature_names", None),
            device=device,
        )

        return rhonu_nupij

def twocenter_features_symmetrized(
    single_center: TensorMap, pair: TensorMap, all_pairs = False, device = 'cpu'
) -> TensorMap:
    from collections import defaultdict

    keys = []
    blocks = []
    if "cell_shift_a" not in pair.keys.names:
        assert "cell_shift_b" not in pair.keys.names
        assert "cell_shift_c" not in pair.keys.names

    for k, b in single_center.items():
        keys.append(tuple(k) + (k["species_center"], 0,))
        if len(list(b.samples.values)) == 0:
            samples_array = b.samples
        else:
            samples_array = b.samples.values
            samples_array = torch.hstack([samples_array, samples_array[:, -1:]])
        blocks.append(
            TensorBlock(
                samples = Labels(
                    names = b.samples.names + ["neighbor", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
                    values = torch.nn.functional.pad(samples_array, (0, 3, 0, 0)),
                ),
                components = b.components,
                properties = b.properties,
                values = b.values,
            ).to(device = device)
        )

    for k, b in pair.items():
        if all_pairs:
            diff_species= k["species_center"] != k["species_neighbor"]
        else: 
            diff_species = k["species_center"] < k["species_neighbor"]

        if k["species_center"] == k["species_neighbor"]:
            # off-site, same species
            atom_i = b.samples["center"]
            atom_j = b.samples["neighbor"]
            Tx = b.samples["cell_shift_a"]
            Ty = b.samples["cell_shift_b"]
            Tz = b.samples["cell_shift_c"]
            cell_is_zero = ((Tx == 0) & (Ty == 0) & (Tz == 0))
            positive_sign = b.samples["sign"] == 1

            if all_pairs:
                different_atoms = (atom_i != atom_j)
                avoid_double_counting_atoms = True
            else:
                different_atoms = (atom_i < atom_j)
                avoid_double_counting_atoms = atom_i <= atom_j

            idx_ij = torch.where(positive_sign & ((cell_is_zero & different_atoms) | (~cell_is_zero & avoid_double_counting_atoms)))[0]

            if len(idx_ij) == 0:
                continue

            samplecopy = b.samples.values[:, :]
            block_values = b.values

            f_ijT = {1: defaultdict(lambda: torch.zeros(block_values.shape[1:], device = device)), 
                     -1: defaultdict(lambda: torch.zeros(block_values.shape[1:], device = device))}

            for idx, AijTs in enumerate(samplecopy.tolist()):
                A, i, j, Tx, Ty, Tz, sign = AijTs

                bv = block_values[idx]
                if sign == 1:   
                    f_ijT[1][A, i, j, Tx, Ty, Tz] += bv
                    f_ijT[-1][A, i, j, Tx, Ty, Tz] += bv
                else:
                    f_ijT[1][A, j, i, Tx, Ty, Tz] += bv
                    f_ijT[-1][A, j, i, Tx, Ty, Tz] -= bv

            samplelist = samplecopy[idx_ij][:,:-1]
            values_plus1 = []
            values_minus1 = []
            [(values_plus1.append(f_ijT[1][tuple(AijT)]), values_minus1.append(f_ijT[-1][tuple(AijT)]))  for AijT in samplelist.tolist()]

            keys.append(tuple(k) + (1,))
            keys.append(tuple(k) + (-1,))
            
            blocks.append(
                TensorBlock(
                    samples = Labels(
                        names = b.samples.names[:-1],
                        values = samplelist,
                    ),
                    components = b.components,
                    properties = b.properties,
                    values = torch.stack(values_plus1),
                )
            )
            blocks.append(
                TensorBlock(
                    samples = Labels(
                        names = b.samples.names[:-1],
                        values = samplelist,
                    ),
                    components = b.components,
                    properties = b.properties,
                    values = torch.stack(values_minus1),
                )
            )
        
        elif diff_species:
            # off-site, different species
            keys.append(tuple(k) + (2,))
            blocks.append(TensorBlock(
                values = b.values, 
                components = b.components,
                properties = b.properties,
                samples = Labels(b.samples.names[:-1], b.samples.values[:,:-1])).to(device = device))

    return TensorMap(
        keys = Labels(
            names = pair.keys.names + ["block_type"],
            values = torch.tensor(keys),
        ).to(device = device),
        blocks = blocks,
    )


from hamlet.data.dataset import QuantumData
def compute_features(dataset: QuantumData,
                     hypers_atom: dict,
                     lcut: int,
                     hypers_pair: Optional[dict] = None,
                     both_centers: Optional[bool] = False,
                     all_pairs: Optional[bool] = False, 
                     device: Optional[str] = 'cpu',
                     return_rhoij = False,
                     **kwargs):
    
    unique_species = set.union(*[frame.symbols.species() for frame in dataset.structures])
    structures = dataset.structures 

    if hypers_pair is None:
        hypers_pair = hypers_atom
    return_rho0ij = kwargs.get("return_rho0ij", False)
    
    rhoij = pair_features(structures, hypers_atom, hypers_pair, order_nu = 1, all_pairs = all_pairs, both_centers = both_centers,
                          kmesh = dataset.kmesh, device = device, lcut = lcut, return_rho0ij = return_rho0ij)  
    
    if return_rhoij:
        return rhoij
    
    if both_centers and not return_rho0ij:
        NU = 3
    else:
        NU = 2
    rhonui = single_center_features(structures, hypers_atom, order_nu = NU, lcut = lcut, device = device, feature_names = rhoij.property_names)

    hfeat = twocenter_features_symmetrized(single_center = rhonui, pair = rhoij, all_pairs = all_pairs, device = device)

    hfeat = mts.slice(hfeat, axis = 'samples', selection = Labels(['structure'], torch.arange(len(dataset.structures)).reshape(-1, 1)))
    keys_to_drop = [k.values for k, b in hfeat.items() if b.values.numel() == 0]
    if len(keys_to_drop) > 0:
        keys_to_drop = Labels(hfeat.keys.names, torch.stack(keys_to_drop))
        hfeat = mts.drop_blocks(hfeat, keys = keys_to_drop)

    return mts.sort(hfeat)

