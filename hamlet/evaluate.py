import torch 
import ase 
import os
import hickle
from hamlet.data.dataset import get_blocks, frame_to_orbital_dict, get_matrix_size_in_basis, compute_projector_slice
from hamlet.data.dataset import QuantumData, MLDataset
from hamlet.utils.twocenter_utils import H_power_from_blocks
from hamlet.models.linear import LinearProjector_Invariantprefactor
import metatensor.torch as mts
from ase.io import read
import numpy as np
def project_to_grassmann(P_tilde, S, occ=2.0):
    """
    Project predicted density matrix onto the closest
    idempotent density matrix in a nonorthogonal basis.

    Parameters
    ----------
    P_tilde : (n, n) ndarray
        Predicted density matrix
    S : (n, n) ndarray
        Overlap matrix
    occ : float
        Occupation per orbital, usually 2

    Returns
    -------
    P_proj : (n, n) ndarray
        Projected density matrix
    """

    evals_S, evecs_S = torch.linalg.eigh(S)
    S_half = evecs_S @ torch.diag(np.sqrt(evals_S)).to(torch.float64) @ evecs_S.T
    S_mhalf = evecs_S @ torch.diag(1.0 / np.sqrt(evals_S)) @ evecs_S.T

    X = S_half @ P_tilde @ S_half

    evals_X, evecs_X = torch.linalg.eigh(X)

    evals_proj = torch.where(evals_X > occ / 2.0, occ, 0.0)

    X_proj = evecs_X @ torch.diag(evals_proj).to(torch.float64) @ evecs_X.T
    P_proj = S_mhalf @ X_proj @ S_mhalf

    return P_proj


def load_data(BASE_DIR, 
              BASIS,
              VEXT_BASIS,
              checkpoint_file,
              path_prefix = 'data/', 
              system_name = '',
              data_slices = slice(None),
              TARGET_BASIS = None,
              DEVICE = 'cpu',
              model_target= 'fock',
              add_core = False,
              invariants_from = 'vext_power',
              shuffle = False,
              vext_power_file = 'vext_power_blocks',
              **kwargs):
    assert  model_target in ['fock', 'dm']
    assert invariants_from in ['vext', 'vext_power']
    M0 = None
    print(f"Loading checkpoint from: {checkpoint_file}", flush=True)
    ckpt = torch.load(checkpoint_file, map_location=DEVICE)
    print("Loading data...", flush=True)
    frames = read(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{system_name}.xyz'), data_slices) 
    S = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{BASIS}/overlap.hickle'))[data_slices]
    H = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{BASIS}/fock.hickle'))[data_slices]
    P = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{BASIS}/dm.hickle'))[data_slices]
    if add_core:
        if model_target == 'fock':
            M0 = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{BASIS}/hcore.hickle'))[data_slices]
            print('Loaded hcore')
        else:
            M0 = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{BASIS}/dmcore.hickle'))[data_slices]
            print('Loaded dmcore')

    if not isinstance(H[0], torch.Tensor):
        S = [torch.from_numpy(s) for s in S]
        H = [torch.from_numpy(h) for h in H]
        P = [torch.from_numpy(h) for h in P]
        M0 = [torch.from_numpy(m) for m in M0] if M0 is not None else None

    qmdata = QuantumData(
        frames=frames,
        data={'fock': H, 'overlap': S, 'dm':P, f'{model_target}0':M0} if M0 is not None else {'fock': H, 'overlap': S, 'dm':P},
        fix_matrix_orbital_order=True,
        basis=BASIS,
        device=DEVICE
    )
    target_block_name = 'cblocks_h' if model_target == 'fock' else 'cblocks_dm'
    try: 
        print(f'Loading {target_block_name}_target.hickle from {os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{BASIS}/{target_block_name}_target.hickle')}', flush=True)
        target_blocks = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{BASIS}/{target_block_name}_target.hickle'))
    except:
        print(f'Could not load, computing blocks from scratch', flush=True)
        target_blocks = get_blocks(qmdata, target=model_target, orbitals=qmdata.orbitals, device=DEVICE, 
                            orbitals_to_properties=True, skip_symmetry=True)
        hickle.dump(target_blocks, os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{BASIS}/{target_block_name}_target.hickle'))
    if add_core:
        try:
            print(f'Loading {model_target}0_target.hickle from {os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{BASIS}/{model_target}0_target.hickle')}', flush=True)
            cblocks_core = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{BASIS}/{model_target}0_target.hickle'))
        except:
            print(f'Could not load, computing blocks from scratch', flush=True)
            cblocks_core = get_blocks(qmdata, target=f'{model_target}0', orbitals=qmdata.orbitals, device=DEVICE, 
                            orbitals_to_properties=True, skip_symmetry=True)
            hickle.dump(cblocks_core, os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{BASIS}/{model_target}0_target.hickle'))
    if TARGET_BASIS is not None and TARGET_BASIS != BASIS:
        unique_numbers = np.unique(np.unique(np.concatenate([np.unique(f.numbers) for f in frames]))) 
        TARGET_orbitals = frame_to_orbital_dict(ase.Atoms(numbers=unique_numbers), basis=TARGET_BASIS)
        _, new_shapes = get_matrix_size_in_basis(qmdata, basis_string=BASIS)
        subset_slices = compute_projector_slice(qmdata, structures=qmdata.structures, orbitals_in=TARGET_orbitals, orbitals_out = qmdata.orbitals, 
                                                new_shapes=new_shapes) 
        

    else:
        TARGET_BASIS = BASIS
        subset_slices = None
    
    target_eigvals = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{TARGET_BASIS}/eigvalues.hickle'))[data_slices]
    target_charges = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{TARGET_BASIS}/lowdin_charges.hickle'))[data_slices]
    target_dipoles = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{TARGET_BASIS}/dipole_moments.hickle'))[data_slices]
    target_occupations = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{TARGET_BASIS}/pops.hickle'))[data_slices]
    target_energy = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{TARGET_BASIS}/energy.hickle'))[data_slices]

    subset_target_eigvals = [target_eigvals[ifr][subset_slices[ifr]] for ifr in range(len(frames))] if subset_slices is not None else target_eigvals 
    VEXT = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{VEXT_BASIS}/vext.hickle'))[data_slices]
    qmdata.add_second_basis(VEXT_BASIS)
    qmdata.add_data({'secondvext': VEXT}, fix_matrix_orbital_order=True, orbitals=qmdata.second_orbitals)
    if invariants_from == 'vext':
        try:
            print(f'Loading secondvext_blocks.hickle from {os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{VEXT_BASIS}/secondvext_blocks.hickle')}', flush=True)
            cblocks_vext = hickle.load(os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{VEXT_BASIS}/secondvext_blocks.hickle'))
        except:
            print(f'Could not load, computing blocks from scratch', flush=True)
        cblocks_vext = get_blocks(qmdata, matrix=qmdata.data['secondvext'], orbitals=qmdata.second_orbitals, 
                                    device=DEVICE, orbitals_to_properties=True, skip_symmetry=True)
        hickle.dump(cblocks_vext, os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{VEXT_BASIS}/secondvext_blocks.hickle'))
    
    vext_power_path = os.path.join(BASE_DIR, f'{path_prefix}/{system_name}/{VEXT_BASIS}/{vext_power_file}.hickle')
    if os.path.isfile(vext_power_path):
        print(f'Loading vext_power_blocks from {vext_power_path}', flush=True)
        vext_power_blocks = hickle.load(vext_power_path)
    else:
        # Compute to match direct.py: cblocks_vext from qmdata, then powers from ckpt params
        print(f'File not found, computing vext_power_blocks (same as experiments/water/direct.py)', flush=True)
        cblocks_vext = get_blocks(qmdata, matrix=qmdata.data['secondvext'], orbitals=qmdata.second_orbitals,
                                  device=DEVICE, orbitals_to_properties=True, skip_symmetry=True)
        matrix_power = ckpt.get('matrix_powers', [1, 2, 2])
        print('matrix_powers', matrix_power, flush=True)
        scale_matrix_power = ckpt.get('matrix_power_scales', [1, 1, 0.4])
        print('scale_matrix_power', scale_matrix_power, flush=True)
        vextblocks = []
        for power, fac in zip(matrix_power, scale_matrix_power):
            if power == 1:
                vb = cblocks_vext
            else:
                vb = H_power_from_blocks(qmdata, matrices=qmdata.data['secondvext'], orbitals=qmdata.second_orbitals,
                                        power=power, orbitals_to_properties=True, skip_symmetry=True)
            vextblocks.append(mts.multiply(vb, fac))
        vext_power_blocks = mts.join(vextblocks, axis='properties')
        # hickle.dump(vext_power_blocks, vext_power_path)

    ml_item_names = ['secondvext_power_blocks', f'second{invariants_from}_invariant', target_block_name, 
                   'eigvals', 'lowdin_charges', 'overlap', 'fock', 'dipoles', 'pops', 'dm', 'energy']
    
    if invariants_from == 'vext':
        ml_item_names.append('secondvext_blocks'  )
    ml_items_precomputed = {
        'secondvext_power_blocks': vext_power_blocks, 
        'secondvext_blocks': cblocks_vext if invariants_from == 'vext' else None,
        'eigvals': subset_target_eigvals, 
        'lowdin_charges': target_charges, 
        target_block_name: target_blocks,
        'dipoles': target_dipoles,
        'pops': target_occupations,
        'energy': target_energy,
    }
    if add_core:
        ml_item_names.append(f'{model_target}0_blocks')
        ml_items_precomputed[f'{model_target}0_blocks'] = cblocks_core
    mldata = MLDataset(
    qmdata,
    ml_item_names=ml_item_names,
    ml_items_precomputed= ml_items_precomputed,
    shuffle_seed=11324,
    shuffle=shuffle,
    features=None,
    calculate_features=False,
    model_type=None,
    train_frac=0.8, val_frac=0.2, test_frac=0,
    cutoff=None,
    skip_symmetry=True,
    )

   
    return qmdata, mldata 

def load_model(checkpoint_file, mldata, model_target, DEVICE, invariants_from = 'vext_power', **kwargs):
    assert invariants_from in ['vext', 'vext_power']
    print(f"Loading checkpoint from: {checkpoint_file}", flush=True)
    ckpt = torch.load(checkpoint_file, map_location=DEVICE)

    print('loss from ckpt', ckpt['val_loss'])

    
    NLAYERS = ckpt.get('nlayers', 3)
    ACTIVATIONS = ckpt.get('activations', ['SiLU', 'SiLU', 'SiLU'])
    HIDDEN_DIMS = ckpt.get('hidden_dims', [256, 128, 64])
    
    init_scale = kwargs.get('init_scale', 1e-1)
    invariant_init_scale = kwargs.get('invariant_init_scale', 1e-3)
    fac = kwargs.get('fac', 2.0)
    
    print(f"Model architecture: {NLAYERS} layers, hidden_dims={HIDDEN_DIMS}, activations={ACTIVATIONS}", flush=True)

    mldata.get_dataloaders(100) 
    batch = next(iter(mldata.train_dl))
    print( getattr(batch, f'second{invariants_from}_invariant')[0].values.shape, flush=True)
    
    seed = 12124
    torch.manual_seed(seed)
    np.random.seed(seed)
    target_block_name = 'cblocks_h' if model_target == 'fock' else 'cblocks_dm'
    
    model = LinearProjector_Invariantprefactor(
        basis_in_template= batch.secondvext_power_blocks, 
        basis_out_template=getattr(batch, target_block_name), 
        bias=True, 
        device=DEVICE,
        init_scale=init_scale,
        invariant_input_dim=getattr(batch, f'second{invariants_from}_invariant')[0].values.shape[-1],
        nhidden=HIDDEN_DIMS,
        nlayers=NLAYERS,
        activation=ACTIVATIONS,
        invariant_init_scale=invariant_init_scale,
        feature_wise=True,
        fac=fac,
    )
    
    print("Loading model weights from checkpoint...", flush=True)
    model.load_state_dict(ckpt['model'])

    return model

from hamlet.data.dataset import QuantumData, compute_eigenvalue, compute_dipole_ml, compute_lowdin_charges_from_dm, compute_dm_nelec
from hamlet.data.pyscf_calculator import _instantiate_pyscf_mol
from hamlet.utils.twocenter_utils import unfix_orbital_order, fix_orbital_order
import pyscf.dft as dft

def compute_properties_from_fock(frames, fock, overlap = None, basis = 'sto-3g', device = 'cpu', compute_energy = True):
    """ expected dm in correct canonical m ordering (-l<=m<-l), which differs from pyscf for l=1 """
    if overlap is None:
        raise ValueError('provide overlaps')
    energy = [] 
    dipoles = []
    relaxed_fock = []
    qmdata = QuantumData(frames = frames,
                         data = {'fock':fock,
                                 'overlap':overlap,
                                 'dm':None
                                },
                     fix_matrix_orbital_order=False,
                     basis = basis,
                     device = device)
    eigvals = compute_eigenvalue(qmdata, matrices = fock, use_overlap = True, overlaps = overlap, device = device)
    dm, _, _ = compute_dm_nelec(qmdata, frames = frames, fock = fock, overlap = overlap, orthogonal = False)

    charges, pops = compute_lowdin_charges_from_dm(qmdata, frames = frames, dm = dm, overlap = overlap, return_ao_pops=False)
    dm_fixed = unfix_orbital_order(dm, frames, qmdata.orbitals)
    if compute_energy:
        for ifr, frame in enumerate(frames):
            mol =  _instantiate_pyscf_mol(frame, basis = basis)
            mf = dft.RKS(mol)
            energy.append(mf.energy_tot(dm=dm_fixed[ifr].numpy()))
            dipoles.append( torch.from_numpy(mf.dip_moment(dm = dm_fixed[ifr].numpy(), unit='a.u.')) )
            relaxed_fock.append(mf.get_fock(dm = dm_fixed[ifr].numpy()))
    
    relaxed_fock = fix_orbital_order(relaxed_fock, frames, qmdata.orbitals)
    return {'eigvals': eigvals, 
            'dm':dm, 
            'charges':charges,
            'atom_pops':pops, 
            'dipoles':torch.stack(dipoles), 
            'energy':torch.tensor(energy),
            'relaxed_fock':relaxed_fock}

def compute_properties_from_dm(frames, dm, overlap = None, basis = 'sto-3g', device = 'cpu'):
    """ expected dm in correct canonical m ordering (-l<=m<-l), which differs from pyscf for l=1"""
    if overlap is None:
        raise ValueError('provide overlaps')
    fock = []
    energy = [] 
    dipoles = []
    qmdata = QuantumData(frames = frames,
                         data = {'dm':dm,
                                 'overlap':overlap, 
                                 'fock':None,
                                },
                     fix_matrix_orbital_order=False,
                     basis = basis,
                     device = device)
    
    dm_fixed = unfix_orbital_order(dm, frames, qmdata.orbitals)
    for ifr, frame in enumerate(frames):
        mol =  _instantiate_pyscf_mol(frame, basis = basis)
        mf = dft.RKS(mol)
        fock.append(mf.get_fock(dm = dm_fixed[ifr].numpy()) )
        energy.append(mf.energy_tot(dm=dm_fixed[ifr].numpy()))
        dipoles.append( torch.from_numpy(mf.dip_moment(dm = dm_fixed[ifr].numpy(), unit='a.u.')) )
    
    fock = fix_orbital_order(fock, frames, qmdata.orbitals)   
    fock = [torch.from_numpy(f) for f in fock]     
    eigvals = compute_eigenvalue(qmdata, matrices = fock, use_overlap = True, overlaps = overlap, device = device)
    charges, pops = compute_lowdin_charges_from_dm(qmdata, frames = frames, dm = dm, overlap = overlap, return_ao_pops=False, pred_mode = True)
    
    # with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
        # dipoles = compute_dipole_ml(qmdata,frames = frames, matrices= None, dm = dm, overlaps = overlap, use_provided_dm = True )
        
    return {'eigvals': eigvals, 
            'fock':fock,
            'energy':torch.tensor(energy),
            'charges':charges,
            'atom_pops':pops, 
            'dipoles':torch.stack(dipoles), 
           }
    

def rmse_matrix(M_true, M_pred):
    
    assert len(M_true) == len(M_pred)
    N_test = len(M_true)

    mse = 0.0
    for M, M_hat in zip(M_true, M_pred):
        assert M.shape == M_hat.shape
        N_A = M.shape[0]

        diff = M - M_hat
        mse+= torch.sum(diff** 2)/ N_A

    return torch.sqrt(mse / N_test)

def rmse_property(prediction, target):
    from hamlet.metrics import L2_loss
    # property may be vector per structure or tensor (fock, dm)
    return torch.sqrt(L2_loss(prediction, target)/len(target))
