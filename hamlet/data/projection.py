import torch 
from hamlet.data.dataset import isqrtp, build_ao2atom
from hamlet.utils.twocenter_utils import unfix_orbital_order

def adaptive_project_lb_in_sb(
    S_cross,
    C_lb,
    evals_lb,
    S_sb,
    lb_indices,
    eps=1e-8,
    max_expand=50,
):
    # initial index window
    if isinstance(lb_indices, slice):
        left = lb_indices.start
        right = lb_indices.stop
    else:
        idx = list(lb_indices)
        left = min(idx)
        right = max(idx) + 1

    target_dim = right - left
    N = C_lb.shape[1]

    # Löwdin orthogonalize SB once for convenience
    s, U = torch.linalg.eigh(S_sb)
    keep = s > eps
    Sinvhalf = U[:, keep] @ torch.diag(s[keep].rsqrt()) @ U[:, keep].T
    S_sb_eff = Sinvhalf.T @ S_sb @ Sinvhalf  # identity on effective SB

    # adaptive expansion to capture target_dim
    for step in range(max_expand + 1):
        l = max(0, left - step)
        r = min(N, right + step)
        cand = list(range(l, r))

        # project LB candidates into SB
        C_proj = S_cross @ C_lb[:, cand]  # shape (nsb, n_cand)
        M = C_proj.T @ S_sb @ C_proj
        rank = torch.linalg.matrix_rank(M, tol=eps)

        if rank >= target_dim:
            break

    if rank < target_dim:
        raise RuntimeError("Not enough representable LB states")

    # pick the first block of columns that achieves full rank
    # perform Löwdin orthonormalization in SB metric
    C_proj = S_cross @ C_lb[:, cand]
    M = C_proj.T @ S_sb @ C_proj
    e, V = torch.linalg.eigh(M)
    Minvhalf = V @ torch.diag(e.rsqrt()) @ V.T
    U_sb = C_proj @ Minvhalf  # orthonormal in SB metric

    # keep only the first target_dim
    U_sb = U_sb[:, :target_dim]
    kept = cand[:target_dim]

    # projected SB Hamiltonian
    H_sb_proj = S_sb @ U_sb @ torch.diag(evals_lb[kept]) @ U_sb.T @ S_sb

    return H_sb_proj, kept

import torch

def adaptive_project_lb_in_sb_greedy(
    S_cross,
    C_lb,
    evals_lb,
    S_sb,
    lb_indices,
    eps=1e-8,
    max_expand=50,
    return_U = False,
):
    # initial index window
    if isinstance(lb_indices, slice):
        left = lb_indices.start
        right = lb_indices.stop
    else:
        idx = list(lb_indices)
        left = min(idx)
        right = max(idx) + 1

    target_dim = right - left
    N = C_lb.shape[1]

    # compute representability for LB orbitals in initial window
    def representability(i):
        c_sb = S_cross @ C_lb[:, i]
        return (c_sb.T @ S_sb @ c_sb).item()

    # greedy selection
    candidates = list(range(max(0, left - max_expand), min(N, right + max_expand)))
    reps = [(i, representability(i)) for i in candidates if representability(i) > eps]
    reps.sort(key=lambda x: -x[1])  # sort descending by representability
    kept = [i for i, r in reps[:target_dim]]

    if len(kept) < target_dim:
        raise RuntimeError("Not enough representable LB states")

    # project selected LB orbitals into SB
    C_proj = S_cross @ C_lb[:, kept]  # shape (nsb, target_dim)

    # Löwdin orthonormalization in SB metric
    M = C_proj.T @ S_sb @ C_proj
    e, V = torch.linalg.eigh(M)
    e = e.clamp(min=eps)
    Minvhalf = V @ torch.diag(e.rsqrt()) @ V.T
    U_sb = C_proj @ Minvhalf  # now orthonormal in S_sb metric

    # projected SB Hamiltonian in SB AO basis
    H_sb_proj = S_sb @ U_sb @ torch.diag(evals_lb[kept]) @ U_sb.T @ S_sb

    # orthonormalized Hamiltonian in U_sb basis (ready for eigvalsh)
    H_ortho = U_sb.T @ H_sb_proj @ U_sb

    if return_U:
        return H_sb_proj, H_ortho, kept, U_sb
    else:
        return H_sb_proj, H_ortho, kept

def align_projected_eigenvectors(U_sb, S_sb, S_cross, C_lb, kept):
    """
    Align projected SB eigenvectors to LB orbitals in SB metric.
    U_sb: N_SB x N_kept (orthonormal in S_sb)
    S_sb: N_SB x N_SB
    S_cross: N_SB x N_AO
    C_lb: N_AO x N_LB
    kept: list of LB indices
    """
    # project LB orbitals into SB
    C_lb_proj = S_cross @ C_lb[:, kept]  # N_SB x N_kept

    # compute overlap in SB metric
    O = U_sb.T @ S_sb @ C_lb_proj  # N_kept x N_kept

    # SVD to find best unitary alignment
    X, _, Yt = torch.linalg.svd(O, full_matrices=True)
    U_sb_aligned = U_sb @ (Yt.T @ X.T)  # rotated to align with LB

    return U_sb_aligned


def compute_projected_dm(U_sb_aligned, kept, n_elec_total):
    """
    Compute density matrix in projected SB basis with occupations adjusted for min(kept)
    U_sb_aligned: nsb x n_kept aligned projected SB eigenvectors
    kept: list of LB indices that were selected
    n_elec_total: total number of electrons in the system
    Returns: rho_proj (nsb x nsb)
    """
    nocc = n_elec_total // 2
    start_idx = min(kept)
    nocc_proj = max(0, nocc - start_idx)

    occ = torch.zeros(U_sb_aligned.shape[1], dtype=U_sb_aligned.dtype, device=U_sb_aligned.device)
    occ[:nocc_proj] = 2.0

    rho_proj = torch.einsum('n,in,jn->ij', occ, U_sb_aligned, U_sb_aligned)
    return rho_proj


def compute_lowdin_charges_from_dm(qmdata, frames=None, dm=None, overlap=None, orthogonal=False, mol_charges=None, orbitals=None, return_ao_pops=False, pred_mode=False):
    """
    Compute Lowdin charges from density matrix in projected SB basis.
    qmdata: QuantumData object
    dm: density matrix in projected SB basis
    overlap: overlap matrix in projected SB basis
    orbitals: dictionary of orbitals
    mol_charges: list of float (default [0.0]*len(frames))
    return_ao_pops: bool
    pred_mode: bool, if False assert charge sum ~ 0
    """
    charges_all = []
    populations_all = []
    if return_ao_pops:
        ao_pops_all = []

    frames = qmdata.structures if frames is None else frames
    dm = qmdata.data['dm'] if dm is None else dm
    overlap = qmdata.data['overlap'] if overlap is None else overlap
    orbitals = qmdata.orbitals if orbitals is None else orbitals
    if mol_charges is None:
        mol_charges = [0.0] * len(frames)
    else:
        assert len(mol_charges) == len(frames)

    for ifr, (rho, frame, S) in enumerate(zip(dm, frames, overlap)):
        if not orthogonal:
            S12 = isqrtp(S)
            P_orth = S12 @ rho @ S12
        else:
            P_orth = rho

        ao_pops = torch.diag(P_orth, 0).to(qmdata.device)
        if return_ao_pops:
            ao_pops_all.append(ao_pops)
        atom_pops = torch.zeros(len(frame.numbers), device=qmdata.device)

        ao2atom, _ = build_ao2atom(qmdata, frame, orbitals=orbitals)
        for iorb, atom in enumerate(ao2atom):

            atom_pops[atom] += ao_pops[iorb]

        charges = torch.tensor(frame.numbers, device=qmdata.device) + mol_charges[ifr] - atom_pops

        if not pred_mode:
            assert torch.allclose(charges.sum(), torch.tensor([0.0], device=qmdata.device), atol=1e-4), f"expected charge sum to be 0, got {charges.sum()}"

        charges_all.append(charges)
        populations_all.append(atom_pops)

    if return_ao_pops:
        return charges_all, populations_all, ao_pops_all
    return charges_all, populations_all

def compute_dipole_from_dm(qmdata, frames=None, dm=None, overlap=None, orthogonal=False, orbitals=None, device='cpu'):
    """
    Unfixes orbital order (Condon-Shortley -> [x,y,z]) for PySCF compatibility , so pass unfixed DM 
    """
    if orthogonal:
        raise ValueError("compute_dipole_from_dm does not support orthogonal=True")
    from hamlet.data.pyscf_calculator import _instantiate_pyscf_mol
    from pyscf.dft import RKS
    dipoles = []
    frames = qmdata.structures if frames is None else frames
    dm = qmdata.data['dm'] if dm is None else dm
    overlap = qmdata.data['overlap'] if overlap is None else overlap
    orbitals = qmdata.orbitals if orbitals is None else orbitals
   

    # Unfix orbital order (Condon-Shortley -> [x,y,z]) for PySCF compatibility
    dm_unfixed = unfix_orbital_order(dm, frames, orbitals)
    for i, frame in enumerate(frames):
        mol = _instantiate_pyscf_mol(frame, basis=qmdata.basis)
        mf = RKS(mol)
        dipoles.append(torch.from_numpy(mf.dip_moment(dm=dm_unfixed[i].numpy(), unit="A.U.")))
    return torch.stack(dipoles)