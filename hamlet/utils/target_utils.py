from typing import Optional, Union, TYPE_CHECKING
import torch
import metatensor.torch as mts

if TYPE_CHECKING:
    from hamlet.data.dataset import QuantumData
from hamlet.utils.twocenter_utils import matrix_to_blocks
from hamlet.utils.twocenter_utils import _to_coupled_basis
import e3nn 

def p_int2str(l):
    return "o" if l%2==1 else "e"

def p_str2int(s):
    return 1 if s=='e' else -1

def parity_rules(l1, l2, s1 = 1, s2=1):
    """
    Determine the resulting parity based on two FIRST-ORDER angular momentum quantum numbers l1 and l2 
    (otherwise we need to account for intermediate parities s1,s2 TODO).
    - If either l1 or l2 is odd, the result is 'o'.
    - If both are even, the result is 'e'.
    """
    # Compute parities of l1 and l2
    p1 = l1 % 2
    p2 = l2 % 2

    return "o" if p1 + p2 == 1 else "e"

def e3nn_parity_to_int(s, l):
    """
    The integer we want to return is 1 if l1+l2+L is even and -1 otherwise 
    i.e. 1 if polar tensor, -1 for pseudotensor 
    """
    if s == "o" and l % 2 == 0 or s == "e" and l % 2 == 1:
        return -1
    else:
        return 1


def fix_e3nn_convention(target_tmap, ztoy = True):
    """E3NN works in the basis of L^2 and Ly (unlike the usual L^2 and Lz).
    This means, we need to fix the convention of each e3nn output block from xyz -> yzx
    Following https://docs.e3nn.org/en/stable/guide/change_of_basis.html we can use the following matrices to change the basis
    """
    R_ztoy =torch.tensor([
        [0., 0., 1.],  # x = original x in Lz convention
        [1., 0., 0.],  # y = original y in Lz convention
        [0., 1., 0.]   # z = original z in Lz convention
    ], dtype=torch.float64)
    # R_ztoy  implements yzx - > x y z
    
    R_ytoz =torch.tensor([
        [0., 1., 0.],  
        [0., 0., 1.],  
        [1., 0., 0.]   
    ], dtype=torch.float64)
    # R_ytoz implements xyz-> yzx , R_ytoz  = R_ztoy^{-1} 

    R = R_ztoy if ztoy else R_ytoz
    
    fixed_blocks = []
    for k, b in target_tmap.items():
        l1 = k['l_1']
        l2 = k['l_2']
        Dl1 = e3nn.o3.Irreps(str(l1)+p_int2str(l1)).D_from_matrix(R)
        Dl2 = e3nn.o3.Irreps(str(l2)+p_int2str(l2)).D_from_matrix(R)
        fixed_blocks.append( mts.TensorBlock(samples = b.samples, components = b.components, properties = b.properties, 
                                        values = torch.einsum("ij, sjkp, kl -> silp", Dl1, b.values, Dl2.T))
                        )
    return mts.TensorMap(target_tmap.keys, fixed_blocks)

def get_blocks(dataset: "QuantumData",
                cutoff: Optional[Union[int,float,None]] = None, 
                target: Optional[str] = 'fock', 
                all_pairs: Optional[bool] = False, 
                sort_orbs: Optional[bool] = True,
                skip_symmetry: Optional[bool] = False,
                device: Optional[str] = "cpu", 
                matrix = None,
                orbitals = None,
                orbitals_to_properties = False,
                return_uncoupled = False,
                translations = True,
                fix_e3nn = False,
                ztoy = True, 
                frames = None,
                **kwargs,
                ):
    """ if both target and matrix are set, the provided matrix is used for computing blocks """
    ORBITALS  = dataset.orbitals if orbitals is None else orbitals    
    matrices = matrix if matrix is not None else dataset.data[target]
    frames = dataset.structures if frames is None else frames
    blocks = matrix_to_blocks(matrices, ORBITALS, frames, device = device, cutoff = cutoff, all_pairs = all_pairs, sort_orbs = sort_orbs)
    if fix_e3nn:
        blocks = fix_e3nn_convention(blocks, ztoy = ztoy)
    coupled_blocks = _to_coupled_basis(blocks, skip_symmetry = skip_symmetry, device = device, translations = translations)
    if dataset._ismolecule:
        blocks = mts.remove_dimension(mts.remove_dimension(mts.remove_dimension(blocks, "samples",'cell_shift_a'), "samples",'cell_shift_b'), "samples",'cell_shift_c')
        coupled_blocks = mts.remove_dimension(mts.remove_dimension(mts.remove_dimension(coupled_blocks, "samples",'cell_shift_a'), "samples",'cell_shift_b'), "samples",'cell_shift_c')
    
    keys = []
    tblocks= []
    for k, b in coupled_blocks.items(): 
        li, lj, L = k['l_1'], k['l_2'], k['L']
        inversion_sigma = (-1) ** (li + lj + L)
        keys.append(torch.cat((k.values, torch.tensor([inversion_sigma]))))
        tblocks.append(b.copy().to(device = device))
    coupled_blocks = mts.TensorMap(mts.Labels(k.names+['parity'], torch.stack(keys).to(device = device)), tblocks)
    if orbitals_to_properties:
        coupled_blocks = coupled_blocks.keys_to_properties(['n_1', 'l_1',  'n_2','l_2'])


    if return_uncoupled:   
        return mts.sort(blocks), mts.sort(coupled_blocks)
    else:
        return mts.sort(coupled_blocks)
