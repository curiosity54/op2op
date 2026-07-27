import hickle
import pyscf
import torch 
import numpy as np
from ase.io import read
from hamlet.data.pyscf_calculator import calculator

data_path = 'data/qm7'
filename = 'qm7.xyz'
frames = read(f'{data_path}/{filename}', ':1000')
basis = 'def2-tzvp'
MAX_CYCLE = 300 

pyscfcalc = calculator(path = None,
                       structures = frames,
                       target = ['density', 'fock', 'overlap', 'vext'], # 'hcore'] 
                       dft = True, 

                       )

pyscfcalc.calculate(basis_set = basis, 
                    max_cycle = MAX_CYCLE, 
                    verbose = 4, 
                    conv_tol = 1e-8, 
                    conv_tol_grad = 1e-8
                    )
print('done')


hickle.dump([torch.from_numpy(x) for x in pyscfcalc.results['vext']], f'{data_path}/{basis}/vext.hickle')
hickle.dump([torch.from_numpy(x) for x in pyscfcalc.results['fock']], f'{data_path}/{basis}/fock.hickle')
hickle.dump([torch.from_numpy(x) for x in pyscfcalc.results['density']], f'{data_path}/{basis}/dm.hickle')
hickle.dump([torch.from_numpy(x) for x in pyscfcalc.results['overlap']], f'{data_path}/{basis}/overlap.hickle')
# hickle.dump([torch.from_numpy(x) for x in pyscfcalc.results['hcore']], f'{data_path}/{basis}/hcore.hickle')
