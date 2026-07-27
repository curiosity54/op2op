import os, warnings 
from collections import defaultdict
from pathlib import Path    
from typing import Dict, List, Optional, Tuple, Union, NamedTuple   
import torch
from torch.utils.data import Dataset
import numpy as np 
import hickle 
import metatensor.torch as mts
from metatensor.learn import IndexedDataset, DataLoader 
from hamlet.utils.twocenter_utils import fix_orbital_order, unfix_orbital_order
from hamlet.utils.metatensor_utils import slice_tensormap_structures
from hamlet.data.pyscf_calculator import frame_to_orbital_dict
from collections import namedtuple
import xitorch 
from xitorch.linalg import symeig
import ase 
import itertools
import gc

root_dir = Path(__file__).parents[3]

def relabel_orbital_dict(d):
    """ return dict of as many invariant orbitals as in the original inout orbital dict per species"""
    new_ = {}
    for key, value in d.items():
        new_list = [[i + 1, 0, 0] for i in range(len(value))]
        new_[key] = new_list
    return new_

class QuantumData(Dataset):
    def __init__(self, 
                 frames, 
                 frame_slice: slice = slice(None),
                 path: Optional[str] = None,
                 orbitals: Optional[Dict] = None,
                 basis: Optional[str] = None, #name of the basis set
                 dimension=3,
                 ismolecule: Optional[bool] = True,
                 device: Optional[str] = "cpu",
                 data: Dict = None,
                 data_keys:List[str] = None,
                 fix_matrix_orbital_order = False,
                 kmesh: Union[List[int], List[List[int]]] = [1, 1, 1],
                 ):
        self.path = path
        if ismolecule is not None:
            self._ismolecule = ismolecule
        else: 
            self._ismolecule = not(any(f.pbc.any() for f in self.structures))
        self.dimension = dimension
        if self._ismolecule:
            self.dimension = 0
        for f in frames:
            if self.dimension == 2:
                f.pbc = [True, True, False]
                f.wrap(center = (0,0,0), eps = 1e-60)
                f.pbc = True
            elif self.dimension == 3:
                f.wrap(center = (0,0,0), eps = 1e-60)
                f.pbc = True
            elif self.dimension == 0: # Handle molecules 
                f.pbc = False    
            else:
                raise NotImplementedError('dimension must be 0, 2 or 3')
        self.frame_slice = frame_slice
        self.device = device
        self.structures = frames 
        self.kmesh = kmesh
        
        if self.dimension==0:
            assert not frames[0].pbc.any()
            self._ismolecule = True

        if orbitals is None:
            self.basis= basis if basis is not None  else 'sto-3g' # assume minimal basis 
            unique_numbers = np.unique(np.unique(np.concatenate([np.unique(f.numbers) for f in frames]))) 
            self.orbitals = frame_to_orbital_dict(ase.Atoms(numbers=unique_numbers), basis = self.basis)

        else:
            assert basis is not None
            self.orbitals = orbitals
            self.basis = basis  
        
    
        self.load_data(data_keys = data_keys, data=data, fix_matrix_orbital_order = fix_matrix_orbital_order)

    def load_data(self, path=None, data_keys = None, data = None, fix_matrix_orbital_order = False):
        if data_keys is None and data is None: 
            self.data = {}
            self.data_keys = []
        
        elif data_keys is None and data is not None:
            self.data = data
            self.data_keys = set(list(data.keys()))
            data_keys = list(data.keys())
            

        if data_keys is not None:
            self.data_keys = set(data_keys)
            try:
                for t in self.data_keys :
                    if data is not None:
                        if isinstance(data[t], list):
                            self.data[t] = [iv.to(device=self.device) for iv in self.data[t]]
                        else:
                            self.data[t] = data[t].to(device=self.device)  #ASSUME correct slice is already supplied 
                    else:
                        path = path if path is not None else self.path
                        warnings.warn(f'Loading data from {path}')
                        self.data[t] = hickle.load(path + "/{}.hickle".format(t))[self.frame_slice].to(device=self.device) 
            except Exception as e:
                warnings.warn(f'Data corresponding to key not found, {e}. Generating data')
            

        if fix_matrix_orbital_order:
            for t in self.data_keys:
                if len(self.data[t][-1].shape) == 2:  
                        warnings.warn(f'Fixing orbital order for {t}')
                        self.data[t] = fix_orbital_order(data[t], self.structures, self.orbitals)
    
    def add_data(self, data: Dict, fix_matrix_orbital_order = False, orbitals = None):

        for k, v in data.items():
            if k in self.data_keys:
                warnings.warn(f'Already found data for {k}, overwriting!')
            if fix_matrix_orbital_order and len(v[-1].shape) == 2:  
                orbitals = self.orbitals if orbitals is None else orbitals
                warnings.warn(f'Fixing orbital order for {k}')
                self.data[k] = fix_orbital_order(v, self.structures, orbitals)
                if isinstance(self.data[k], torch.Tensor):
                    self.data[k] = self.data[k].to(device=self.device)
                elif isinstance(self.data[k], list):
                    self.data[k] = [iv.to(device=self.device) for iv in self.data[k]]


            else: 
                if isinstance(v, list):
                    self.data[k] = [iv.to(device=self.device) for iv in v]
                else:
                    self.data[k] = v.to(device=self.device) 
            self.data_keys.add(k)  
    
    def add_second_basis(self, basis:str = None, orbitals:dict = None):    
        if basis is not None:
            self.second_basis = basis 
            unique_numbers = np.unique(np.unique(np.concatenate([np.unique(f.numbers) for f in self.structures]))) 
            if orbitals is None:
                self.second_orbitals = frame_to_orbital_dict(ase.Atoms(numbers=unique_numbers), basis = self.second_basis)
        if orbitals is not None:
            warnings.warn('Setting provided orbitals to QM data, pls check consistency between basis name and orbitals')
            self.second_orbitals = orbitals
       


    def get_nao(self, orbitals= None):
        orbitals = self.orbitals if orbitals is None else orbitals
        self.nao = [sum(len(orbitals[s]) for s in frame.numbers) for frame in self.structures]

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
from hamlet.utils.target_utils import get_blocks
class MLDataset():
    '''
    Transform QuantumData to a torch-compatible dataset ready for machine learning
    '''
    def __init__(
        self,
        qmdata: QuantumData,
        device: str = None,
        model_type: Optional[str] = "acdc",
        ml_item_names: Optional[Union[str, List]] = 'fock_blocks',
        ml_items_precomputed: Optional[Dict[str, torch.ScriptObject]] = None,
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
        features: Optional[torch.ScriptObject] = None,
        hypers_atom: Optional[Dict] = None,
        hypers_pair: Optional[Dict] = None,
        lcut: Optional[int] = 4,
        cutoff: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        """
            Args:
                qmdata: QuantumData instance
                ml_items_precomputed: Optional dict mapping ml_item names to precomputed TensorMaps.
                                     If provided, these items will be used instead of computing them.
            kwargs:
                train_frac
                test_frac
                val_frac

                sort_orbts
                all_pairs
                skip_symmetry
                orbitals_to_properties
                calculate_features
        """
    
        self.qmdata = qmdata 
        self.device = self.qmdata.device if device is None else device
        self.nstructs = len(self.qmdata.structures)
        self.LCUT = lcut 

        self.kwargs = kwargs
        self.cutoff = cutoff
        self.structures = self.qmdata.structures
        self.natoms_list = [len(frame) for frame in self.structures]
        
        self.sort_orbs = kwargs.get('sort_orbs', True)
        self.all_pairs = kwargs.get('all_pairs', False)
        self.skip_symmetry = kwargs.get('skip_symmetry', False)
        self.orbitals_to_properties = kwargs.get('orbitals_to_properties', True)
        self.props_to_keys = kwargs.get('props_to_keys', None)
        self.match_keys = kwargs.get('match_keys', False)
        self.matching_1 = kwargs.get('matching_1', 0)
        self.matching_2 = kwargs.get('matching_2', 1)

        self.rng = None
        if shuffle:
            self._shuffle(shuffle_seed)
        else:
            self.indices = torch.arange(self.nstructs).to(self.device)

        self.model_type = model_type  # flag to know if we are using features or not (could be a specific type of model)
        # Initialize all things to be machine learned 
        if isinstance(ml_item_names, str):
            ml_item_names = [ml_item_names]
        elif ml_item_names is None:
            ml_item_names = []

        if self.model_type =='acdc':
            self._set_features(features, hypers_atom, hypers_pair, self.LCUT, kwargs.get('calculate_features', True))
            ml_item_names.append('features')
        self.ml_item_names = ml_item_names

        self.ml_items= {}
       
        base_blocks_items = []
        tp_blocks_items = []
        invariant_items = []
        other_items = []
        
        for item_name in self.ml_item_names:
            if 'invariant' in item_name.split('_'):
                invariant_items.append(item_name)
            elif 'tp' in item_name.split('_') and 'blocks' in item_name.split('_'):
                tp_blocks_items.append(item_name)
            elif 'blocks' in item_name.split('_'):
                base_blocks_items.append(item_name)
            else:
                other_items.append(item_name)
        
        reordered_items = base_blocks_items + tp_blocks_items + invariant_items + other_items
        
        ml_items_precomputed = ml_items_precomputed if ml_items_precomputed is not None else {}
        
        for item_name in reordered_items:
            if item_name in ml_items_precomputed:
                self.ml_items[item_name] = ml_items_precomputed[item_name]
                continue
            
            if item_name == 'features':
                self.ml_items['features'] = self.features
            
            elif item_name in self.qmdata.data_keys:
                if isinstance(self.qmdata.data[item_name], list):
                    self.ml_items[item_name] = [iv.to(device=self.device) for iv in self.qmdata.data[item_name]]
                else:
                    self.ml_items[item_name] = self.qmdata.data[item_name]
            
            else:
                if 'blocks' in item_name.split('_'):
                    
                    matrix_type = item_name.split('_')[0]
                    assert matrix_type in qmdata.data_keys, f'{matrix_type} not found in QM data keys, please specify which matrix to compute blocks of'
                    
                    prefix = 'second' if item_name[:6] == 'second' else ''
                    base_blocks_name = f'{prefix}{matrix_type}_blocks'
                    needs_tp = 'tp' in item_name.split('_')
                    
                    if needs_tp and base_blocks_name in self.ml_items:
                        base_blocks = self.ml_items[base_blocks_name]
                        from hamlet.models.linear import tensor_product_matrix_blocks
                        import time
                        print(f'Computing tensor product for {item_name}...', flush=True)
                        start_time = time.time()
                        self.ml_items[item_name] = tensor_product_matrix_blocks(base_blocks, base_blocks, match_species = True, other_keys_match=['block_type'], lcut = self.LCUT)
                        print(f'Tensor product for {item_name} took {time.time() - start_time:.2f} seconds', flush=True)
                    else:
                        ORBITALS = qmdata.second_orbitals if item_name[:6] == 'second' else qmdata.orbitals
                         
                        kwargs_filtered = {k: v for k, v in kwargs.items() 
                                          if k not in ['skip_symmetry', 'orbitals_to_properties', 'sort_orbs', 'all_pairs', 'cutoff', 'orbitals']}
                        import time
                        print(f'Computing blocks for {item_name}...', flush=True)
                        start_time_blocks = time.time()
                        self.ml_items[item_name] = compute_blocks(self.qmdata, device = self.device, matrix_type = matrix_type, 
                                                                  orbitals_to_properties = self.orbitals_to_properties, sort_orbs = self.sort_orbs, 
                                                                   skip_symmetry = self.skip_symmetry, all_pairs =self.all_pairs, orbitals= ORBITALS,
                                                                   cutoff = self.cutoff,
                                                                   **kwargs_filtered
                                                                   )
                        print(f'Computing blocks for {item_name} took {time.time() - start_time_blocks:.2f} seconds', flush=True)

                        if needs_tp:
                            from hamlet.models.linear import tensor_product_matrix_blocks
                            print(f'Computing tensor product for {item_name}...', flush=True)
                            start_time_tp = time.time()
                            self.ml_items[item_name] = tensor_product_matrix_blocks(self.ml_items[item_name], self.ml_items[item_name], match_species = True, other_keys_match=['block_type'])
                            print(f'Tensor product for {item_name} took {time.time() - start_time_tp:.2f} seconds', flush=True)

                elif 'invariant' in item_name.split('_'):
                    #NOTE: BE careful if wanting to use TP to compute invariants! 
                    invariant_idx = item_name.split('_').index('invariant')
                    prefix_before_invariant = '_'.join(item_name.split('_')[:invariant_idx])
                    needs_tp = 'tp' in item_name.split('_')

                    prefix_blocks_name = f'{prefix_before_invariant}_blocks'
                    if not needs_tp and prefix_blocks_name in self.ml_items:
                        cblocks = self.ml_items[prefix_blocks_name]
                    
                    else:  
                        matrix_type = item_name.split('_')[0]  
                        tp_blocks_name = f'{matrix_type}_tp_blocks' if needs_tp else None
                        base_blocks_name = f'{matrix_type}_blocks'
                        
                        if needs_tp and tp_blocks_name in self.ml_items:
                            cblocks = self.ml_items[tp_blocks_name]
                            print('Using precomputed tensor product for invariant {item_name}')
                        elif needs_tp and base_blocks_name in self.ml_items:
                            from hamlet.models.linear import tensor_product_matrix_blocks
                            import time
                            print(f'Computing tensor product for invariant {item_name}...', flush=True)
                            start_time = time.time()
                            base_blocks = self.ml_items[base_blocks_name]
                            cblocks = tensor_product_matrix_blocks(base_blocks, base_blocks, match_species = True, other_keys_match=['block_type'])
                            del base_blocks
                            gc.collect()
                            print(f'Tensor product for invariant {item_name} took {time.time() - start_time:.2f} seconds', flush=True)
                        elif not needs_tp and base_blocks_name in self.ml_items:
                            cblocks = self.ml_items[base_blocks_name]
                        else:
                            assert matrix_type in self.qmdata.data_keys, f'{matrix_type} not found in QM data keys, please specify which matrix to compute invariants from'
                            print(f'Computing blocks for {item_name}', flush=True)
                            import time
                            start_time = time.time()
                            ORBITALS = self.qmdata.second_orbitals if item_name[:6] == 'second' else self.qmdata.orbitals
                            
                            kwargs_filtered = {k: v for k, v in kwargs.items() 
                                                if k not in ['skip_symmetry', 'orbitals_to_properties', 'sort_orbs', 'all_pairs', 'cutoff', 'orbitals']}
                            cblocks = compute_blocks(self.qmdata, device = self.device, matrix_type = matrix_type, 
                                                                        orbitals_to_properties = self.orbitals_to_properties, sort_orbs = self.sort_orbs, 
                                                                        skip_symmetry = self.skip_symmetry, all_pairs =self.all_pairs, orbitals= ORBITALS,
                                                                        cutoff = self.cutoff,
                                                                        **kwargs_filtered
                                                                        )
                            print(f'Computing blocks for {item_name} took {time.time() - start_time:.2f} seconds', flush=True)

                        if needs_tp:
                            from hamlet.models.linear import tensor_product_matrix_blocks
                            import time
                            print(f'Computing tensor product for invariant {item_name} (from scratch)...', flush=True)
                            start_time = time.time()
                            base_blocks = cblocks
                            cblocks = tensor_product_matrix_blocks(base_blocks, base_blocks, match_species = True, other_keys_match=['block_type'])
                            del base_blocks
                            gc.collect()
                            print(f'Tensor product for invariant {item_name} took {time.time() - start_time:.2f} seconds', flush=True)
                    if cblocks.device !='cpu':
                        _cblocks = cblocks.to('cpu')
                        _cblocks = _cblocks.keys_to_properties(['species_i', 'species_j', 'block_type'])
                    else:
                        _cblocks = cblocks.keys_to_properties(['species_i', 'species_j', 'block_type'])
                    del cblocks
                    gc.collect()
                     
                    self.ml_items[item_name] = mts.TensorMap(mts.Labels(['L', 'parity'], torch.tensor([[0, 1]], device = self.device)), [_cblocks.block({'L':0, 'parity':1}).to(self.device)])
                    del _cblocks
                    gc.collect()

                else: 
                    func = getattr(globals(), f'compute_{item_name}')
                    if not callable(func):  
                        raise NotImplementedError(f'Function to get {item_name} not found')
                    
                    self.ml_items[item_name] = func(self.qmdata, device = self.device, **kwargs)
    
        if self.match_keys:
            tmap1 = self.ml_items[self.matching_1]
            tmap2 = self.ml_items[self.matching_2]
            matchingkeys, index, _ = tmap1.keys.intersection_and_mapping(tmap2.keys)
            blocks = [tmap1.block(k) for k in matchingkeys]
            self.ml_items[self.matching_1] = mts.TensorMap(matchingkeys, blocks)

        for item_name in self.ml_items:
            if self.props_to_keys is not None:
                try:
                    self.ml_items[item_name] = self.ml_items[item_name].keys_to_properties(self.props_to_keys)
                except:
                   pass
        print('Generating train/validation/test splits...', flush=True)
        # Train/validation/test fractions
        self.train_frac = kwargs.get("train_frac", 0.7)
        self.val_frac = kwargs.get("val_frac", 0.2)
        self.test_frac = kwargs.get("test_frac", 0.1)

        if self.train_frac is not None:
            self._split_indices(self.train_frac, self.val_frac, self.test_frac)
            self._split_items(self.train_frac, self.val_frac, self.test_frac)


    def _shuffle(self, random_seed: int = None):
        '''
        Shuffle structure indices
        '''
        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)

        self.indices = torch.randperm(self.nstructs, generator=self.rng).to(self.device)

    def get_item(self, item_name):
            """Retrieve the instantiated target for a specific target name."""
            if item_name not in self.ml_item_names:
                raise ValueError(f"Target '{item_name}' not found in dataset.")
            return self.ml_items[item_name]
        

    
    def _split_by_structure(self, tmap: mts.TensorMap) -> mts.TensorMap:
        struct_labels = [mts.Labels(names = 'structure', 
                                     values = A.reshape(-1, 1)) for A in mts.unique_metadata(tmap, 
                                                                              axis = 'samples', 
                                                                              names = 'structure').values]
        return mts.split(tmap, 'samples', struct_labels), struct_labels

    def _split_indices(
        self,
        train_frac: float = None,
        val_frac: Optional[float] = None,
        test_frac: Optional[float] = None,
    ):
        '''
        Train/validation/test splitting
        '''

        fractions = [train_frac, val_frac, test_frac]
        defined_fractions = [f for f in fractions if f is not None]
        
        if not defined_fractions:
            # If all fractions are None, use default split
            train_frac = self.train_frac
            val_frac = self.val_frac 
            test_frac = self.test_frac 
        
        elif len(defined_fractions) == 3:
            # If all fractions are defined, check if they sum to 1
            if abs(sum(defined_fractions) - 1.0) > 1e-6:
                raise ValueError("The sum of train, validation, and test fractions must be 1.")
            self.train_frac, self.val_frac, self.test_frac = fractions
        else:
            # Handle cases where some fractions are defined
            remaining = 1.0 - sum(defined_fractions)
            undefined_count = fractions.count(None)
            
            if remaining < 0:
                raise ValueError("The sum of defined fractions must be less than or equal to 1.")
            
            # Assign defined fractions and distribute remaining among undefined
            self.train_frac = train_frac if train_frac is not None else (remaining / undefined_count if undefined_count > 0 else 0)
            self.val_frac = val_frac if val_frac is not None else (remaining / undefined_count if undefined_count > 0 else 0)
            self.test_frac = test_frac if test_frac is not None else (remaining / undefined_count if undefined_count > 0 else 0)

        # Ensure all fractions are set and sum to 1
        assert all(0 <= f <= 1 for f in [self.train_frac, self.val_frac, self.test_frac]), \
            "All fractions must be between 0 and 1."
        assert abs(sum([self.train_frac, self.val_frac, self.test_frac]) - 1.0) < 1e-6, \
            "The sum of all fractions must be 1."

        splits = [int(round(s * self.nstructs)) for s in [self.train_frac, self.val_frac, self.test_frac]]

        splits = [int(round(s * self.nstructs)) for s in [self.train_frac, self.val_frac, self.test_frac]]

        while sum(splits) != self.nstructs:
            diff = self.nstructs - sum(splits)
            splits[np.argmax(splits)] += diff
    
        self.train_idx, self.val_idx, self.test_idx = torch.split(self.indices, splits)

        # Ensure test set is not empty if test_frac > 0
        assert len(self.test_idx) > 0 if self.test_frac > 0 else True, \
            "Split indices not generated properly"

        self.train_frames = [self.structures[i] for i in self.train_idx]
        self.val_frames = [self.structures[i] for i in self.val_idx]
        self.test_frames = [self.structures[i] for i in self.test_idx]

    def _set_features(self, features, hypers_atom, hypers_pair, lcut, calc_features):
        from hamlet.features.acdc import compute_features
    
        if not calc_features:
            self.features = features
            if features is None: 
                warnings.warn('Features not computed nor set.')
        elif features is None and self.model_type == "acdc":
            assert hypers_atom is not None, "`hypers_atom` must be present when `features` is not provided."
            assert lcut is not None, f"`lcut` must be present when `features` is not provided."
            
            if hypers_pair is None:
                hypers_pair = hypers_atom
                if self.cutoff is not None: 
                    hypers_pair['cutoff']['radius'] = self.cutoff 

            self.features = compute_features(self.qmdata, 
                                                 hypers_atom, 
                                                 hypers_pair = hypers_pair, 
                                                 lcut = self.LCUT, 
                                                 all_pairs = self.all_pairs, 
                                                 device = self.device,
                                                 **self.kwargs)
                                                 

    def __len__(self):
        return self.nstructs
    
    
    
    def _split_items(self, train_frac= None, val_frac=None, test_frac=None):
        """
        Optimized version that uses mts.slice directly on original TensorMaps
        instead of splitting into per-structure lists and then joining.
        """
        if train_frac is None or val_frac is None or test_frac is None:
            if self.train_frac is None: 
                raise ValueError('`train_frac` not provided, nor set, call _split_indices first')
            train_frac = self.train_frac
            val_frac = self.val_frac
            test_frac = self.test_frac
            warnings.warn('`train_frac`, `val_frac`, or `test_frac` not provided. Using default values.')

        train_struct_labels = mts.Labels(names='structure', values=self.train_idx.reshape(-1, 1))
        val_struct_labels = mts.Labels(names='structure', values=self.val_idx.reshape(-1, 1))
        test_struct_labels = mts.Labels(names='structure', values=self.test_idx.reshape(-1, 1))

        train_dict = {'frames': [self.structures[i] for i in self.train_idx]}
        val_dict = {'frames': [self.structures[i] for i in self.val_idx]}
        test_dict = {'frames': [self.structures[i] for i in self.test_idx]}
        
        self.train_tmaps = {}
        self.val_tmaps = {}
        self.test_tmaps = {}
        
        for k, item in self.ml_items.items():
            if isinstance(item, torch.ScriptObject):
                if item._type().name() == 'TensorMap':
                    # Store the full sliced TensorMap - we'll slice by batch in collate_fn
                    self.train_tmaps[k] = mts.slice(item, axis='samples', selection=train_struct_labels)
                    self.val_tmaps[k] = mts.slice(item, axis='samples', selection=val_struct_labels)
                    self.test_tmaps[k] = mts.slice(item, axis='samples', selection=test_struct_labels)
                    train_dict[k] = [None] * len(self.train_idx)
                    val_dict[k] = [None] * len(self.val_idx)
                    test_dict[k] = [None] * len(self.test_idx)
                else:
                    split_items, _ = self._split_by_structure(item)
                    train_dict[k] = [split_items[A] for A in self.train_idx]
                    val_dict[k] = [split_items[A] for A in self.val_idx]
                    test_dict[k] = [split_items[A] for A in self.test_idx]
            elif isinstance(item, list) or isinstance(item, torch.Tensor):
                train_dict[k] = [item[i] for i in self.train_idx]
                val_dict[k] = [item[i] for i in self.val_idx]
                test_dict[k] = [item[i] for i in self.test_idx]

        self.train_dataset = IndexedDataset(sample_id=self.train_idx.tolist(), **train_dict)
        self.val_dataset = IndexedDataset(sample_id=self.val_idx.tolist(), **val_dict)
        self.test_dataset = IndexedDataset(sample_id=self.test_idx.tolist(), **test_dict)

    def get_dataloaders(self, batch_size):
        self.train_dl = DataLoader(
            self.train_dataset, 
            batch_size = batch_size, 
            collate_fn = lambda x: self.group_and_join(
                x, 
                self.train_tmaps, 
                self.train_idx,
                join_kwargs = {'different_keys': 'union', 'remove_tensor_name': True}
            )
        )
        
        self.val_dl = DataLoader(
            self.val_dataset, 
            batch_size = batch_size, 
            collate_fn = lambda x: self.group_and_join(
                x, 
                self.val_tmaps, 
                self.val_idx,
                join_kwargs = {'different_keys': 'union', 'remove_tensor_name': True}
            )
        )
        
        self.test_dl = DataLoader(
            self.test_dataset, 
            batch_size = batch_size, 
            collate_fn = lambda x: self.group_and_join(
                x, 
                self.test_tmaps, 
                self.test_idx,
                join_kwargs = {'different_keys': 'union', 'remove_tensor_name': True}
            )
        )

    
    

    def group_and_join(self,
        batch: List[NamedTuple],
        tmap_dict: dict,  
        structure_indices: torch.Tensor,  
        fields_to_join: Optional[List[str]] = None,
        join_kwargs: Optional[dict] = {'different_keys': 'union', 'sort_samples':False, 'remove_tensor_name': True},
        sort_samples: bool = False,
    ) -> NamedTuple:
        """
        Optimized version that slices TensorMaps directly from full sliced TensorMaps
        instead of joining per-structure slices. For non-TensorMap fields, uses normal group_and_join.
        """
        data = []
        names = batch[0]._fields
        if fields_to_join is None:
            fields_to_join = names
        if join_kwargs is None:
            join_kwargs = {}
        
        batch_struct_ids = [int(b.sample_id) for b in batch]
        # batch_struct_tensor = torch.tensor(batch_struct_ids, dtype=structure_indices.dtype, device = self.device)
        
        for name, field in zip(names, list(zip(*batch))):
            if name == "sample_id":  
                data.append(field)
                continue

            if name in fields_to_join:
                # If this is a TensorMap field, slice directly from the full sliced TensorMap
                if name in tmap_dict:
                    data.append(
                        slice_tensormap_structures(tmap_dict[name], batch_struct_ids, sort_samples=sort_samples)
                    )
                elif isinstance(field[0], torch.ScriptObject) and field[0]._has_method("keys_to_properties"):
                    # This should never happen - all TensorMaps should be in tmap_dict
                    raise ValueError(f"TensorMap field '{name}' not found in ticmap_dict. Available keys: {list(tmap_dict.keys())}")
                elif isinstance(field[0], torch.Tensor):
                    try:
                        data.append(torch.stack(field))
                    except RuntimeError:
                        data.append(field)
                else:
                    data.append(field)
            else:
                data.append(field)

        return namedtuple("Batch", names)(*data)     
    
    
    
    def _compute_model_metadata(self, qmdata = None, use_second_orbitals = False):
        qmdata = self.qmdata if qmdata is None else qmdata
        species_pair = np.unique([comb for frame in qmdata.structures for comb in itertools.combinations_with_replacement(np.unique(frame.numbers), 2)], axis = 0)
        max_count = defaultdict(lambda: 0)
        for species_counts in [np.unique(frame.numbers, return_counts=True) for frame in qmdata.structures]:
            for s, c in zip(*species_counts):
                max_count[s] = c if c > max_count[s] else max_count[s]
    
        key_names = ['block_type', 'species_i', 'n_i', 'l_i', 'species_j', 'n_j', 'l_j', 'L']
        if self.orbitals_to_properties:
            key_names += ['parity']
        keys = []

        orbitals = qmdata.orbitals if not use_second_orbitals else qmdata.second_orbitals
        for s1, s2 in species_pair:
            same_species = s1 == s2

            nl1 = np.unique([nlm[:2] for nlm in orbitals[s1]], axis = 0)
            nl2 = np.unique([nlm[:2] for nlm in orbitals[s2]], axis = 0)

            if same_species:
                block_types = [0] if max_count[s1] == 1 else [-1,0,1]
                orbital_list = [(a, b) for a, b in itertools.product(nl1.tolist(), nl2.tolist()) if a <= b]
            else: 
                if s1 > s2:
                    continue
                block_types = [2]
                orbital_list = itertools.product(nl1, nl2)
            
            for block_type in block_types:
                for (n1, l1), (n2, l2) in orbital_list:
                    for L in range(abs(l1-l2), l1+l2+1):
                        sigma = (-1)**(l1+l2+L)

                        if s1 == s2 and n1 == n2 and l1 == l2:
                            if ((sigma == -1 and block_type in (0, 1)) or (sigma == 1 and block_type == -1)) and not self.skip_symmetry:
                                continue

                        if self.orbitals_to_properties:
                            key = block_type, s1, n1, l1, s2, n2, l2, L, sigma
                        else:
                            key = block_type, s1, n1, l1, s2, n2, l2, L

                        keys.append(key)
                        
        blocks = []
        dummy_label = mts.Labels(['dummy'], torch.tensor([[0]], device = qmdata.device))
        for k in keys:
            blocks.append(
                mts.TensorBlock(
                    samples = dummy_label,
                    properties = dummy_label,
                    components = [dummy_label],
                    values = torch.zeros((1, 1, 1), device = qmdata.device) 
                )
            )

        self.model_metadata = mts.TensorMap(mts.Labels(key_names, torch.tensor(keys, device = qmdata.device)), blocks)
        if self.orbitals_to_properties:
            self.model_metadata = self.model_metadata.keys_to_properties(['n_i', 'l_i', 'n_j', 'l_j'])
        self.model_metadata = mts.sort(self.model_metadata)

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

def compute_homolumoidx(qmdata: QuantumData,
                        device = 'cpu',
                        **kwargs):
    """
    kwargs:
        offset : number of electrons to subtract from the total number of electrons
        structures : list of ase.Atoms objects (uses qmdata.structures if not provided)
        return_lumo : bool, whether to return the LUMO index or not
    """

    offset = kwargs.get('offset', 0)
    structures = kwargs.get('structures', qmdata.structures)
    return_lumo = kwargs.get('return_lumo', False)
    if isinstance(offset, int):
        offset = [offset]*len(structures)
    elif isinstance(offset, list):
        assert len(offset) == len(structures)
    else:
        raise ValueError('offset must be an integer or a list of ints')
    homoidx = []
    for ifr, frame in enumerate(structures):
        nelec = sum(frame.numbers) - offset[ifr]
        homo = nelec//2 -1 # zero based indexing!
        # lumo = homo + 1
        homoidx.append(homo)

    if return_lumo:
        lumoidx = torch.tensor(homoidx, device = device)+1 
        return homoidx, list(lumoidx)  
    
    return homoidx

def compute_projector_slice(qmdata: QuantumData,
                            orbitals_in: str = None,
                            orbitals_out: str = None,
                            device = 'cpu',
                            align_zero=False,
                            **kwargs):
    """ slices of eigvalues that will be matched from the original matrix to a new matrix
    orbitals_in: orbitals to use for the original matrices, uses provided obitals, over the ones in qmdata
    orbitals_out: orbitals to use to compute the shape of the new matrices, uses provided obitals, over qmdata.secondorbitals
    kwargs: 
            new_shapes : [nout1, nout2, ...] list of new sizes for the matrices 
            align_zero : bool, whether to align the zero index of the new matrices to the zero index of the original matrices
                        if False, centers the new matrices at the HOMO index of the original matrices
    Returns:
        slices : list of slices for the new matrices
    """
    orbitals_in = qmdata.orbitals if orbitals_in is None else orbitals_in
    orbitals_out = qmdata.second_orbitals if orbitals_out is None else orbitals_out 
    homoidx = compute_homolumoidx(qmdata, device = device,  **kwargs) # doesnt need basis
    original_shapes = compute_nao(qmdata, device, orbitals = orbitals_in, **kwargs) # doesnt need basis  
    new_shapes = kwargs.get('new_shapes', None)
    if new_shapes is None: 
        new_shapes = compute_nao(qmdata, device, orbitals = orbitals_out, **kwargs)
    
    slices = []

    for oshape, homo, nshape in zip(original_shapes, homoidx, new_shapes):
        if align_zero:
            # match the first N eigvals of the original matrix to the first N eigvals of the new matrix
            slices.append(slice(0, nshape))
        else:
            # center the new matrices at the HOMO index of the original matrices
            start = max(0, homo -  (nshape - 1) // 2 )
            end = min(oshape, start + nshape+1)
            start = max(1, end- nshape)
            assert end-start == nshape
            slices.append(slice(start, end))  
    return slices 

def compute_nao(qmdata, 
                device = 'cpu', 
                 **kwargs, 
                ): 
        """ 
        kwargs: 
            orbitals : 
            structures : 
        """
        orbitals = kwargs.get('orbitals', qmdata.orbitals)
        structures = kwargs.get('structures', qmdata.structures)
        return [sum(len(orbitals[s]) for s in frame.numbers) for frame in structures]

def make_spd(S, eps=1e-8):
    w, _ = torch.linalg.eigh(S)
    shift = max(torch.tensor(0., dtype=S.dtype, device=S.device),
                -w.min() + eps)
    S_spd = S + shift * torch.eye(S.size(0), device=S.device, dtype=S.dtype)
    return S_spd

def compute_eigenvalue(qmdata: QuantumData, 
                       device = None,
                        **kwargs): 
    """ 
    kwargs: 
        subset_eigenvalues : return this slice of eigenvalues  
        return_eigenvector : whether to return eigvectrs or not
        use_overlap: wherether to use overlaps and solve the generalized eigval eqn, True by default
        overlaps: uses qmdata overlap by default
        make_overlap_spd: if overlap is not semipositive definite, call make_spd
    NOTE: implemented only for molecules, not periodic systems
    """
    device = qmdata.device if device is None else device
    # in principle subset could be a list of slices, or the number of levels defining the active space above and below HOMO? 
    subset_eigenvalues = kwargs.get('subset_eigenvalues', slice(None))
    return_eigenvector = kwargs.get('return_eigenvector', False)
    matrices = kwargs.get('matrices', qmdata.data['fock'])
    use_overlap = kwargs.get('use_overlap', True)
    overlaps = kwargs.get('overlaps', qmdata.data['overlap']) if use_overlap else None 
    make_overlap_spd = kwargs.get('make_overlap_spd', False)
    if not isinstance(subset_eigenvalues, list):
        subset_eigenvalues = [subset_eigenvalues]*len(matrices)
    if qmdata.dimension>0: 
        raise NotImplementedError
    eigenvalues = []
    eigenvectors = []
    for ifr, A in enumerate(matrices):
        if use_overlap:
            s = overlaps[ifr]
        if make_overlap_spd:
            s = make_spd(s,eps=1e-5)
        Ax = xitorch.LinearOperator.m(A)
        Mx = xitorch.LinearOperator.m(s) if use_overlap else None 
        eigvals, eigvecs = symeig(Ax, M = Mx)
        eigenvalues.append(eigvals[subset_eigenvalues[ifr]])    
        eigenvectors.append(eigvecs[:, subset_eigenvalues[ifr]])
    if return_eigenvector:
        return eigenvalues, eigenvectors
    
    return eigenvalues

def compute_blocks(qmdata: QuantumData, 
                   device = 'cpu',
                   matrix_type = None,
                   orbitals_to_properties = False,
                   sort_orbs = True,
                   skip_symmetry = False,
                   all_pairs = False,
                   orbitals = None,
                   cutoff = None, 
                   **kwargs):
    return get_blocks(qmdata, target = matrix_type,
                       orbitals = orbitals,
                        orbitals_to_properties=orbitals_to_properties, sort_orbs=sort_orbs, skip_symmetry=skip_symmetry, all_pairs=all_pairs, 
                        device = device,
                        cutoff = cutoff,
                        **kwargs)

def compute_dm_nelec(qmdata, frames=None, fock=None, overlap=None, orthogonal=False):
    """
    kwargs:
        frames : list of ase.Atoms objects (uses qmdata.structures if not provided)
        fock : list of torch.Tensor objects (uses qmdata.data['fock'] if not provided)
        overlap : list of torch.Tensor objects (uses qmdata.data['overlap'] if not provided)
        orthogonal : bool, whether fock is already orthogonalized or not. If True, overlap is not used
    Returns: 
        Tr(rho) if orthogonal is False, otherwise Tr(rho S)
    """
    if fock is None: 
        fock = qmdata.data['fock']
    if overlap is None: 
        overlap = qmdata.data['overlap']
    if frames is None: 
        frames = qmdata.structures
    if not orthogonal:
        eigenvalues, eigenvectors = compute_eigenvalue(qmdata, matrices = fock, overlaps = overlap, 
                                                   return_eigenvector = True, use_overlap=True, device = qmdata.device)
    else:
        eigenvalues, eigenvectors = compute_eigenvalue(qmdata, matrices = fock, 
                                                   return_eigenvector = True, use_overlap=False, device = qmdata.device)

    nelec_through_dm = []
    nelec_actual = []
    dm = []
    for (evec, frame, S) in zip( eigenvectors, frames, overlap):
        nelec = sum(frame.numbers) # NO CHARGE 
    
        occ = torch.tensor([2.0 if i < nelec//2 else 0.0 for i in range(evec.shape[0])]).to(qmdata.device)
        rho = torch.einsum('n,in,jn->ij', occ, evec, evec)
        dm.append(rho)
        
        nelec_through_dm .append(torch.trace(rho@S)) if not orthogonal else nelec_through_dm .append(torch.trace(rho))
        nelec_actual.append(nelec)

    return dm, nelec_through_dm, nelec_actual



os.environ["PYSCFAD_BACKEND"] = "torch"

def compute_dipole(qmdata: QuantumData, 
                         device = 'cpu',
                         orthogonal = False,
                         unfix_matrix_orbital_order = True,
                         **kwargs):
    """
    UNFIXES orbital order of the matrices if not provided, else please provide the unfixed matrices
    kwargs:
        matrices : list of torch.Tensor objects (uses qmdata.data['fock'] if not provided)
        use_overlap : bool, whether to use the overlap matrix or not
        overlaps : list of torch.Tensor objects (uses qmdata.data['overlap'] if not provided)
        frames : list of ase.Atoms objects (uses qmdata.structures if not provided)
        orthogonal : bool, whether the input matrices are already orthogonalized or not
    """
    from hamlet.data.pyscf_calculator import _instantiate_pyscf_mol
    from pyscfad import ops
    from pyscfad.ml.scf import hf
    frames = kwargs.get('frames', qmdata.structures)
    orbitals = kwargs.get('orbitals', qmdata.orbitals)
    focks = kwargs.get('matrices', unfix_orbital_order(qmdata.data['fock'], frames, orbitals) if unfix_matrix_orbital_order else qmdata.data['fock'])
    use_overlap = kwargs.get('use_overlap', True)
    overlaps = kwargs.get('overlaps', unfix_orbital_order(qmdata.data['overlap'], frames, orbitals) if unfix_matrix_orbital_order else qmdata.data['overlap']) if use_overlap else None 
    
    dipoles = []
    for i, frame in enumerate(frames):
        mol = _instantiate_pyscf_mol(frame, basis = qmdata.basis)
        mf = hf.SCF(mol)
        focks[i] = torch.autograd.Variable(focks[i].type(torch.float64),
                                           requires_grad=True)
        # - if orthogonal is True: matrices are already orthogonalized; do NOT use overlap in eigensolve
        # - else: use overlap when available/requested
        if orthogonal:
            mo_energy, mo_coeff = mf.eig(focks[i], None)
        else:
            if use_overlap and overlaps is not None:
                mo_energy, mo_coeff = mf.eig(focks[i], overlaps[i])
            else:
                warnings.warn("Dipole called with use overlap but Overlap matrix not provided, using identity matrix for eigenproblem")
                mo_energy, mo_coeff = mf.eig(focks[i], None)
        mo_occ = mf.get_occ(mo_energy)  
        mo_occ = torch.as_tensor(mo_occ)
        if orthogonal:
            # Back-transform eigenvectors from orthonormal basis to AO and build AO density
            if use_overlap and overlaps is not None:
                S = overlaps[i]
            else:
                S = torch.as_tensor(mol.intor("int1e_ovlp"))
            mo_coeff_ao = isqrtm(S) @ mo_coeff
            dm1 = mo_coeff_ao @ torch.diag(mo_occ) @ mo_coeff_ao.T
        else:
            dm1 = mf.make_rdm1(mo_coeff, mo_occ)
        dip = mf.dip_moment(dm=dm1, unit="A.U.")

        dipoles.append(dip)
    
    return torch.stack(dipoles)

def compute_dipole_ml(qmdata: QuantumData, 
                         device = 'cpu',
                         orthogonal = False,
                         **kwargs):
    """
     UNFIXES orbital order ---(pass unfixed matrices)
    kwargs:
        matrices : list of torch.Tensor objects (uses qmdata.data['fock'] if not provided)
        use_overlap : bool, whether to use the overlap matrix or not
        overlaps : list of torch.Tensor objects (uses qmdata.data['overlap'] if not provided)
        frames : list of ase.Atoms objects (uses qmdata.structures if not provided)
        orthogonal : bool, whether the input matrices are already orthogonalized or not
    """
    from hamlet.data.pyscf_calculator import _instantiate_pyscf_mol
    from pyscfad.ml.scf import hf
    frames = kwargs.get('frames', qmdata.structures)
    orbitals = kwargs.get('orbitals', qmdata.orbitals)
    focks = kwargs.get('matrices', qmdata.data['fock'])
    use_overlap = kwargs.get('use_overlap', True)
    overlaps = kwargs.get('overlaps',qmdata.data['overlap'] ) if use_overlap else None 
    dm = kwargs.get('dm', None)
    use_dm = kwargs.get('use_dm', False)
    dipoles = []

    if not use_dm or (use_dm and dm is None):
        warnings.warn("Using fock matrices to compute dipole moments")
        focks = unfix_orbital_order(focks, frames, orbitals)
        overlaps = unfix_orbital_order(overlaps, frames, orbitals) if overlaps is not None else None
        dm, _, _ = compute_dm_nelec(qmdata, list(frames), fock=focks, overlap=overlaps, orthogonal=orthogonal)
    
    elif orthogonal and overlaps is not None:
        overlaps = unfix_orbital_order(overlaps, frames, orbitals)

    assert len(dm) == len(frames)

    for  i, frame in enumerate(frames):
        mol = _instantiate_pyscf_mol(frame, basis = qmdata.basis)
        mf = hf.SCF(mol)
        dm_ao = dm[i]
        if orthogonal:
            S = overlaps[i] if overlaps is not None else torch.as_tensor(mol.intor("int1e_ovlp"))
            S12inv = isqrtm(S)
            dm_ao = S12inv @ dm[i] @ S12inv
        dipoles.append(mf.dip_moment(dm=dm_ao, unit="A.U."))
    return torch.stack(dipoles)

def compute_polarizability(qmdata: QuantumData, 
                         device = 'cpu',
                         **kwargs):

    from hamlet.data.pyscf_calculator import _instantiate_pyscf_mol
    from pyscfad import ops
    from pyscfad.ml.scf import hf
    from torch.autograd.functional import jacobian
    from pyscfad import numpy as pynp
    focks = kwargs.get('matrices', qmdata.data['fock'])
    use_overlap = kwargs.get('use_overlap')
    overlaps = kwargs.get('overlaps', qmdata.data['overlap']) if use_overlap else None 
    frames = kwargs.get('frames', qmdata.structures)


    polarizability = []
    for i, frame in enumerate(frames):
        mol = _instantiate_pyscf_mol(frame)
        ao_dip = mol.intor("int1e_r", comp=3)
        ao_dip = torch.as_tensor(ao_dip)
        mf = hf.SCF(mol)
        fock = focks[i]
        if overlaps is None:
            ovlp = torch.from_numpy(mol.intor("int1e_ovlp"))
            fock = torch.einsum("ij,jk,kl->il", isqrtp(ovlp),
                                fock, isqrtp(ovlp))
        else:
            ovlp = overlaps[i]

        def apply_perturb(E):
            p_fock = fock + pynp.einsum("x,xij->ij", E, ao_dip)
            mo_energy, mo_coeff = mf.eig(p_fock, ovlp)
            mo_occ = mf.get_occ(mo_energy)
            mo_occ = torch.as_tensor(mo_occ)
            dm1 = mf.make_rdm1(mo_coeff, mo_occ)
            dip = mf.dip_moment(dm=dm1, unit="A.U.")
            return dip

        E = torch.zeros((3,), dtype=float)
        pol = jacobian(apply_perturb, E)
        polarizability.append(pol)

    return torch.stack(polarizability)
    pass    



#---- utils for compression to different sizes ----


def find_homo_index(eigvals, occupations=None, fermi_level=None):
    """
    Find the HOMO index given eigenvalues and either occupations or Fermi level.
    If occupations is given, HOMO is the highest occupied orbital.
    If Fermi level is given, HOMO is the largest eigenvalue <= Fermi level.
    """
    eigvals = np.array(eigvals)
    
    if occupations is not None:
        occ = np.array(occupations)
        occ_mask = occ > 1e-8
        return np.where(occ_mask)[0][-1]  
    
    elif fermi_level is not None:
        return np.where(eigvals <= fermi_level)[0][-1]
    
    else:
        raise ValueError("Provide either occupations or fermi_level")

def align_homo(eigvals_list, occupations_list=None, fermi_levels=None):
    """
    Shift eigenvalues so HOMO is at index 0 for all sets.
    Returns list of (eigvals, homo_idx).
    """
    aligned = []
    for i, eigvals in enumerate(eigvals_list):
        if occupations_list is not None:
            homo_idx = find_homo_index(eigvals, occupations=occupations_list[i])
        elif fermi_levels is not None:
            homo_idx = find_homo_index(eigvals, fermi_level=fermi_levels[i])
        else:
            raise ValueError("Need occupations_list or fermi_levels")

        aligned.append((eigvals, homo_idx))
    return aligned

def slice_around_homo(eigvals, homo_idx, window=5):
    """
    Return slice of eigenvalues centered around HOMO index.
    """
    start = max(homo_idx - window, 0)
    end = min(homo_idx + window + 1, len(eigvals))
    return eigvals[start:end]

def project_around_homo(H, eigvals, eigvecs, homo_idx, window=5):
    """
    Slice the Hamiltonian/projector around HOMO using eigenvectors.
    """
    start = max(homo_idx - window, 0)
    end = min(homo_idx + window + 1, len(eigvals))
    return compute_projector_slice(H, eigvecs, start, end)


def isqrtm(A: torch.Tensor) -> torch.Tensor:
    eva, eve = torch.linalg.eigh(A)
    idx = eva > 1e-15
    return eve[:, idx] @ torch.diag(eva[idx] ** (-0.5)) @ eve[:, idx].T

def isqrtp(A):
    eva, eve = torch.linalg.eigh(A)
    idx = eva > 1e-15
    return eve[:, idx] @ torch.diag(eva[idx] ** (0.5)) @ eve[:, idx].T




def build_ao2atom(qmdata, frame, orbitals = None):
    orbitals = orbitals if orbitals is not None else qmdata.orbitals
    ao2atom = []
    prev_idx = 0
    ao_masks = []

    for atom_idx, Z in enumerate(frame.numbers):
        n_orb = len(orbitals[Z])  # how many AOs for this species
        ao_range = list(range(prev_idx, prev_idx + n_orb))
        ao2atom.extend([atom_idx] * n_orb)
        ao_masks.append(ao_range)
        prev_idx += n_orb

    return torch.tensor(ao2atom, device = qmdata.device), ao_masks

def compute_lowdin_charges(qmdata, frames=None, fock=None, overlap=None, orthogonal=False, mol_charges = None, orbitals = None, return_ao_pops = False, pred_mode = False):
    """
    kwargs:
        frames : list of ase.Atoms objects (uses qmdata.structures if not provided)
        fock : list of torch.Tensor objects (uses qmdata.data['fock'] if not provided)
        overlap : list of torch.Tensor objects (uses qmdata.data['overlap'] if not provided)
        orthogonal : bool, whether fock is already orthogonalized or not. If True, overlap is not used
        mol_charges : list of float, charges of the molecules
        orbitals : dict, orbitals of the atoms
    """
    charges_all = []
    populations_all = []
    if return_ao_pops:
        ao_pops_all = []
    fock = qmdata.data['fock'] if fock is None else fock
    overlap = qmdata.data['overlap'] if overlap is None else overlap    
    frames = qmdata.structures if frames is None else frames
    orbitals = qmdata.orbitals if orbitals is None else orbitals
    if mol_charges is None: 
        mol_charges = [0.0]*len(frames)
    else:
        assert len(mol_charges) == len(frames)
    

    
    dm, nelec_computed, nelec_actual = compute_dm_nelec(
        qmdata, frames=frames, fock=fock, overlap=overlap, orthogonal=orthogonal
    )

    for ifr, (rho, frame, S) in enumerate(zip(dm, frames, overlap)):
        nelec = nelec_computed[ifr]
        if not orthogonal: 
            S12 = isqrtp(S)    
            P_orth = S12 @ rho @ S12
        else: 
            P_orth = rho

        ao_pops = torch.diag(P_orth, 0).to(qmdata.device) # (shape= fock.shape[0] = \sum_{nat} \sum_{norbs_per atom}
        if return_ao_pops:
           ao_pops_all.append(ao_pops) 
        atom_pops = torch.zeros(len(frame.numbers), device = qmdata.device) 
        
        
        ao2atom, _ = build_ao2atom(qmdata, frame, orbitals = orbitals)
        for iorb, atom in enumerate(ao2atom):
            atom_pops[atom] += ao_pops[iorb]

        charges = torch.tensor(frame.numbers, device = qmdata.device) + mol_charges[ifr] - atom_pops

        if not pred_mode: # not passing predicted matrices, so we can check consistency
            assert torch.allclose(atom_pops.sum(), torch.tensor(float(nelec), device = qmdata.device), atol=1e-4)
            assert torch.allclose(charges.sum(), torch.tensor([0.0], device = qmdata.device), atol=1e-4)

        charges_all.append(charges)
        populations_all.append(atom_pops)
    if return_ao_pops:
        return charges_all, populations_all, ao_pops_all

    return charges_all, populations_all

def compute_lowdin_charges_from_dm(qmdata, frames=None, dm=None, overlap=None, orthogonal=False, mol_charges = None, orbitals = None, return_ao_pops = False, pred_mode = False):
    charges_all = []
    populations_all = []
    if return_ao_pops:
        ao_pops_all = []
   
    
    frames = qmdata.structures if frames is None else frames
    dm = qmdata.data['dm'] if dm is None else dm
    overlap = qmdata.data['overlap'] if overlap is None else overlap    
    orbitals = qmdata.orbitals if orbitals is None else orbitals
    if mol_charges is None: 
        mol_charges = [0.0]*len(frames)
    else:
        assert len(mol_charges) == len(frames)
    for ifr, (rho, frame, S) in enumerate(zip(dm, frames, overlap)):
        if not orthogonal: 
            S12 = isqrtp(S)    
            P_orth = S12 @ rho @ S12
        else: 
            P_orth = rho
        ao_pops = torch.diag(P_orth, 0).to(qmdata.device) # (shape= fock.shape[0] = \sum_{nat} \sum_{norbs_per atom}
        if return_ao_pops:
           ao_pops_all.append(ao_pops) 
        atom_pops = torch.zeros(len(frame.numbers), device = qmdata.device) 
        
        
        ao2atom, _ = build_ao2atom(qmdata, frame, orbitals = orbitals)
        for iorb, atom in enumerate(ao2atom):
            atom_pops[atom] += ao_pops[iorb]

        charges = torch.tensor(frame.numbers, device = qmdata.device) + mol_charges[ifr] - atom_pops

        if not pred_mode: # not passing predicted matrices, so we can check consistency
            assert torch.allclose(charges.sum(), torch.tensor([0.0], device = qmdata.device), atol=1e-4)

        charges_all.append(charges) 
        populations_all.append(atom_pops)
    if return_ao_pops:
        return charges_all, populations_all, ao_pops_all
    return charges_all, populations_all





def get_matrix_size_in_basis(qmdata, frames = None, basis_string:str = None):
    frames = qmdata.structures if frames is None else frames
    unique_numbers = np.unique(np.unique(np.concatenate([np.unique(f.numbers) for f in qmdata.structures])))
    orbitals = frame_to_orbital_dict(ase.Atoms(numbers=unique_numbers), basis = basis_string)
    matrix_sizes = compute_nao(qmdata, orbitals = orbitals)
    return orbitals, matrix_sizes

