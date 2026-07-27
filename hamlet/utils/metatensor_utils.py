import numpy as np
import torch
import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap
from typing import Dict



class TensorBuilder:
    def __init__(self, key_names, sample_names, component_names, property_names, device = 'cpu'):
        self._key_names = key_names
        self.blocks = {}

        self._sample_names = sample_names
        self._component_names = component_names
        self._property_names = property_names

        self.device = device

    def add_block(
        self, key, gradient_samples=None, *, samples=None, components, properties=None
    ):
        if samples is None and properties is None:
            raise Exception("can not have both samples & properties unset")

        if samples is not None and properties is not None:
            raise Exception("can not have both samples & properties set")

        if samples is not None:
            if isinstance(samples, torch.ScriptObject):
                if samples._type().name() == "Labels":
                    samples = samples.values.reshape(samples.shape[0], -1)
            samples = Labels(self._sample_names, samples)

        if gradient_samples is not None:
            if not isinstance(gradient_samples, torch.ScriptObject):
                if gradient_samples._type().name() == "Labels":
                    raise Exception("must pass gradient samples for the moment")

        if all([isinstance(component, torch.ScriptObject) for component in components]):
            components = [component.values.reshape(components.shape[0], -1) for component in components]
        
        components_label = []
        for names, values in zip(self._component_names, components):
            components_label.append(Labels(names, values.to(self.device)))
        components = components_label

        if properties is not None:
            if isinstance(properties, torch.ScriptObject):
                if properties._type().name() == "Labels":
                    properties = properties.view(dtype = torch.int32).reshape(properties.shape[0], -1)
            elif isinstance(properties, np.ndarray):
                properties = torch.from_numpy(properties)
            elif isinstance(properties, list):
                properties = torch.tensor(list)

            properties = Labels(self._property_names, properties.to(device = self.device))

        if properties is not None:
            block = TensorBuilderPerSamples(properties, components, self._sample_names, gradient_samples, device = self.device)

        if samples is not None:
            block = TensorBuilderPerProperties(samples, components, self._property_names, gradient_samples, device = self.device)

        self.blocks[key] = block
        return block

    def build(self):

        keys = Labels(self._key_names, torch.tensor(list(self.blocks.keys()), dtype = torch.int32)).to(device = self.device)

        blocks = []
        for block in self.blocks.values():
            if isinstance(block, torch.ScriptObject):
                if block._type().name() == "TensorBlock":
                    blocks.append(block)
            elif isinstance(block, TensorBuilderPerProperties):
                blocks.append(block.build())
            elif isinstance(block, TensorBuilderPerSamples):
                blocks.append(block.build())
            else:
                Exception("Invalid block type")

        self.blocks = {}
        return TensorMap(keys, blocks)


class TensorBuilderPerSamples:
    def __init__(self, properties, components, sample_names, gradient_samples=None, device = 'cpu'):

        assert isinstance(properties, torch.ScriptObject) and properties._type().name() == "Labels"
        assert all([(isinstance(component, torch.ScriptObject) and component._type().name() == "Labels") for component in components])
        assert (gradient_samples is None) or (isinstance(gradient_samples, torch.ScriptObject) and gradient_samples._type().name() == "Labels")

        self._gradient_samples = gradient_samples
        self._properties = properties
        self._components = components

        self._sample_names = sample_names
        self._samples = []

        self._data = []
        self._gradient_data = []
        self.device = device

    def add_samples(self, labels, data, gradient=None):

        if not isinstance(data, torch.Tensor):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).to(device=self.device)
            assert isinstance(data, torch.Tensor), "Data must be numpy.ndarray or torch.tensor."
        assert data.shape[-1] == self._properties.values.shape[0], "The property dimension of data does not match."
        
        for i in range(len(self._components)):
            assert data.shape[i + 1] == self._components[i].values.shape[0], f"The {i}-th component dimension of data does not match."

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).to(dtype = torch.int32, device = self.device)
        elif isinstance(labels, list):
            labels = torch.tensor(labels, dtype = torch.int32, device = self.device)

        if len(data.shape) == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
        assert data.shape[0] == labels.shape[0], ("data.shape[0]", data.shape[0], "labelsshape", labels.shape[0])

        self._samples.append(labels)
        self._data.append(data.to(device = self.device))

        if gradient is not None:
            raise (Exception("Gradient data not implemented for BlockBuilderSamples"))

    def build(self):
        samples = Labels(self._sample_names, torch.vstack(self._samples).to(self.device))
        block = TensorBlock(
            values = torch.cat(self._data, axis=0).to(device = self.device),
            samples = samples.to(device = self.device),
            components = [c.to(device = self.device) for c in self._components],
            properties = self._properties.to(device = self.device),
        )

        if self._gradient_samples is not None:
            raise (Exception("Gradient data not implemented for BlockBuilderSamples"))

        self._gradient_data = []
        self._data = []
        self._properties = []

        return block


class TensorBuilderPerProperties:
    def __init__(self, samples, components, property_names, gradient_samples=None, device='cpu'):
        assert isinstance(samples, torch.ScriptObject)
        assert all([isinstance(component, torch.ScriptObject) for component in components])
        assert (gradient_samples is None) or isinstance(gradient_samples, torch.ScriptObject)
        self._gradient_samples = gradient_samples
        self._samples = samples
        self._components = components

        self._property_names = property_names
        self._properties = []

        self._data = []
        self._gradient_data = []

        self.device = device

    def add_properties(self, labels, data, gradient = None):

        if not isinstance(data, torch.Tensor):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            elif isinstance(data, list):
                data = torch.tensor(data)
            assert isinstance(data, torch.Tensor)

        assert data.shape[0] == self._samples.shape[0]
        for i in range(len(self._components)):
            assert data.shape[i + 1] == self._components[i].shape[0]

        labels = np.array(labels)
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
        assert data.shape[2] == labels.shape[0]

        self._properties.append(labels)
        self._data.append(data)

        if gradient is not None:
            if len(gradient.shape) == 2:
                gradient = gradient.reshape(gradient.shape[0], gradient.shape[1], 1)

            assert gradient.shape[2] == labels.shape[0]
            self._gradient_data.append(gradient)

    def build(self):
        properties = Labels(self._property_names, torch.vstack(self._properties))
        block = TensorBlock(
            values = torch.cat(self._data, dim = 2).to(device = self.device),
            samples = self._samples.to(device = self.device),
            components = [c.to(device = self.device) for c in self._components],
            properties = properties.to(device = self.device),
        )

        if self._gradient_samples is not None:
            block.add_gradient(
                "positions",
                self._gradient_samples,
                torch.cat(self._gradient_data, dim = 2),
            )

        self._gradient_data = []
        self._data = []
        self._properties = []

        return block


def labels_where(labels, selection, return_idx = False):
    keys_out_vals = [[k[name] for name in selection.names] for k in labels]

    for slct in selection.values:
        if not torch.any([torch.all(slct == k) for k in keys_out_vals]):
            raise ValueError(
                f"selected key {selection.names} = {slct} not found"
                " in the output keys. Check the `selection` argument."
            )

    mask = [torch.any([torch.all(i == j) for j in selection.values]) for i in keys_out_vals]

    labels = Labels(names = labels.names, values = labels.values[mask])
    if return_idx:
        return labels, torch.where(mask)[0]
    return labels

def move_cell_shifts_to_keys(blocks):
    """ Move cell shifts when present in samples, to keys"""

    out_blocks = []
    out_block_keys = []

    for key, block in blocks.items():        
        translations = torch.unique(block.samples.values[:, -3:], dim = 0)
        for T in translations:
            block_view = block.samples.view(["cell_shift_a", "cell_shift_b", "cell_shift_c"]).values
            idx = torch.where(torch.all(torch.isclose(block_view, torch.tensor([T[0], T[1], T[2]])), dim = 1))[0]

            if len(idx):
                out_block_keys.append(list(key.values) + [T[0], T[1], T[2]])
                out_blocks.append(TensorBlock(
                        samples = Labels(blocks.sample_names[:-3], values = block.samples.values[idx][:, :-3]),
                        values = block.values[idx],
                        components = block.components,
                        properties = block.properties,
                    ))
                
    return TensorMap(Labels(blocks.keys.names + ["cell_shift_a", "cell_shift_b", "cell_shift_c"], torch.tensor(out_block_keys)), out_blocks)

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
        block_view = b.properties.view(['n_i', 'l_i', 'n_j', 'l_j']).values
        
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
    keys = Labels(in_blocks.keys.names+['n_i', 'l_i', 'n_j', 'l_j'], torch.stack(keys).to(device = device))
    tmap = TensorMap(keys, blocks)
    return mts.permute_dimensions(tmap, axis='keys', dimensions_indexes = [0,1,5,6,2,7,8,3,4])


def tmap_to_dict(tmap):
    temp={}
    for k,b in tmap.items():
        kl = tuple(k.values.tolist())
        temp[kl] = {}
        bsamp = np.array(b.samples.values.tolist())
        values = b.values.clone()
        ifrij = np.unique(bsamp[:,:3], axis = 0)
        for I in ifrij:
            idx = np.where(np.all(bsamp[:,:3] == I, axis = 1))[0]
            temp[kl][tuple(I.tolist())] = values[idx]
    return temp


def relabel_structure_indices(blocks: TensorMap, sample_ids) -> TensorMap:
    old_to_new = {int(sid): i for i, sid in enumerate(sample_ids)}
    mapped_blocks = []
    for _, block in blocks.items():
        sample_values = block.samples.values.clone()
        for isample in range(len(sample_values)):
            sample_values[isample, 0] = old_to_new[int(sample_values[isample, 0])]
        mapped_blocks.append(
            TensorBlock(
                samples=Labels(block.samples.names, sample_values),
                components=block.components,
                properties=block.properties,
                values=block.values,
            )
        )
    return TensorMap(blocks.keys, mapped_blocks)


def slice_tensormap_structures(tmap: TensorMap, structure_ids, sort_samples=False) -> TensorMap:
    """Slice ``tmap`` to ``structure_ids`` and keep that batch order in sample rows.
       sort samples NOT USED in new version
    """
    if len(structure_ids) == 0:
        raise ValueError("structure_ids must be non-empty")
    device = next(iter(tmap)).values.device
    parts = []
    for idx in structure_ids:
        parts.append(mts.slice(tmap, axis="samples", selection=Labels(["structure"], torch.tensor([[int(idx)]], device=device))))
    return mts.join(parts, axis="samples")


def species_to_keys(tmap, frames):
    """Fast version: reconstruct a tmap like cblocks_h_collapse_btype from a (which has species_i, species_j in properties)
    Args:
        tmap: a tmap with species_i, species_j in properties
        frames: a list of frames
    Returns:
        a tmap with keys (species_i, species_j) + keys from tmap
    """
    species_per_structure = [f.numbers for f in frames]
    reconstructed_blocks = []
    reconstructed_keys = []
    prop_names = tmap.property_names
    spi_index, spj_index=prop_names.index('species_i'), prop_names.index('species_j')
    properties_to_keep = [i for i in range(len(prop_names)) if i != spi_index and i != spj_index]
    filtered_prop_names = [prop_names[i] for i in properties_to_keep]
    

    for key, block in tmap.items():
        properties = block.properties
        species_i_vals = properties['species_i']
        species_j_vals = properties['species_j']
        species_pairs = torch.stack([species_i_vals, species_j_vals], dim=1)
        unique_species_pairs = torch.unique(species_pairs, dim=0)
        sample_values = block.samples.values
        struct_indices = sample_values[:, 0]
        center_indices = sample_values[:, 1]
        neighbor_indices = sample_values[:, 2]
        device = sample_values.device
        
        # Pre-compute all sample species pairs
        all_sample_indices = []
        all_sample_species = []
        for A, struct_idx in enumerate(torch.unique(struct_indices)):
            struct_idx_map = int(A)
            if struct_idx_map >= len(species_per_structure):
                continue
                
            struct_mask = (struct_indices == struct_idx)
            struct_species = torch.from_numpy(species_per_structure[struct_idx_map]).to(device=device)
            
            centers, neighbors = center_indices[struct_mask], neighbor_indices[struct_mask]
            center_species, neighbor_species = struct_species[centers], struct_species[neighbors]
            species_pair_per_sample = torch.stack([center_species, neighbor_species], dim=1)
            
            all_sample_indices.append(torch.where(struct_mask)[0])
            all_sample_species.append(species_pair_per_sample)
        
        if len(all_sample_indices) == 0:
            continue
            
        all_sample_indices = torch.cat(all_sample_indices)
        all_sample_species = torch.cat(all_sample_species)
        
        # For each unique species pair in properties, find matching samples
        for species_pair in unique_species_pairs:
            spi, spj = species_pair
            mask = (species_i_vals == spi) & (species_j_vals == spj)
            prop_indices = torch.where(mask)[0]
            
            # Find sample indices for this species pair
            sample_mask = (all_sample_species[:, 0] == spi) & (all_sample_species[:, 1] == spj)
            sample_indices = all_sample_indices[sample_mask]
            
            if len(sample_indices) == 0:
                continue
            
            # Create the filtered block
            block_values = block.values[sample_indices][..., prop_indices]
            block_samples = mts.Labels(block.samples.names, block.samples.values[sample_indices])
            block_components = block.components    
            block_properties = mts.Labels(filtered_prop_names, properties.values[prop_indices][:, properties_to_keep])
            
            new_key = (spi, spj) + tuple(key.values.tolist())
            reconstructed_keys.append(new_key)
            reconstructed_blocks.append(mts.TensorBlock(samples=block_samples, values=block_values, components=block_components, properties=block_properties))

    return mts.TensorMap(
        mts.Labels(['species_i', 'species_j'] + list(tmap.keys.names), torch.tensor(reconstructed_keys)),
        reconstructed_blocks
    )


def blocktype_to_keys(tmap):
    """Reconstruct a tmap with keys (block_type, species_i, species_j, L) from a tmap with keys (species_i, species_j, L, sigma)
    
    Filters samples based on block_type:
    - block_type 0: only samples where center == neighbor (diagonal blocks)
    - block_type 1, -1: only samples where center != neighbor (off-diagonal blocks)
    """
    reconstructed_blocks = []
    reconstructed_keys = []
    possible_block_type = {'same':[-1,0,1], 'different':[2]}
    prop_names = tmap.property_names
    blockindex=prop_names.index('block_type')
    if blockindex is None: 
        raise ValueError("Block type not found in properties")
    properties_to_keep = [i for i in range(len(prop_names)) if i != blockindex]
    filtered_prop_names = [prop_names[i] for i in properties_to_keep]
    

    for key, block in tmap.items():
        species_i, species_j = key['species_i'], key['species_j']
        if species_i == species_j:
            block_types = possible_block_type['same']
        else:
            block_types = possible_block_type['different']
        
        properties = block.properties
        sample_values = block.samples.values  # [structure, center, neighbor]
        center_indices = sample_values[:, 1]
        neighbor_indices = sample_values[:, 2]
        
        for block_type in block_types:
            prop_mask = properties['block_type'] == block_type
            if not torch.any(prop_mask):
                continue
            
            if block_type == 0:
                # i ==j for block_type 0
                sample_mask = center_indices == neighbor_indices
            else:
                # i !=j for block_type 1, -1
                sample_mask = center_indices != neighbor_indices
            
            if not torch.any(sample_mask):
                continue
            filtered_samples = mts.Labels(block.samples.names, sample_values[sample_mask])
            filtered_values = block.values[sample_mask][..., prop_mask]
            filtered_properties = mts.Labels(filtered_prop_names, properties.values[prop_mask][:, properties_to_keep])
            
            new_key = (block_type,) + tuple(key.values.tolist())
            reconstructed_keys.append(new_key)
            reconstructed_blocks.append(
                                        mts.TensorBlock(
                                            samples=filtered_samples,
                                            values=filtered_values,
                                            components=block.components,
                                            properties=filtered_properties
                                        )
            )
    
    return mts.sort(mts.TensorMap(
        mts.Labels(['block_type'] + list(tmap.keys.names), torch.tensor(reconstructed_keys)),
        reconstructed_blocks
    ))


def collapse_template(tmap, props_to_move=['parity']):
        collapsed = tmap.keys_to_properties(props_to_move)
        collapsed = collapsed.components_to_properties(['M'])
        collapsed = collapsed.keys_to_properties(['L'])
        
        return collapsed

def collapse_blocks( in_blocks, props_to_move=['parity']):

        collapsed = collapse_template(in_blocks, props_to_move)
        
        return collapsed[0].samples, collapsed.block(0).values 


def samples_to_properties(tmap, samples_to_move=[ 'center', 'neighbor']):
    blocks = []
    property_names = tmap.property_names
    property_names = property_names + samples_to_move
    samples_indices = [tmap.sample_names.index(sample) for sample in samples_to_move]
    remaining_sample_names = [name for name in tmap.sample_names if name not in samples_to_move]
    remaining_sample_indices = [tmap.sample_names.index(name) for name in remaining_sample_names]
    for key, block in tmap.items():
        nsamples, ncomponents, nproperties = block.values.shape
        sample_values_to_move = block.samples.values[:, samples_indices]
        new_samples = mts.Labels(remaining_sample_names, block.samples.values[:, remaining_sample_indices])
        properties = block.properties
        properties = mts.Labels(property_names, torch.cat([properties.values, sample_values_to_move], dim=1))
        values = block.values
        values = values.reshape(len(new_samples), -1, ncomponents, nproperties) 
        values = torch.permute(values, (0, 2, 3, 1)).reshape(len(new_samples), ncomponents,  -1)
        blocks.append(mts.TensorBlock(samples=new_samples, 
                                     values=block.values, 
                                     components=block.components, 
                                     properties=properties))


def props_to_keys(in_blocks, props_to_move=['species_i', 'species_j', 'L', 'parity']):
    blocks = []
    keys = []
    for k,b in in_blocks.items():
        properties = b.properties
        properties_to_move = [properties.names.index(prop) for prop in props_to_move]
        properties_to_keep = [i for i in range(len(properties.names)) if i not in properties_to_move]
        filtered_properties = mts.Labels(properties.names[properties_to_keep], properties.values[:, properties_to_keep])
        
        for props in properties_to_move:
            idx = torch.where(torch.all(torch.isclose(block_view, props), dim = 1))[0]
            
            keys.append(torch.hstack((k.values, props.clone().detach())))
            if len(idx):
                blocks.append(TensorBlock(
                            samples = b.samples,
                            values = b.values[...,idx],
                            components = b.components,
                            properties = filtered_properties
                        )
                )
    keys = mts.Labels(in_blocks.keys.names+list(props_to_move), torch.stack(keys))
    tmap = mts.TensorMap(keys, blocks)
    return tmap 
 