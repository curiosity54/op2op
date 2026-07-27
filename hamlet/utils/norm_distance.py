import torch
from collections import defaultdict

from hamlet.utils.twocenter_utils import _orbs_offsets, _atom_blocks_idx

import numpy as np
import matplotlib.pyplot as plt

def get_norms_and_distances(blocks, frames):
    """
    norms of matrix elements and corresponding distances for each block.
    Returns:
        dict: {block_key: {'distances': [...], 'norms': [...]}}
    """
    results = defaultdict(lambda: {'distances': [], 'norms': []})
    
    for key, block in blocks.items():
        block_key = tuple(key.values.tolist())
        
        for (sample, value) in (zip(block.samples.values, block.values)):
            A, i, j = sample.tolist()
            frame = frames[A]
            distance = frame.get_distance(i, j)
            norm = torch.norm(value).item()
            results[block_key]['distances'].append(distance)
            results[block_key]['norms'].append(norm)
    
    return results

def get_atom_block_norms_and_distances(matrices, frames, orbitals, exclude_i_equal_j=False):
    """
    Extract norms of entire i-j atom subblocks (all orbitals on atom i × all orbitals on atom j) and corresponding distances 
    
    Returns:
        dict: {(species_i, species_j): {'distances': [...], 'norms': [...]}}
    """
    results = defaultdict(lambda: {'distances': [], 'norms': []})
    
    orbs_tot, _ = _orbs_offsets(orbitals)
    atom_idx = _atom_blocks_idx(frames, orbs_tot) 
    for A, (matrix, frame) in enumerate(zip(matrices, frames)):        
        for i in range(len(frame)):
            for j in range(i, len(frame)):
                if exclude_i_equal_j and i == j:
                    continue  
                if (A, i, j) in atom_idx:
                    i_start, j_start = atom_idx[(A, i, j)]
                    atom_i = frame.numbers[i]
                    atom_j = frame.numbers[j]
                    
                    i_slice = slice(i_start, i_start + orbs_tot[atom_i])
                    j_slice = slice(j_start, j_start + orbs_tot[atom_j])
                    subblock = matrix[i_slice, j_slice]
                    
                    norm = torch.norm(subblock).item()
                    distance = frame.get_distance(i, j)
                    # sorted order so (C,H) and (H,C) are combined
                    species_i, species_j = frame.symbols[i], frame.symbols[j]
                    species_key = tuple(sorted([species_i, species_j]))
                    results[species_key]['distances'].append(distance)
                    results[species_key]['norms'].append(norm)
    
    return results


def plot_atom_block_norms(data_dict, title_prefix="Fock", color = 'b', marker = 'o', markersize = 10, alpha = 0.5, save_path=None, fig=None, axes=None):
    """
    Plot norms of full i-j atom subblocks vs distance, grouped by species pairs.
    
    Args:
        data_dict: Dictionary from extract_atom_block_norms_and_distances
                   Keys are (species_i, species_j) tuples
        title_prefix: Prefix for plot title
        save_path: Path to save the figure
       
    """
    filtered_dict = {k: v for k, v in data_dict.items() if len(v['distances']) > 0}
    
    def get_atomic_number(symbol):
        from ase.data import atomic_numbers
        if isinstance(symbol, str):
            return atomic_numbers.get(symbol, 0)
        return symbol if isinstance(symbol, int) else 0
    
    def sort_key(species_pair):
        #Sort key: (is_same_species, atomic_num_i, atomic_num_j)
        species_i, species_j = species_pair
        is_same = (species_i == species_j)
        z_i = get_atomic_number(species_i)
        z_j = get_atomic_number(species_j)
        if is_same:
            return (0, z_i, z_i)  
        else:
            return (1, min(z_i, z_j), max(z_i, z_j))  

    sorted_keys = sorted(filtered_dict.keys(), key=sort_key)
    filtered_dict = {k: filtered_dict[k] for k in sorted_keys}
    
    # same x and y ranges for all plots 
    all_distances = []
    all_norms = []
    for species_pair, data in filtered_dict.items():
        distances = np.array(data['distances'])
        norms = np.array(data['norms'])
        if len(distances) > 0:
            all_distances.extend(distances.tolist())
            all_norms.extend(norms.tolist())
    
    if len(all_distances) > 0:
        x_min, x_max = np.min(all_distances), np.max(all_distances)
        y_min, y_max = np.min(all_norms), np.max(all_norms)
        x_range = x_max - x_min
        y_range = y_max - y_min if y_max > y_min else y_max * 0.1
        x_min_pad = max(0, x_min - 0.05 * x_range)
        x_max_pad = x_max + 0.05 * x_range
        y_min_pad = y_min / (10 ** (0.05 * np.log10(y_max / y_min))) if y_min > 0 else y_min
        y_max_pad = y_max * (10 ** (0.05 * np.log10(y_max / y_min))) if y_max > 0 else y_max
    else:
        x_min_pad, x_max_pad = 0, 10
        y_min_pad, y_max_pad = 1e-10, 1
    
    if fig is None or axes is None:
        n_plots = len(filtered_dict)
        ncols = int(np.ceil((1 + np.sqrt(1 + 4 * n_plots)) / 2))
        nrows = ncols - 1
        while nrows * ncols < n_plots:
            ncols += 1
            nrows = ncols - 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(6,4))
        axes = [axes] if nrows == 1 and ncols == 1 else axes.flatten()
        
        for idx in range(len(filtered_dict), len(axes)):
            axes[idx].axis('off')
        
        for idx, species_pair in enumerate(filtered_dict.keys()):
            ax = axes[idx]
            ax.set_xlabel(r'$d_{ij}$ (Å)', fontsize=12)
            if idx % ncols == 0:  
                ax.set_ylabel(r'$|\mathbf{M}_{ij}|$ (a.u.)', fontsize=12)
                ax.tick_params(axis='both', which='both', labelsize=12)
            else:
                ax.tick_params(axis='y', labelleft=False)
                ax.tick_params(axis='x', which='both', labelsize=12)
            ax.set_yscale('log')
            ax.set_xlim(x_min_pad, x_max_pad)
            ax.set_ylim(y_min_pad, y_max_pad)
            ax.set_title(f"{species_pair[0]}-{species_pair[1]}", fontsize=12)
            # ax.grid(True, alpha=0.3)
    else:
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for idx, species_pair in enumerate(filtered_dict.keys()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        distances = np.array(filtered_dict[species_pair]['distances'])
        norms = np.array(filtered_dict[species_pair]['norms'])
        
        if len(distances) == 0:
            continue  
        
        ax.scatter(distances, norms, color=color, alpha=alpha, marker=marker, s=markersize, label=title_prefix, rasterized=True)
    
    if fig is not None:
        fig.suptitle(title_prefix, fontsize=10, y=0.995)
        plt.tight_layout()  
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved atom-block plot to {save_path}", flush=True)
    
    return fig, axes

def plot_norms_vs_distance(data_dict, title_prefix="Fock", color = 'b', marker = 'o', markersize = 10, alpha = 0.5, save_path=None, max_subplots=25, exclude_zero_distance=False, fig=None, axes=None):
    
    filtered_dict = {}
    for block_key, data in data_dict.items():
        if exclude_zero_distance and len(block_key) > 0 and block_key[0] == 0:
            continue  
        filtered_dict[block_key] = data
    
    n_blocks = len(filtered_dict)
    if n_blocks == 0:
        print("No blocks to plot!", flush=True)
        return None
    
    if fig is None or axes is None:
        n_plots = min(n_blocks, max_subplots)
        ncols = int(np.ceil(np.sqrt(n_plots)) + 1)
        nrows = ncols - 1
        while nrows * ncols < n_plots:
            ncols += 1
            nrows = ncols - 1
        if n_blocks > max_subplots:
            print(f"Warning: {n_blocks} blocks found, showing first {max_subplots} subplots", flush=True)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6,4))
        axes = [axes] if nrows == 1 and ncols == 1 else axes.flatten()
        
        for idx in range(n_blocks, len(axes)):
            axes[idx].axis('off')
        
        for idx, block_key in enumerate(filtered_dict.keys()):
            if idx >= len(axes) or idx >= max_subplots:
                break
            ax = axes[idx]
            ax.set_xlabel('Distance (Å)', fontsize=9)
            ax.set_ylabel('Matrix Norm', fontsize=9)
            ax.set_yscale('log')
            label = f"Block {idx}: {block_key}"
            ax.set_title(label, fontsize=8)
    else:
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for idx, block_key in enumerate(filtered_dict.keys()):
        if idx >= len(axes) or idx >= max_subplots:
            break
            
        ax = axes[idx]
        distances = np.array(filtered_dict[block_key]['distances'])
        norms = np.array(filtered_dict[block_key]['norms'])
        
        if len(distances) == 0:
            continue  
        
        ax.scatter(distances, norms, color=color, alpha=alpha, marker=marker, s=markersize, label=title_prefix, rasterized=True)
    
    if fig is not None:
        if save_path or (axes is None):
            fig.suptitle(title_prefix, fontsize=10, y=0.995)
        plt.tight_layout()  
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', tight_layout=True)
        print(f"Saved plot to {save_path}", flush=True)
    
    return fig, axes