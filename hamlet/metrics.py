import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from metatensor.torch import TensorMap


def L2_loss(pred: Union[torch.tensor, TensorMap, List], target: Union[torch.tensor, TensorMap, Tuple, List], loss_per_block = False, norm = 1):
    """L2 loss function"""
    
    if isinstance(pred, torch.Tensor):
        assert isinstance(target, torch.Tensor)
        assert (
            pred.shape == target.shape
        ), "Prediction and target must have the same shape"
        return torch.sum((pred - target)**2)
    
    elif isinstance(pred, torch.ScriptObject):
        if pred._type().name() == "TensorMap":
            assert isinstance(target, torch.ScriptObject) and target._type().name() == "TensorMap", "Target must be a TensorMap if prediction is a TensorMap"
        losses = []
        for key, block in pred.items():
            targetblock = target.block(key)
            
            assert (
                block.samples == targetblock.samples
            ), "Prediction and target must have the same samples"

            losses.append(torch.sum((block.values - targetblock.values)**2) / norm)
             

       
    elif isinstance(pred, dict):
        assert isinstance(target, dict), "Target must be a dictionary"
        losses = []
        for key, p in pred.items():
            t = target[key]
            losses.append(torch.norm(p - t)**2 / norm)

    elif isinstance(pred, list):

        losses = []
        for p, t in zip(pred, target):
            losses.append(torch.norm(p - t)**2)

    if loss_per_block:
        return losses, sum(losses)
    else:
        return sum(losses)


def rmse_eigenvalue(pred, target):
    loss = 0
    for A in range(len(pred)):
        struct_mse = torch.sum(torch.square(pred[A]-target[A]))/len(pred[A])
        loss+=struct_mse
    return torch.sqrt(loss/len(pred))


def weighted_block_l2_loss(pred_blocks, target_blocks, device = None, all_pairs = False):  
    device = device if device is not None else pred_blocks.device  
    total_weighted_loss = 0.0
    
    for key, pred_block in pred_blocks.items():
        target_block = target_blocks.block(key)
        block_type = key["block_type"]
        
        diff_squared = (pred_block.values - target_block.values) ** 2
        if not all_pairs:        
            if block_type == 0 or abs(block_type) == 1:
                n_properties = diff_squared.shape[-1]
                fac = torch.full((n_properties,), 2.0, device=device, dtype=diff_squared.dtype)
                props = pred_block.properties.values
                
                diagonal_mask = (props[:, 0] == props[:, 2]) & (props[:, 1] == props[:, 3])
                
                fac[diagonal_mask] = 1.0
                
                fac = fac.view(1, 1, 1, -1)
                weighted_diff = diff_squared * fac
                total_weighted_loss += weighted_diff.sum()
            elif block_type == 2:
                weighted_diff = diff_squared * 2.0
                total_weighted_loss += weighted_diff.sum()
            else:
                total_weighted_loss += diff_squared.sum()
    
        else:
            total_weighted_loss += diff_squared.sum()
    return total_weighted_loss


def weighted_irrep_l2_loss(
    pred: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    target: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    # edge_graphs: List[EdgeGraphPerFrame],
    basis_out_template: TensorMap,
    device: Optional[Union[str, torch.device]] = None,
    all_pairs = False,
) -> torch.Tensor:
    """
    Same weighting as :func:`weighted_block_l2_loss`, but on readout-flat tensors from
    `hamlet.models.gnn_coupled_blocks.tensormap_to_irrep_tensors` / ``predict_structure_irrep_tensors``.
    """
    from hamlet.models.gnn_coupled_blocks import _tensormap_key2str

    pred_edge, pred_node = pred
    tgt_edge, tgt_node = target
    if device is None:
        device = next(iter(pred_edge.values())).device
    dtype = next(iter(pred_edge.values())).dtype
    total_weighted_loss = 0.0 
    for key, templ_block in basis_out_template.items():
        k_str = _tensormap_key2str(key)
        block_type = int(key["block_type"])
        n_comp = templ_block.values.shape[1]
        n_prop = templ_block.values.shape[2]
        if block_type == 0:
            diff = pred_node[k_str] - tgt_node[k_str]
        else:
            diff = pred_edge[k_str] - tgt_edge[k_str]
        diff_squared = diff.reshape(diff.shape[0], n_comp, n_prop) ** 2


        if not all_pairs:
            raise NotImplementedError("all_pairs=False is not implemented")
        else:
            total_weighted_loss += diff_squared.sum()

    return total_weighted_loss


def _cat_structure_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate same-key readout flats from multiple structures (with possibly diff num atoms)."""
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=0)


def weighted_irrep_l2_loss_batch_loop(
    preds: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    targets: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    basis_out_template: TensorMap,
    device: Optional[Union[str, torch.device]] = None,
    all_pairs: bool = False,
) -> torch.Tensor:
    """Per-structure Python loop (reference implementation)."""
    tot = 0.0
    for p, t in zip(preds, targets):
        tot = tot + weighted_irrep_l2_loss(p, t, basis_out_template, device=device, all_pairs=all_pairs)
    return tot


def weighted_irrep_l2_loss_batch(
    preds: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    targets: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    # edge_graphs: List[EdgeGraphPerFrame],
    basis_out_template: TensorMap,
    device: Optional[Union[str, torch.device]] = None,
    all_pairs: bool = False) -> torch.Tensor:
    """
    Sum :func:`weighted_irrep_l2_loss` over structures after mapping ``target_blocks`` with
    :func:`hamlet.models.gnn_coupled_blocks.tensormap_to_irrep_tensors`.

    With ``all_pairs=True``, concatenates all structures per readout key on
    dim 0 and sums squared differences in one pass — same scalar as the per-structure loop,
    including batches of different-sized molecules (only ``flat_dim`` must match per key).

    ``structure_ids`` must align ``frames[i]`` with the ``structure`` sample column in ``target_blocks`` when the
    batch TensorMap joins multiple global structure indices (e.g. pass ``batch.sample_id`` from :class:`MLDataset`).
    """
    if not all_pairs:
        return weighted_irrep_l2_loss_batch_loop(
            preds, targets, basis_out_template, device=device, all_pairs=all_pairs
        )
    if len(preds) != len(targets):
        raise ValueError(f"len(preds)={len(preds)} != len(targets)={len(targets)}")
    if len(preds) == 0:
        if device is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=device)

    from hamlet.models.gnn_coupled_blocks import _tensormap_key2str

    pred_edges = [p[0] for p in preds]
    pred_nodes = [p[1] for p in preds]
    target_edges = [t[0] for t in targets]
    target_nodes = [t[1] for t in targets]

    if device is None:
        device = next(iter(pred_edges[0].values())).device

    total_weighted_loss = torch.zeros((), device=device, dtype=next(iter(pred_edges[0].values())).dtype)
    for key, _templ_block in basis_out_template.items():
        k_str = _tensormap_key2str(key)
        block_type = int(key["block_type"])
        if block_type == 0:
            pred_cat = _cat_structure_tensors([pred_nodes[i][k_str] for i in range(len(preds))])
            target_cat = _cat_structure_tensors([target_nodes[i][k_str] for i in range(len(targets))])
        else:
            pred_cat = _cat_structure_tensors([pred_edges[i][k_str] for i in range(len(preds))])
            target_cat = _cat_structure_tensors([target_edges[i][k_str] for i in range(len(targets))])
        if pred_cat.shape != target_cat.shape:
            raise ValueError(
                f"loss key {k_str!r}: pred shape {tuple(pred_cat.shape)} != target {tuple(target_cat.shape)}"
            )
        diff = pred_cat - target_cat
        total_weighted_loss = total_weighted_loss + diff.square().sum()

    return total_weighted_loss

