import math
import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
def save_batch_heatmaps(tensor, save_dir, prefix="sample"):
    os.makedirs(save_dir, exist_ok=True)

    tensor = tensor.detach().cpu().numpy()
    B, C, H, W = tensor.shape

    for b in range(B):
        heatmaps = []

        for c in range(C):
            x = tensor[b, c]

            # normalize to [0, 255]
            x_norm = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
            x_norm = x_norm.astype(np.uint8)

            # pseudocolor
            heatmap = cv2.applyColorMap(x_norm, cv2.COLORMAP_JET)

            heatmaps.append(heatmap)

        # concatenate C channels horizontally
        concat = np.concatenate(heatmaps, axis=1)

        save_path = os.path.join(save_dir, f"{prefix[b]}.png")
        cv2.imwrite(save_path, concat)

def transitive_closure(layer_edges, contain_edges=()):
    # -------- Step 1: build layer graph --------
    graph = defaultdict(list)
    nodes = set()
    for u, v in layer_edges:
        graph[u].append(v)
        nodes.add(u)
        nodes.add(v)
    nodes = sorted(nodes)

    # -------- Step 2: compute layer transitive closure --------
    layer_closure = {u: set() for u in nodes}
    for start in nodes:
        visited = set()
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    layer_closure[start].add(v)
                    queue.append(v)

    # -------- Step 3: assign each lesion to its nearest layer --------
    # lesion -> shallowest layer
    lesion_owner = {}
    for layer, lesion in contain_edges:
        if lesion not in lesion_owner or layer < lesion_owner[lesion]:
            lesion_owner[lesion] = layer

    # -------- Step 4: propagate lesions only to ancestor layers --------
    result = {u: set() for u in nodes}

    for u in nodes:
        # strictly below layers
        result[u] |= layer_closure[u]

        # lesions in strictly lower layers
        for lesion, owner in lesion_owner.items():
            # owner is strictly below u
            if owner in layer_closure[u]:
                result[u].add(lesion)
    return result
def unreachable_nodes(reach_map: Dict[int, List[int]]) -> Dict[int, List[int]]: 
    universe = set(reach_map.keys())
    for nodes in reach_map.values():
        universe.update(nodes)
    # compute the complement
    complement = {}
    for node in universe:
        reachable = set(reach_map.get(node, []))
        complement[node] = sorted(universe - {node} - reachable)
    return complement

def build_matrices(directed_edges: List[Tuple[int, int]],
                   contain_edges: List[Tuple[int, int]],
                   C: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    From the directed edge list and the number of classes C, generate:
        adj_up_matrix   : reachability matrix (closure)
        adj_down_matrix : non-reachability matrix (complement graph)
    """
    # for each class, the classes that must not appear above it
    adj_up_list = transitive_closure(directed_edges, contain_edges=[])
    adj_up_matrix = adjacency_tensor(adj_up_list, C)
    adj_down_matrix =  adj_up_matrix.T
    
    return adj_up_matrix, adj_down_matrix

def ffill_along_h_torch(label, child_labels):
    """
        Contained classes may have more than one containment relation; in the label they follow their nearest parent class and are filled downward.
        label: torch.Tensor of shape (B, H, W)
        child_labels: list[int] or tensor of child labels to be replaced
        return: torch.Tensor of same shape (B, H, W)
        """
    device = label.device
    B, H, W = label.shape
    # Step 1: replace child labels with NaN
    mask_child = torch.isin(label, torch.tensor(child_labels, device=device))
    label = label.float()
    label[mask_child] = float('nan')
    label[:, 0,:][torch.isnan(label[:, 0,:])] = 0

    mask = ~torch.isnan(label)  # True where valid

    idx = torch.where(
        mask,
        torch.arange(H, device=device).view(1, H, 1),
        torch.zeros(1, H, 1, device=device, dtype=torch.long)
    )
    idx = torch.cummax(idx, dim=1)[0].long()
    out = label.gather(1, idx)
    return out.long()

def get_weight(labels, adj_up_matrix, adj_down_matrix, child_labels=None):
    # contained classes follow the parent class
    if child_labels is not None:
       father_label = ffill_along_h_torch(labels, child_labels)
    else:
       father_label =  labels    
    # upward non-adjacency of each pixel
    #B,H,W,C
    up_matrix = adj_up_matrix[father_label]
    # downward non-adjacency of each pixel
    down_matrix = adj_down_matrix[father_label]
    # accumulate bottom-up along H
    up_cum = torch.flip(torch.cumsum(torch.flip(up_matrix, dims=[1]), dim=1), dims=[1])
    # accumulate top-down along H
    down_cum = torch.cumsum(down_matrix, dim=1)
    cum = up_cum + down_cum

    
    cum_max, _ = cum.max(dim=1, keepdim=True)
    # normalization fix: the original exp(cum/cum.sum(HxW)) squeezes values to ~1e-5, so after exp they are ≈1 (almost no weighting).
    # divide by the global max of cum instead, weight ∈ [1, e], with real spatial structure; all-zero classes (cum_max=0) fall back to 1 (safe).
    weight = torch.exp(cum/(cum.amax(dim=(1,2), keepdim=True)+1e-8))
    return weight


def adjacency_tensor(adj_dict: dict, C: int) -> torch.Tensor:
    """
    Build an adjacency tensor with a zero diagonal from adjacency list adj_dict and the number of classes C.

    Parameters
    ----------
    adj_dict : dict[int, list[int]]
        Adjacency list; key is the source node, value is the list of target nodes.
    C : int
        Number of classes (node ids should fall in 0..C-1).

    Returns
    -------
    torch.Tensor
        0/1 adjacency tensor of shape (C, C), where A[i, j] = 1 means there is an edge i → j.
    """
    # collect all node ids
    nodes = set(adj_dict.keys())
    for v_list in adj_dict.values():
        nodes.update(v_list)
    nodes = sorted(nodes)
    # validity check
    if any(n < 0 or n >= C for n in nodes):
        raise ValueError(f"Node index must be between 0 and {C - 1}.")
    # initialize the all-zero tensor
    A = torch.zeros((C, C), dtype=torch.int)
    # fill 1s from the adjacency list
    for src, dst_list in adj_dict.items():
        for dst in dst_list:
            A[src, dst] = 1
    # force the diagonal to 0
    A.fill_diagonal_(0)
    return A


class weighted_bce_with_logits(nn.Module):
    def __init__(self, contain_edges, directed_edges, num_classes, device='cpu'):
        super().__init__()
        self.device = device    
        self.num_classes = num_classes
        self.containment_adj = self._build_containment_adj(contain_edges, num_classes).to(device)
        #self.register_buffer('containment_adj', self._build_containment_adj(contain_edges, num_classes))
        self.adj_up_matrix, self.adj_down_matrix = build_matrices(directed_edges, contain_edges, num_classes)

    def _build_containment_adj(self, contain_edges, num_classes):
        # initialize identity matrix A
        A = torch.eye(num_classes)
        # convert the relations in contain_edges to an adjacency matrix
        for src, dst in contain_edges:
            A[dst, src] = 1.0  # src contains dst
        return A    
    def forward(self, logits, targets, weight=True, eps=1e-8,):    
        """
        logits: [B, C, H, W] arbitrary shape
        targets: same shape, 0/1
        weight: same shape or broadcastable, used for weighting
        eps: prevents log(0)
        """
        if isinstance(logits, dict):
             logits = logits['segment_output']
        logits = rearrange(logits, 'b c h w -> b h w c')
        if isinstance(targets, dict):
            
            if 'Layer_single_lab' in targets and 'label' in targets:
                Layer_lab = targets['Layer_single_lab'].to(self.device)
                Layer_fluid_lab = targets['label'].to(self.device)
                father_one_hot = F.one_hot(Layer_lab, num_classes=self.num_classes)
                child_one_hot = F.one_hot(Layer_fluid_lab, num_classes=self.num_classes) 
                multi_label_one_hot = torch.max(father_one_hot, child_one_hot).float()            
            else:
                Layer_fluid_lab = targets['label']  
                multi_label_one_hot = self.containment_adj[Layer_fluid_lab].float()    
        else:
            multi_label_one_hot = self.containment_adj[targets].float()     
        if weight:
                  labels_weight_list = get_weight(Layer_lab, self.adj_up_matrix.to(self.device),self.adj_down_matrix.to(self.device))
        else:
                  labels_weight_list = None  
               
        probs_flat_new =torch.sigmoid(logits)#[:,:,-1:]
        
        if labels_weight_list is not None:
            bce = - (multi_label_one_hot * torch.log(probs_flat_new + eps) + (1 - multi_label_one_hot) * torch.log(1 - probs_flat_new + eps)*labels_weight_list )
            
            
        else:
            bce = - (multi_label_one_hot * torch.log(probs_flat_new + eps) + (1 - multi_label_one_hot) * torch.log(1 - probs_flat_new + eps))   
        #.sum(dim=1)
        return bce.mean()
