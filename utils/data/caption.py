# -*- coding: utf-8 -*-
import os, random
import torch
from torch_geometric.data import Data, Dataset

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def _load_graph_pt(dir_graph: str, fname: str): 
    path = os.path.join(dir_graph, fname) 
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, Data): 
        return obj 
    raise RuntimeError(f"[graph] {path} is not a torch_geometric.data.Data (got {type(obj)}); please check processed_data.")

def _common_filenames(dir_graph: str, take_all_graph=True):
    gnames = sorted(f for f in os.listdir(dir_graph) if f.endswith(".pt"))
    return gnames




class FixVoxelLabelTransform:
    """PyG transform that hot-fixes legacy one-hot labels on load: [N, C] FloatTensor -> [N,] LongTensor (-1, 0..C-1)."""
    def __call__(self, data):
        if hasattr(data, 'voxel_label') and data.voxel_label.dim() == 2 and data.voxel_label.is_floating_point():
            label_sum = data.voxel_label.sum(dim=-1)
            label_index = data.voxel_label.argmax(dim=-1)
            new_label = label_index.long()
            new_label[label_sum == 0] = -1
            data.voxel_label = new_label
        return data


class CaptionAugmentedDataset(Dataset):
    def __init__(self, dir_graph: str,  filenames: list = None, transform=None):
        super().__init__()
        self.dir_graph = dir_graph
        self.filenames = filenames or sorted(f for f in os.listdir(dir_graph) if f.endswith(".pt"))
        self.transform = transform

    def len(self):
        return len(self.filenames)

    def get(self, idx: int):
        data = _load_graph_pt(self.dir_graph, self.filenames[idx])

        llm = list(getattr(data, "llm_variants", []) or [])
        if llm:
            # store a single string (PyG batching concatenates into a list); random sample = data augmentation
            data.prompts = random.choice(llm)
        else:
            brief = getattr(data, "rule_based_brief", "") or ""
            desc  = getattr(data, "rule_based_description", "") or ""
            data.prompts = brief or desc or ""

        if self.transform:
            data = self.transform(data)
        return data
