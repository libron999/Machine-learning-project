"""Strict filename pairing and stateless augmentation for reproducible resume."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def image_index(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Missing image directory: {directory}')
    result = {}
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            if p.stem in result:
                raise ValueError(f'Duplicate image stem {p.stem}: {directory}')
            result[p.stem] = p
    if not result:
        raise ValueError(f'No images in {directory}')
    return result


def paired_index(root, split, ir_dir='ir', vis_dir='vi'):
    base = Path(root) / split
    ir, vis = image_index(base / ir_dir), image_index(base / vis_dir)
    if ir.keys() != vis.keys():
        raise ValueError(f'Unpaired files: IR-only={sorted(ir.keys()-vis.keys())[:8]}, '
                         f'VIS-only={sorted(vis.keys()-ir.keys())[:8]}')
    return {k: (ir[k], vis[k]) for k in sorted(ir)}


def make_manifest(root, cfg, seed):
    pairs = paired_index(root, cfg['train_split'], cfg['ir_dir'], cfg['vis_dir'])
    names = list(pairs)
    if cfg.get('train_list') or cfg.get('val_list'):
        if not cfg.get('train_list') or not cfg.get('val_list'):
            raise ValueError('Both train_list and val_list are required')
        def read(path):
            items = [line.strip() for line in Path(path).read_text(encoding='utf-8-sig').splitlines()
                     if line.strip() and not line.startswith('#')]
            if len(set(items)) != len(items):
                raise ValueError(f'Duplicate IDs in {path}')
            return items
        train, val = read(cfg['train_list']), read(cfg['val_list'])
        if set(train) & set(val) or set(train + val) != set(names):
            raise ValueError('Custom lists must be disjoint and together cover the official train directory')
    else:
        if len(names) < 2:
            raise ValueError('Need at least two training pairs for held-out validation')
        if not 0 < cfg['val_fraction'] < 1:
            raise ValueError('val_fraction must be between 0 and 1')
        shuffled = names.copy()
        random.Random(seed).shuffle(shuffled)
        n = min(len(names)-1, max(1, round(len(names) * cfg['val_fraction'])))
        val, train = sorted(shuffled[:n]), sorted(shuffled[n:])
    # Content hashes catch data changes even when names and dimensions are unchanged.
    hashes = {}
    for name, paths in pairs.items():
        hashes[name] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    body = {'train': train, 'val': val, 'sha256': hashes}
    body['fingerprint'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


class MSRSDataset(Dataset):
    def __init__(self, root, split, names=None, crop_size=None, augment=False, seed=42,
                 repeats=1, ir_dir='ir', vis_dir='vi'):
        self.pairs = paired_index(root, split, ir_dir, vis_dir)
        self.names = list(self.pairs) if names is None else list(names)
        if not self.names or set(self.names) - self.pairs.keys():
            raise ValueError('Empty split or unknown sample IDs')
        self.crop_size, self.augment, self.seed = crop_size, augment, seed
        self.repeats, self.epoch = repeats, 0
        if repeats < 1:
            raise ValueError('repeats must be >=1')

    def __len__(self):
        return len(self.names) * self.repeats

    def __getitem__(self, index):
        name = self.names[index % len(self.names)]
        ir_path, vis_path = self.pairs[name]
        with Image.open(ir_path) as image:
            if image.mode not in ('L', 'RGB', 'RGBA'):
                raise ValueError(f'Expected 8-bit IR image, got {image.mode}: {ir_path}')
            ir = np.array(image.convert('L'), dtype=np.float32)[..., None] / 255
        with Image.open(vis_path) as image:
            if image.mode not in ('RGB', 'RGBA'):
                raise ValueError(f'Expected genuinely RGB visible image, got {image.mode}: {vis_path}')
            vis = np.array(image.convert('RGB'), dtype=np.float32) / 255
        if ir.shape[:2] != vis.shape[:2]:
            raise ValueError(f'IR/VIS size mismatch: {name}')
        rng = random.Random(self.seed + self.epoch * 1000000007 + index)
        if self.crop_size:
            s = self.crop_size
            h, w = ir.shape[:2]
            if min(h, w) < s:
                raise ValueError(f'{name}: {h}x{w} smaller than crop_size={s}; no silent resize')
            top, left = rng.randint(0, h-s), rng.randint(0, w-s)
            ir, vis = ir[top:top+s, left:left+s], vis[top:top+s, left:left+s]
        if self.augment:
            for axis in (0, 1):
                if rng.random() < 0.5:
                    ir, vis = np.flip(ir, axis), np.flip(vis, axis)
        def tensor(x):
            return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))) * 2 - 1
        return {'ir': tensor(ir), 'vis': tensor(vis), 'name': name}
