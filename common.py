import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml


def load_config(path):
    with open(path, encoding='utf-8-sig') as stream:
        cfg = yaml.safe_load(stream)
    expected = {'seed', 'model', 'data', 'train', 'loss'}
    if set(cfg) != expected:
        raise ValueError(f'Config sections must be {expected}')
    if cfg['train']['crop_size'] < 32 or cfg['train']['crop_size'] % 4:
        raise ValueError('Training crop_size must be >=32 and divisible by 4')
    for key in ['batch_size', 'epochs', 'd_steps', 'save_every_steps', 'log_every_steps', 'repeats']:
        if cfg['train'][key] < 1:
            raise ValueError(f'train.{key} must be >=1')
    if cfg['train']['workers'] < 0:
        raise ValueError('workers must be nonnegative')
    return cfg


def choose_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for a smoke test')
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # Full training checkpoints contain RNG metadata. Only load trusted files.
    return torch.load(path, map_location='cpu', weights_only=False)


def save_rgb(tensor, path):
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError('save_rgb expects [3,H,W]')
    array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.permute(1, 2, 0).numpy()).save(path)


def set_grad(module, enabled):
    for p in module.parameters():
        p.requires_grad_(enabled)


def assert_finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'Non-finite {name}; stop rather than save corrupted weights')


def source_digest():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        if '__pycache__' not in p.parts:
            digest.update(p.relative_to(root).as_posix().encode())
            digest.update(p.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
