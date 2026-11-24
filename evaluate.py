"""Transparent diagnostic metrics; not a complete published fusion benchmark.

Y uses BT.601 coefficients, image intensity is [0,255]. MI uses 256 fixed bins.
Do not compare values against another implementation without matching definitions.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from data import paired_index, image_index


def read(path, rgb):
    with Image.open(path) as image:
        if rgb and image.mode != 'RGB':
            raise ValueError(f'RGB image required: {path}')
        return np.array(image.convert('RGB' if rgb else 'L'), dtype=np.float64)


def y(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114])


def color(rgb):
    luma = y(rgb)
    return np.stack((0.564*(rgb[..., 2]-luma), 0.713*(rgb[..., 0]-luma)), -1)


def entropy(a):
    hist = np.histogram(a, bins=256, range=(0, 256))[0].astype(float)
    p = hist[hist > 0] / hist.sum()
    return float(-(p*np.log2(p)).sum())


def mi(a, b):
    hist = np.histogram2d(a.ravel(), b.ravel(), bins=256, range=((0, 256), (0, 256)))[0]
    p = hist/hist.sum()
    independent = p.sum(0, keepdims=True)*p.sum(1, keepdims=True)
    nz = p > 0
    return float((p[nz]*np.log2(p[nz]/independent[nz])).sum())


def metrics(fused, ir, vis):
    yf, yv = y(fused), y(vis)
    dx, dy = np.diff(yf, axis=1), np.diff(yf, axis=0)
    # AG uses forward differences on the common (H-1)x(W-1) interior.
    return {'EN': entropy(yf), 'SD': float(yf.std()),
            'SF': float(np.sqrt((dx*dx).mean()+(dy*dy).mean())),
            'AG': float(np.sqrt((dx[:-1]**2+dy[:, :-1]**2)/2).mean()),
            'MI_sum': mi(yf, ir)+mi(yf, yv),
            'chroma_MAE_255': float(np.abs(color(fused)-color(vis)).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--fused-dir', required=True)
    parser.add_argument('--output', required=True, help='CSV file')
    args = parser.parse_args()
    pairs = paired_index(args.data_root, args.split)
    fused = image_index(args.fused_dir)
    if fused.keys() != pairs.keys():
        raise ValueError('Fused images must match the ENTIRE selected split exactly')
    rows = []
    for name, (irp, vp) in pairs.items():
        ir, vis, fusion = read(irp, False), read(vp, True), read(fused[name], True)
        if fusion.shape != vis.shape or fusion.shape[:2] != ir.shape:
            raise ValueError(f'Shape mismatch: {name}')
        rows.append({'name': name, **metrics(fusion, ir, vis)})
    path = Path(args.output)
    if path.exists() or path.with_suffix('.summary.json').exists():
        raise FileExistsError('Evaluation output already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'name'}
    path.with_suffix('.summary.json').write_text(json.dumps({'count': len(rows), **summary}, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
