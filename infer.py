"""Direct full-image RGB inference; no chroma replacement or per-image rescaling."""
import argparse
import time
from pathlib import Path
import torch
from common import choose_device, load_checkpoint, save_rgb, write_json
from data import MSRSDataset
from models import Generator


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=0, help='0=all images; nonzero is for debugging only')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory to avoid stale/mixed results')
    ckpt = load_checkpoint(args.checkpoint)
    cfg = ckpt['config']
    model = Generator(**cfg['model']).to(device).eval()
    model.load_state_dict(ckpt['generator'], strict=True)
    ds = MSRSDataset(args.data_root, args.split, ir_dir=cfg['data']['ir_dir'], vis_dir=cfg['data']['vis_dir'])
    output.mkdir(parents=True, exist_ok=True)
    times, shapes = [], set()
    count = min(len(ds), args.limit) if args.limit > 0 else len(ds)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for i in range(count):
        item = ds[i]
        ir, vis = item['ir'][None].to(device), item['vis'][None].to(device)
        size = tuple(ir.shape[-2:])
        if size not in shapes:
            for _ in range(2):
                model(ir, vis)
            shapes.add(size)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fused = model(ir, vis)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        times.append(time.perf_counter()-start)
        save_rgb(fused[0], output / (item['name']+'.png'))
        if (i+1) % 20 == 0 or i == 0:
            print(f'{i+1}/{count}', flush=True)
    write_json(output / 'inference.json', {'checkpoint': str(Path(args.checkpoint).resolve()),
               'model': cfg['model'], 'training_step': ckpt['step'], 'count': count,
               'mean_forward_seconds': sum(times)/len(times), 'image_sizes_hw': sorted(shapes),
               'parameters': sum(p.numel() for p in model.parameters()), 'device': str(device),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
               'note': 'Forward only; excludes I/O and warmup. No tiled inference.'})


if __name__ == '__main__':
    main()
