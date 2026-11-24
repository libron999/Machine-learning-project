"""Measure actual memory/time on the rented GPU using synthetic inputs, no checkpoint."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from common import choose_device, load_config
from models import Generator, Discriminator
from engine import training_step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=640)
    args = p.parse_args()
    device = choose_device('cuda')
    cfg = load_config(args.config)
    g = Generator(**cfg['model']).to(device)
    di, dv = Discriminator(1, cfg['train']['discriminator_tanh']).to(device), Discriminator(3, cfg['train']['discriminator_tanh']).to(device)
    opts = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in [g, di, dv]]
    n, s = args.batch_size, cfg['train']['crop_size']
    ir, vis = torch.rand(n,1,s,s,device=device)*2-1, torch.rand(n,3,s,s,device=device)*2-1
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    training_step(g, di, dv, opts, ir, vis, cfg)
    torch.cuda.synchronize(device)
    report = {'gpu': torch.cuda.get_device_name(device), 'batch': n, 'crop': s,
              'train_step_seconds_including_first_use': time.perf_counter()-start,
              'train_peak_GiB': torch.cuda.max_memory_allocated(device)/2**30}
    del opts, di, dv, ir, vis
    g.zero_grad(set_to_none=True)
    g.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        ir = torch.rand(1,1,args.height,args.width,device=device)*2-1
        vis = torch.rand(1,3,args.height,args.width,device=device)*2-1
        g(ir,vis)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        g(ir,vis)
        torch.cuda.synchronize(device)
    report.update(full_image_hw=[args.height,args.width],
                  full_forward_seconds=time.perf_counter()-start,
                  inference_peak_GiB=torch.cuda.max_memory_allocated(device)/2**30)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
