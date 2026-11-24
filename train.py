"""Single-GPU/CPU training with exact batch-boundary resume on the same platform."""
import argparse
import json
import math
import signal
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from common import (load_config, choose_device, seed_everything, rng_state, restore_rng,
                    save_checkpoint, load_checkpoint, save_rgb, source_digest, write_json)
from data import MSRSDataset, make_manifest
from engine import training_step
from losses import content_loss
from models import Generator, Discriminator


def resume_signature(cfg):
    # Runtime paths, workers, checkpoint frequency and total epochs may change.
    keys = ('batch_size', 'crop_size', 'repeats', 'd_steps', 'g_lr', 'd_lr',
            'betas', 'discriminator_tanh')
    return {'model': cfg['model'], 'loss': cfg['loss'], 'seed': cfg['seed'],
            'train': {k: cfg['train'][k] for k in keys}}


@torch.no_grad()
def validate(model, dataset, device, loss_cfg, image_dir):
    model.eval()
    sums = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        ir = sample['ir'].unsqueeze(0).to(device)
        vis = sample['vis'].unsqueeze(0).to(device)
        fused = model(ir, vis)
        loss, parts = content_loss(fused, ir, vis, loss_cfg)
        parts['selection_loss'] = loss
        for key, value in parts.items():
            sums[key] = sums.get(key, 0) + value.item()
        if index < 4:
            save_rgb(fused[0], Path(image_dir) / (sample['name'] + '.png'))
    return {k: v / len(dataset) for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', help='Trusted latest.pt checkpoint')
    parser.add_argument('--max-steps', type=int, default=0, help='Stop at this TOTAL global step; 0=full run')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--threads', type=int, default=4, help='CPU intra-op threads')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    if args.data_root:
        cfg['data']['root'] = args.data_root
    if args.workers is not None:
        cfg['train']['workers'] = args.workers
    if args.batch_size is not None:
        cfg['train']['batch_size'] = args.batch_size
    if cfg['train']['batch_size'] < 1 or cfg['train']['workers'] < 0:
        raise ValueError('batch-size must be positive, workers nonnegative')
    device = choose_device(args.device)
    seed_everything(cfg['seed'])
    run = Path(args.run_dir)
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise FileExistsError('Run directory is not empty; use --resume or choose a new directory')
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint and checkpoint.get('format_version') != 3:
        raise ValueError('Original-head version requires a new run; cannot resume fixed-head checkpoints')
    if checkpoint and resume_signature(checkpoint['config']) != resume_signature(cfg):
        raise ValueError('Resume configuration changes model/loss/optimization or batch order')
    print('Checking train/validation data content hashes...', flush=True)
    manifest = make_manifest(cfg['data']['root'], cfg['data'], cfg['seed'])
    if checkpoint and checkpoint['manifest']['fingerprint'] != manifest['fingerprint']:
        raise ValueError('Data contents or split changed since checkpoint')
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / 'config.json', cfg)
    write_json(run / 'split.json', manifest)
    code_hash = source_digest()
    write_json(run / 'environment.json', {'torch': torch.__version__, 'device': str(device),
               'cuda_runtime': torch.version.cuda, 'source_sha256': code_hash,
               'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None})
    if checkpoint and checkpoint.get('source_sha256') != code_hash:
        print('WARNING: code fingerprint changed since checkpoint; record this deviation.', flush=True)
    data_cfg, tc = cfg['data'], cfg['train']
    shared = dict(root=data_cfg['root'], split=data_cfg['train_split'],
                  ir_dir=data_cfg['ir_dir'], vis_dir=data_cfg['vis_dir'], seed=cfg['seed'])
    training = MSRSDataset(**shared, names=manifest['train'], crop_size=tc['crop_size'],
                           augment=True, repeats=tc['repeats'])
    validation = MSRSDataset(**shared, names=manifest['val'])
    generator = Generator(**cfg['model']).to(device)
    dir_ = Discriminator(1, tc['discriminator_tanh']).to(device)
    dvis = Discriminator(3, tc['discriminator_tanh']).to(device)
    optimizers = [torch.optim.Adam(model.parameters(), lr=lr, betas=tuple(tc['betas']))
                  for model, lr in ((generator, tc['g_lr']), (dir_, tc['d_lr']), (dvis, tc['d_lr']))]
    start_epoch, next_batch, step, best = 0, 0, 0, float('inf')
    if checkpoint:
        for model, key in ((generator, 'generator'), (dir_, 'dir'), (dvis, 'dvis')):
            model.load_state_dict(checkpoint[key], strict=True)
        for opt, state in zip(optimizers, checkpoint['optimizers']):
            opt.load_state_dict(state)
        start_epoch, next_batch, step, best = (checkpoint[k] for k in ('epoch', 'next_batch', 'step', 'best'))
        restore_rng(checkpoint['rng'])
    stop = {'requested': False}
    def request_stop(signum, frame):
        stop['requested'] = True
        print('Stop requested: saving after the current complete optimizer step.', flush=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    def save(epoch, batch, filename='latest.pt'):
        save_checkpoint(run / filename, {
            'format_version': 3, 'config': cfg, 'manifest': manifest,
            'source_sha256': code_hash, 'generator': generator.state_dict(),
            'dir': dir_.state_dict(), 'dvis': dvis.state_dict(),
            'optimizers': [o.state_dict() for o in optimizers], 'rng': rng_state(),
            'epoch': epoch, 'next_batch': batch, 'step': step, 'best': best})
    print(f'G parameters: {sum(p.numel() for p in generator.parameters()):,}; '
          f'train={len(manifest["train"])} val={len(validation)} device={device}', flush=True)
    if args.max_steps and step >= args.max_steps:
        print('Checkpoint already reached --max-steps; no update performed.')
        return
    for epoch in range(start_epoch, tc['epochs']):
        training.epoch = epoch
        # Independent generators prevent loader construction from consuming model RNG.
        order_rng = torch.Generator().manual_seed(cfg['seed'] + epoch)
        order = torch.randperm(len(training), generator=order_rng).tolist()
        offset = next_batch if epoch == start_epoch else 0
        total_batches = math.ceil(len(order) / tc['batch_size'])
        if offset > total_batches:
            raise ValueError('Checkpoint batch offset exceeds dataset length')
        remaining = Subset(training, order[offset * tc['batch_size']:])
        loader = DataLoader(remaining, batch_size=tc['batch_size'], num_workers=tc['workers'],
                            shuffle=False, pin_memory=device.type == 'cuda',
                            generator=torch.Generator().manual_seed(cfg['seed'] + epoch),
                            persistent_workers=False)
        started = time.monotonic()
        for batch_no, batch in enumerate(loader, start=offset):
            ir, vis = batch['ir'].to(device, non_blocking=True), batch['vis'].to(device, non_blocking=True)
            stats = training_step(generator, dir_, dvis, optimizers, ir, vis, cfg)
            step += 1
            if step % tc['log_every_steps'] == 0 or step == 1:
                record = {'epoch': epoch+1, 'step': step, 'batch': batch_no+1,
                          'seconds': time.monotonic()-started, **stats}
                with (run / 'train.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
            stopping = stop['requested'] or (args.max_steps and step >= args.max_steps)
            if step % tc['save_every_steps'] == 0 or stopping:
                save(epoch, batch_no+1)
            if stopping:
                print(f'Saved {run / "latest.pt"} at step {step}; resume uses the next batch.', flush=True)
                return
        metrics = validate(generator, validation, device, cfg['loss'], run / 'preview')
        with (run / 'validation.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'epoch': epoch+1, 'step': step, **metrics}) + '\n')
        improved = metrics['selection_loss'] < best
        if improved:
            best = metrics['selection_loss']
        save(epoch+1, 0)
        if improved:
            save(epoch+1, 0, 'best.pt')
        print(f'Validation epoch {epoch+1}: {metrics}; best={best:.6f}', flush=True)
        if stop['requested']:
            return
    print('Training complete.', flush=True)


if __name__ == '__main__':
    main()
