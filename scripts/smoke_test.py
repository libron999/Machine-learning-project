"""Small functional checks, not a performance benchmark or a training result."""
import argparse
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from common import (choose_device, load_config, seed_everything, save_checkpoint,
                    load_checkpoint, save_rgb, rng_state, restore_rng)
from data import MSRSDataset
from models import Generator, Discriminator
from models.modules import dct, idct, dct2, idct2
from losses import luminance, chroma, content_loss
from engine import training_step


def run(device, work, base_channels):
    work.mkdir(parents=True, exist_ok=True)
    seed_everything(123)
    # Compare the FFT implementation against the independent DCT definition.
    for n in [1, 7, 8, 13]:
        x = torch.randn(2, n, device=device, dtype=torch.float64)
        k = torch.arange(n, device=device, dtype=x.dtype)[:, None]
        j = torch.arange(n, device=device, dtype=x.dtype)[None, :]
        basis = torch.cos(math.pi / n * (j+0.5) * k) * math.sqrt(2/n)
        basis[0] /= math.sqrt(2)
        torch.testing.assert_close(dct(x), x @ basis.T, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(idct(dct(x)), x, atol=1e-10, rtol=1e-10)
    x = torch.randn(2, 3, 11, 17, device=device, requires_grad=True)
    torch.testing.assert_close(idct2(dct2(x)), x, atol=2e-6, rtol=2e-6)
    torch.autograd.grad(dct2(x).square().mean(), x)
    print('PASS DCT definition, round-trip, odd sizes and backward', flush=True)
    # Constant RGB has zero chroma; RGB projection is differentiable.
    gray = torch.randn(2, 1, 9, 11, device=device).expand(-1, 3, -1, -1)
    torch.testing.assert_close(luminance(gray), gray[:, :1], atol=1e-6, rtol=1e-6)
    assert chroma(gray).abs().max() < 1e-6
    for enabled in [False, True]:
        config = load_config(Path(__file__).resolve().parents[1] / 'configs' /
                             ('icafusion_rgb_all.yaml' if enabled else 'icafusion_rgb.yaml'))
        config['model']['base_channels'] = base_channels
        config['train']['d_steps'] = 1
        generator = Generator(**config['model']).to(device)
        ir = torch.rand(2, 1, 32, 32, device=device)*2-1
        vis = torch.rand(2, 3, 32, 32, device=device)*2-1
        output = generator(ir, vis)
        assert output.shape == vis.shape and output.abs().max() <= 1
        loss, _ = content_loss(output, ir, vis, config['loss'])
        loss.backward()
        assert generator.vis_stem.conv.weight.grad.abs().sum() > 0
        assert (generator.conv12.conv.weight.grad.abs().flatten(1).sum(1) > 0).all()
        if enabled:
            for key in ('mgdc', 'legm', 'hfp', 'ffcm'):
                block = getattr(generator, key)
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters()), key
        dir_, dvis = Discriminator(1).to(device), Discriminator(3).to(device)
        optimizers = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in [generator, dir_, dvis]]
        before = dvis.l1.conv2d.weight.detach().clone()
        stats = training_step(generator, dir_, dvis, optimizers, ir, vis, config)
        assert not torch.equal(before, dvis.l1.conv2d.weight)
        assert all(math.isfinite(v) for v in stats.values())
        assert all(p.grad is None for p in dir_.parameters())
        generator.eval()
        odd_ir = torch.rand(1, 1, 35, 41, device=device)*2-1
        odd_vis = torch.rand(1, 3, 35, 41, device=device)*2-1
        with torch.no_grad():
            expected = generator(odd_ir, odd_vis)
        assert expected.shape == odd_vis.shape
        path = work / f'roundtrip_{enabled}.pt'
        save_checkpoint(path, {'generator': generator.state_dict(), 'rng': rng_state()})
        loaded = load_checkpoint(path)
        clone = Generator(**config['model']).to(device).eval()
        clone.load_state_dict(loaded['generator'])
        with torch.no_grad():
            torch.testing.assert_close(clone(odd_ir, odd_vis), expected, atol=0, rtol=0)
        restore_rng(loaded['rng'])
        first = torch.rand(4, device=device)
        restore_rng(loaded['rng'])
        assert torch.equal(first, torch.rand(4, device=device))
        save_rgb(expected[0], work / f'random_untrained_{enabled}.png')
        with Image.open(work / f'random_untrained_{enabled}.png') as image:
            assert image.mode == 'RGB' and image.size == (41, 35)
        print(f'PASS {"all modules" if enabled else "RGB baseline"}: forward, RGB gradients, '
              'full GAN step/GP, original-head convolution update, odd sizes, checkpoint and RGB save', flush=True)
    # Verify synchronized crop/flip using identical content in all channels.
    root = work / 'paired_fixture'
    for name in ['ir', 'vi']:
        (root / 'train' / name).mkdir(parents=True, exist_ok=True)
    pixels = np.arange(48*48, dtype=np.uint16).reshape(48, 48).astype(np.uint8)
    Image.fromarray(pixels).save(root / 'train/ir/a.png')
    Image.fromarray(np.repeat(pixels[..., None], 3, -1)).save(root / 'train/vi/a.png')
    ds = MSRSDataset(root, 'train', crop_size=32, augment=True, seed=17)
    sample = ds[0]
    torch.testing.assert_close(sample['ir'], sample['vis'][:1], atol=0, rtol=0)
    torch.testing.assert_close(ds[0]['vis'], sample['vis'], atol=0, rtol=0)
    print('PASS paired data crop/flip and stateless repeatability', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--work-dir', default='work/smoke')
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(choose_device(args.device), Path(args.work_dir), args.base_channels)
    print('ALL SMOKE TESTS PASSED. No trained quality claim.')
