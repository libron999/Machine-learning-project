"""Synthetic train/resume/infer/evaluate regression test; writes only --work-dir."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, help='New empty directory for test outputs')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    test = Path(args.work_dir).resolve()
    if test.exists() and any(test.iterdir()):
        raise FileExistsError('Integration test work directory must be empty')
    test.mkdir(parents=True, exist_ok=True)
    data = test / 'data'
    rng = np.random.default_rng(19)
    for split, n in [('train', 5), ('test', 2)]:
        for folder in ['ir', 'vi']:
            (data / split / folder).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.fromarray(rng.integers(0, 256, (36, 40), dtype=np.uint8)).save(data / split / 'ir' / f'{i}.png')
            Image.fromarray(rng.integers(0, 256, (36, 40, 3), dtype=np.uint8)).save(data / split / 'vi' / f'{i}.png')
    config = yaml.safe_load((project / 'configs/icafusion_rgb_all.yaml').read_text())
    config['train'].update(epochs=2, crop_size=32, batch_size=2, repeats=1, workers=0,
                           d_steps=1, log_every_steps=1, save_every_steps=1)
    config['data']['root'] = str(data)
    (test / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    def run(script, *options):
        subprocess.run([sys.executable, '-B', str(project / script), *map(str, options)],
                       cwd=project, check=True)
    def train(name, *extra):
        run('train.py', '--config', test / 'config.yaml', '--run-dir', test / name,
            '--device', args.device, '--threads', '2', *extra)
    train('continuous', '--max-steps', '3')
    train('resumed', '--max-steps', '1')
    train('resumed', '--resume', test / 'resumed/latest.pt', '--max-steps', '3')
    a = torch.load(test / 'continuous/latest.pt', map_location='cpu', weights_only=False)
    b = torch.load(test / 'resumed/latest.pt', map_location='cpu', weights_only=False)
    maximum = 0.
    for name in ['generator', 'dir', 'dvis']:
        for key in a[name]:
            left, right = a[name][key], b[name][key]
            maximum = max(maximum, (left-right).abs().max().item())
            if args.device == 'cpu':
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            else:
                torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    assert a['step'] == b['step'] == 3
    print(f'PASS continuous/resumed G and D tensors; maximum difference={maximum}', flush=True)
    train('resumed', '--resume', test / 'resumed/latest.pt')
    assert (test / 'resumed/best.pt').exists()
    run('infer.py', '--checkpoint', test / 'resumed/best.pt', '--data-root', data,
        '--output', test / 'fused', '--device', args.device, '--threads', '2')
    run('evaluate.py', '--data-root', data, '--fused-dir', test / 'fused', '--output', test / 'metrics.csv')
    summary = json.loads((test / 'metrics.summary.json').read_text())
    assert summary['count'] == 2
    print('ALL INTEGRATION TESTS PASSED (synthetic data, not fusion quality).')


if __name__ == '__main__':
    main()
