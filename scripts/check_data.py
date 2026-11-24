"""Read-only MSRS validation. Does not generate patches or change source data."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from data import paired_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'test'])
    parser.add_argument('--hashes', action='store_true', help='Also check exact image-pair duplicates across splits')
    parser.add_argument('--report', help='Optional JSON output path')
    args = parser.parse_args()
    result, seen, duplicates = {}, {}, []
    for split in args.splits:
        pairs = paired_index(args.data_root, split)
        sizes, modes = set(), set()
        for name, (ir_path, vis_path) in pairs.items():
            with Image.open(ir_path) as ir, Image.open(vis_path) as vis:
                ir.load()
                vis.load()
                if ir.size != vis.size:
                    raise ValueError(f'{split}/{name}: size mismatch')
                if ir.mode != 'L' or vis.mode != 'RGB':
                    raise ValueError(f'{split}/{name}: expected IR=L VIS=RGB, got {ir.mode}/{vis.mode}')
                sizes.add(ir.size)
                modes.add((ir.mode, vis.mode))
            if args.hashes:
                digest = hashlib.sha256(ir_path.read_bytes() + vis_path.read_bytes()).hexdigest()
                if digest in seen:
                    duplicates.append([seen[digest], f'{split}/{name}'])
                else:
                    seen[digest] = f'{split}/{name}'
        result[split] = {'pairs': len(pairs), 'sizes_wh': sorted(sizes), 'modes': sorted(modes)}
    result['exact_duplicate_pairs'] = duplicates if args.hashes else 'not checked'
    print(json.dumps(result, indent=2))
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if duplicates:
        raise SystemExit('Duplicate pairs found: review the report before training')


if __name__ == '__main__':
    main()
