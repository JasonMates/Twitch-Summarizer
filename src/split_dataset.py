import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def source_key(example):
    ex_id = example.get('id', '')
    # Format currently looks like Source_windowId
    return ex_id.split('_', 1)[0] if '_' in ex_id else 'unknown'


def split_group(rows, train_ratio, val_ratio, rng):
    rows = list(rows)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description='Split chat-summary jsonl into train/val/test.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--train-ratio', type=float, default=0.85)
    parser.add_argument('--val-ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError('Ratios must be >0 and train+val < 1.')

    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError('Input file is empty.')

    groups = {}
    for r in rows:
        groups.setdefault(source_key(r), []).append(r)

    rng = random.Random(args.seed)
    train, val, test = [], [], []
    for _, grp in sorted(groups.items()):
        tr, va, te = split_group(grp, args.train_ratio, args.val_ratio, rng)
        train.extend(tr)
        val.extend(va)
        test.extend(te)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.outdir / 'train.jsonl', train)
    write_jsonl(args.outdir / 'val.jsonl', val)
    write_jsonl(args.outdir / 'test.jsonl', test)

    print(f'total={len(rows)} train={len(train)} val={len(val)} test={len(test)}')


if __name__ == '__main__':
    main()
