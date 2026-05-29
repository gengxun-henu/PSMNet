"""Helpers for building DOM sample file lists.

Mirrors the role of ``listflowfile.py`` (SceneFlow) for the DOM-Anchored
Dual-Reference mode.

File-list format (CSV)
-----------------------
One sample per line; columns are comma-separated absolute (or
datapath-relative) file paths:

    nac_l, nac_r, render_l, render_r, geom,
    view_dir_l, view_dir_r, tan_t_l, tan_t_r,
    mask_l, mask_r [, dz_gt [, dem_init [, sun_dir_l [, normal_base_l]]]]

Lines starting with ``#`` are treated as comments.

Example usage (generate a list from a structured directory tree)
-----------------------------------------------------------------
Assumes the following layout::

    dataset_root/
        scene_001/
            nac_l.tif
            nac_r.tif
            render_l.tif
            render_r.tif
            geom.tif
            view_dir_l.tif
            view_dir_r.tif
            tan_t_l.tif
            tan_t_r.tif
            mask_l.tif
            mask_r.tif
            dz_gt.tif          # optional
            dem_init.tif       # optional
            sun_dir_l.tif      # optional
            normal_base_l.tif  # optional
        scene_002/
            ...

Run::

    python dataloader/dom_listfile.py \
        --dataroot /path/to/dataset_root \
        --out_train /path/to/train_list.csv \
        --out_test  /path/to/test_list.csv \
        --train_ratio 0.9
"""

import os
import csv
import random
import argparse

# Required filenames within each sample directory
REQUIRED_FILES = [
    'nac_l.tif', 'nac_r.tif', 'render_l.tif', 'render_r.tif',
    'geom.tif',
    'view_dir_l.tif', 'view_dir_r.tif',
    'tan_t_l.tif', 'tan_t_r.tif',
    'mask_l.tif', 'mask_r.tif',
]

OPTIONAL_FILES = [
    'dz_gt.tif',
    'dem_init.tif',
    'sun_dir_l.tif',
    'normal_base_l.tif',
]


def scan_dataset(dataroot):
    """Scan a dataset root for valid sample directories.

    A directory is considered a valid sample if it contains all
    ``REQUIRED_FILES``.

    Args:
        dataroot (str): Root directory to scan.

    Returns:
        list[dict]: Each dict maps column name → absolute file path (or None).
    """
    samples = []
    all_dirs = sorted([
        os.path.join(dataroot, d)
        for d in os.listdir(dataroot)
        if os.path.isdir(os.path.join(dataroot, d))
    ])
    for d in all_dirs:
        missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(d, f))]
        if missing:
            continue  # skip incomplete sample directories
        entry = {}
        for fname in REQUIRED_FILES:
            key = fname.replace('.tif', '')
            entry[key] = os.path.join(d, fname)
        for fname in OPTIONAL_FILES:
            key = fname.replace('.tif', '')
            fpath = os.path.join(d, fname)
            entry[key] = fpath if os.path.isfile(fpath) else ''
        samples.append(entry)
    return samples


def write_csv(samples, out_path):
    """Write sample list to CSV file."""
    col_order = [
        'nac_l', 'nac_r', 'render_l', 'render_r',
        'geom',
        'view_dir_l', 'view_dir_r',
        'tan_t_l', 'tan_t_r',
        'mask_l', 'mask_r',
        'dz_gt', 'dem_init', 'sun_dir_l', 'normal_base_l',
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['# ' + ', '.join(col_order)])  # header comment
        for s in samples:
            row = [s.get(c, '') for c in col_order]
            writer.writerow(row)
    print(f"Wrote {len(samples)} samples to {out_path}")


def dataloader(dataroot, train_list=None, test_list=None, train_ratio=0.9,
               seed=42):
    """Return (train_samples, test_samples) lists.

    If ``train_list`` / ``test_list`` CSV files are provided they are read
    directly.  Otherwise the dataroot is scanned and a random split is made.

    Args:
        dataroot    (str): Dataset root directory.
        train_list  (str | None): Path to pre-existing train CSV.
        test_list   (str | None): Path to pre-existing test CSV.
        train_ratio (float): Fraction of samples for training (auto-split).
        seed        (int): RNG seed for reproducible split.

    Returns:
        (list[dict], list[dict])
    """
    if train_list is not None and test_list is not None:
        train_samples = _read_csv(train_list)
        test_samples  = _read_csv(test_list)
    else:
        samples = scan_dataset(dataroot)
        random.seed(seed)
        random.shuffle(samples)
        n_train = int(len(samples) * train_ratio)
        train_samples = samples[:n_train]
        test_samples  = samples[n_train:]
    return train_samples, test_samples


def _read_csv(path):
    """Read a CSV file produced by ``write_csv``; return list of dicts."""
    col_order = [
        'nac_l', 'nac_r', 'render_l', 'render_r',
        'geom',
        'view_dir_l', 'view_dir_r',
        'tan_t_l', 'tan_t_r',
        'mask_l', 'mask_r',
        'dz_gt', 'dem_init', 'sun_dir_l', 'normal_base_l',
    ]
    samples = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            row = [r.strip() for r in row]
            if not row or row[0].startswith('#'):
                continue
            entry = {}
            for i, col in enumerate(col_order):
                entry[col] = row[i] if i < len(row) else ''
            samples.append(entry)
    return samples


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate DOM sample CSV file lists from a dataset root.')
    parser.add_argument('--dataroot', required=True,
                        help='Root directory containing per-scene sub-dirs.')
    parser.add_argument('--out_train', default='train_dom.csv',
                        help='Output path for training CSV.')
    parser.add_argument('--out_test',  default='test_dom.csv',
                        help='Output path for test/validation CSV.')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Fraction of scenes used for training.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible split.')
    args = parser.parse_args()

    train_s, test_s = dataloader(
        args.dataroot, train_ratio=args.train_ratio, seed=args.seed)

    write_csv(train_s, args.out_train)
    write_csv(test_s,  args.out_test)
