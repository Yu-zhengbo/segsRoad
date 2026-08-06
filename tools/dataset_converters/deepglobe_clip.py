# Copyright (c) OpenMMLab. All rights reserved.
"""Clip DeepGlobe road images and masks into smaller paired patches.

The expected source layout is:

deepglobe/
  images/{train,val}/*.jpg
  annotations/{train,val}/*.png

The output keeps the same layout and only writes train/val splits.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
ANN_EXTS = {'.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg'}
SPLITS = ('train', 'val')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Clip DeepGlobe road dataset into paired patches.')
    parser.add_argument(
        '--data-root',
        default='/data1/datasets/zhengbo/roaddataset/deepglobe',
        help='DeepGlobe dataset root.')
    parser.add_argument(
        '-o',
        '--out-dir',
        default=None,
        help='Output directory. Defaults to <data-root>_clip<clip-size>.')
    parser.add_argument(
        '--clip-size',
        type=int,
        default=512,
        help='Patch size for both image and annotation.')
    parser.add_argument(
        '--stride',
        type=int,
        default=512,
        help='Sliding-window stride. Use a smaller value for overlap.')
    parser.add_argument(
        '--skip-empty',
        action='store_true',
        help='Skip patches whose annotation contains only background value 0.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing patch files.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only check pairs and report patch counts, without writing files.')
    return parser.parse_args()


def list_files(path, suffixes):
    return sorted(p for p in path.iterdir()
                  if p.is_file() and p.suffix.lower() in suffixes)


def build_stem_map(path, suffixes):
    files = list_files(path, suffixes)
    stem_map = {}
    for file in files:
        if file.stem in stem_map:
            raise RuntimeError(
                f'Duplicate stem "{file.stem}" in {path}: '
                f'{stem_map[file.stem].name}, {file.name}')
        stem_map[file.stem] = file
    return stem_map


def get_windows(width, height, clip_size, stride):
    if width < clip_size or height < clip_size:
        raise ValueError(
            f'Image size ({width}, {height}) is smaller than clip size '
            f'{clip_size}.')

    num_cols = max(1, math.ceil((width - clip_size) / stride) + 1)
    num_rows = max(1, math.ceil((height - clip_size) / stride) + 1)
    xs = [min(i * stride, width - clip_size) for i in range(num_cols)]
    ys = [min(i * stride, height - clip_size) for i in range(num_rows)]

    windows = []
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            windows.append((x, y, x + clip_size, y + clip_size))
    return windows


def is_empty_annotation(patch):
    ann = np.asarray(patch)
    return not np.any(ann)


def clip_pair(img_path, ann_path, img_out_dir, ann_out_dir, args):
    with Image.open(img_path) as img, Image.open(ann_path) as ann:
        if img.size != ann.size:
            raise ValueError(
                f'Size mismatch for {img_path.name}: image {img.size}, '
                f'annotation {ann.size}')

        width, height = img.size
        windows = get_windows(width, height, args.clip_size, args.stride)
        written = 0
        skipped = 0
        for x1, y1, x2, y2 in windows:
            ann_patch = ann.crop((x1, y1, x2, y2))
            if args.skip_empty and is_empty_annotation(ann_patch):
                skipped += 1
                continue

            patch_name = f'{img_path.stem}_{x1}_{y1}_{x2}_{y2}'
            img_out = img_out_dir / f'{patch_name}{img_path.suffix.lower()}'
            ann_out = ann_out_dir / f'{patch_name}{ann_path.suffix.lower()}'

            if not args.overwrite and (img_out.exists() or ann_out.exists()):
                raise FileExistsError(
                    f'Output patch already exists: {img_out} or {ann_out}. '
                    'Use --overwrite to replace it.')

            if not args.dry_run:
                img.crop((x1, y1, x2, y2)).save(img_out)
                ann_patch.save(ann_out)
            written += 1

    return written, skipped


def process_split(data_root, out_dir, split, args):
    img_dir = data_root / 'images' / split
    ann_dir = data_root / 'annotations' / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f'Missing image directory: {img_dir}')
    if not ann_dir.is_dir():
        raise FileNotFoundError(f'Missing annotation directory: {ann_dir}')

    img_map = build_stem_map(img_dir, IMG_EXTS)
    ann_map = build_stem_map(ann_dir, ANN_EXTS)

    missing_ann = sorted(set(img_map) - set(ann_map))
    missing_img = sorted(set(ann_map) - set(img_map))
    if missing_ann or missing_img:
        raise RuntimeError(
            f'Pair mismatch in split "{split}": '
            f'{len(missing_ann)} images without annotation, '
            f'{len(missing_img)} annotations without image. '
            f'Examples missing_ann={missing_ann[:5]}, '
            f'missing_img={missing_img[:5]}')

    img_out_dir = out_dir / 'images' / split
    ann_out_dir = out_dir / 'annotations' / split
    if not args.dry_run:
        img_out_dir.mkdir(parents=True, exist_ok=True)
        ann_out_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skipped = 0
    stems = sorted(img_map)
    for idx, stem in enumerate(stems, start=1):
        written, skipped = clip_pair(img_map[stem], ann_map[stem],
                                     img_out_dir, ann_out_dir, args)
        total_written += written
        total_skipped += skipped
        if idx % 100 == 0 or idx == len(stems):
            print(
                f'[{split}] {idx}/{len(stems)} pairs, '
                f'patches={total_written}, skipped_empty={total_skipped}')

    return len(stems), total_written, total_skipped


def main():
    args = parse_args()
    if args.clip_size <= 0:
        raise ValueError('--clip-size must be positive.')
    if args.stride <= 0:
        raise ValueError('--stride must be positive.')

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f'{args.data_root}_clip{args.clip_size}')

    print(f'Data root: {data_root}')
    print(f'Output dir: {out_dir}')
    print(f'clip_size={args.clip_size}, stride={args.stride}')
    if args.dry_run:
        print('Dry run: no files will be written.')

    summary = {}
    for split in SPLITS:
        summary[split] = process_split(data_root, out_dir, split, args)

    print('Done.')
    for split, (num_pairs, written, skipped) in summary.items():
        print(
            f'{split}: pairs={num_pairs}, patches={written}, '
            f'skipped_empty={skipped}')


if __name__ == '__main__':
    main()
