#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
from PIL import Image


def make_grid(images: list[Path], rows: int, cols: int, out: Path, pad: int = 10, bg=(255, 255, 255)) -> None:
    # Load images (some slots can be None if fewer images than grid slots)
    imgs = [Image.open(p).convert('RGB') for p in images]
    # Resize all to the same size (use the first as reference)
    w0, h0 = imgs[0].size
    imgs = [im.resize((w0, h0), Image.BICUBIC) for im in imgs]
    total_slots = rows * cols
    if len(imgs) < total_slots:
        # Fill with blank placeholders
        blanks = total_slots - len(imgs)
        for _ in range(blanks):
            imgs.append(Image.new('RGB', (w0, h0), color=bg))
    # Canvas size with padding
    W = cols * w0 + (cols + 1) * pad
    H = rows * h0 + (rows + 1) * pad
    canvas = Image.new('RGB', (W, H), color=bg)
    # Paste
    k = 0
    for r in range(rows):
        for c in range(cols):
            x = pad + c * (w0 + pad)
            y = pad + r * (h0 + pad)
            canvas.paste(imgs[k], (x, y))
            k += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format='PNG')
    print(f'Saved grid: {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', nargs='+', required=True, help='Input PNGs in order')
    ap.add_argument('--rows', type=int, default=2)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--out', required=True, help='Output PNG path')
    args = ap.parse_args()

    paths = [Path(p) for p in args.images]
    out = Path(args.out)
    make_grid(paths, args.rows, args.cols, out)


if __name__ == '__main__':
    main()
