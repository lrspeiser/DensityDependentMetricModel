#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont


def load_images(paths: List[Path], max_width: int) -> List[Tuple[Image.Image, str]]:
    imgs = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        w, h = im.size
        if w > max_width:
            scale = max_width / float(w)
            im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        imgs.append((im, p.stem))
    return imgs


def make_panel(images: List[Tuple[Image.Image, str]], cols: int, rows: int, pad: int = 12, title_h: int = 32) -> Image.Image:
    # Determine cell size
    cell_w = max(im.size[0] for im, _ in images)
    cell_h = max(im.size[1] for im, _ in images) + title_h
    W = cols * cell_w + (cols + 1) * pad
    H = rows * cell_h + (rows + 1) * pad
    panel = Image.new('RGB', (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for idx, (im, name) in enumerate(images[: cols*rows]):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + pad)
        # Title
        label = name.replace('sparc_overlay_', '').replace('_', ' ')
        if font is not None:
            draw.text((x0, y0), label, fill=(0, 0, 0), font=font)
        # Image below title
        panel.paste(im, (x0, y0 + title_h))
    return panel


def main():
    ap = argparse.ArgumentParser(description='Compose a panel of SPARC overlays (RAR vs GR vs data).')
    ap.add_argument('--images-root', required=True, help='Directory containing sparc_overlay_*.png files')
    ap.add_argument('--pattern', default='sparc_overlay_*.png', help='Glob pattern for overlays')
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--rows', type=int, default=2)
    ap.add_argument('--max-width', type=int, default=900, help='Max width to scale each tile to')
    ap.add_argument('--out', required=True, help='Output panel PNG path')
    args = ap.parse_args()

    root = Path(args.images_root)
    paths = sorted(root.glob(args.pattern))
    if not paths:
        raise SystemExit(f'No images match {args.pattern} under {root}')

    imgs = load_images(paths, max_width=int(args.max_width))
    panel = make_panel(imgs, cols=int(args.cols), rows=int(args.rows))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    panel.save(outp, format='PNG')
    print(f'Saved {outp}')


if __name__ == '__main__':
    main()

