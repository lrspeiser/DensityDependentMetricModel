#!/usr/bin/env python3
"""
Compose a dual-case SPARC panel contrasting LSB extended galaxies vs HSB/typical galaxies
using the already-generated overlays.

Usage:
  python tools/make_dual_case_panel.py \
    --lsb images/lsb_candidates/overlays/UGC00128_overlay.png \
           images/lsb_candidates/overlays/UGC05005_overlay.png \
           images/lsb_candidates/overlays/UGC01230_overlay.png \
    --hsb images/lsb_candidates/overlays/NGC5055_overlay.png \
           images/lsb_candidates/overlays/NGC2841_overlay.png \
           images/lsb_candidates/overlays/NGC3198_overlay.png \
    --out images/next_steps/enhanced_20250805_115400/sparc_panel_dual_cases.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageFont


def load_and_scale(paths: List[Path], max_width: int) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        w, h = im.size
        if w > max_width:
            scale = max_width / float(w)
            im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        imgs.append(im)
    return imgs


def stack_column(images: List[Image.Image], title: str, pad: int = 12, title_h: int = 40) -> Image.Image:
    if not images:
        raise ValueError('No images for column')
    cell_w = max(im.size[0] for im in images)
    cell_h = max(im.size[1] for im in images)
    W = cell_w + 2*pad
    H = len(images) * (cell_h + pad) + pad + title_h
    col = Image.new('RGB', (W, H), color=(255,255,255))
    draw = ImageDraw.Draw(col)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # Title
    if font is not None:
        draw.text((pad, pad), title, fill=(0,0,0), font=font)
    # Place images
    y = pad + title_h
    for im in images:
        x = pad + (cell_w - im.size[0])//2
        col.paste(im, (x, y))
        y += im.size[1] + pad
    return col


def make_dual_panel(lsb_paths: List[Path], hsb_paths: List[Path], out: Path, max_width: int = 900) -> None:
    lsb_imgs = load_and_scale(lsb_paths, max_width)
    hsb_imgs = load_and_scale(hsb_paths, max_width)
    col_l = stack_column(lsb_imgs, 'Low Surface Brightness (extended, flat outer RC)')
    col_r = stack_column(hsb_imgs, 'High/Typical Surface Brightness')
    pad = 12
    W = col_l.size[0] + col_r.size[0] + 3*pad
    H = max(col_l.size[1], col_r.size[1]) + 2*pad
    panel = Image.new('RGB', (W, H), color=(255,255,255))
    panel.paste(col_l, (pad, pad))
    panel.paste(col_r, (pad + col_l.size[0] + pad, pad))
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out, format='PNG')
    print(f'Saved panel: {out}')


def main():
    ap = argparse.ArgumentParser(description='Make dual-case SPARC panel (LSB vs HSB).')
    ap.add_argument('--lsb', nargs='+', required=True, help='List of LSB overlay PNGs (3 recommended)')
    ap.add_argument('--hsb', nargs='+', required=True, help='List of HSB overlay PNGs (3 recommended)')
    ap.add_argument('--out', required=True, help='Output PNG path')
    ap.add_argument('--max-width', type=int, default=900)
    args = ap.parse_args()
    make_dual_panel([Path(p) for p in args.lsb], [Path(p) for p in args.hsb], Path(args.out), max_width=int(args.max_width))


if __name__ == '__main__':
    main()
