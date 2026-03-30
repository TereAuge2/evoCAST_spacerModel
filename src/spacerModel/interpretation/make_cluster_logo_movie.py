from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logo-dir",
        default="output/data_processing/logos",
        help="Directory containing logo_cluster*.png frames",
    )
    ap.add_argument("--fps", type=float, default=3.0, help="Frames per second for the movie")
    ap.add_argument(
        "--which",
        choices=["clusters", "splits"],
        default="clusters",
        help="Which logos to animate: cluster logos or non-cluster (cv1-5, full, test)",
    )
    ap.add_argument(
        "--out-mp4",
        default="cluster_logos.mp4",
        help="Output MP4 filename (relative to logo-dir)",
    )
    ap.add_argument(
        "--out-gif",
        default="cluster_logos.gif",
        help="Output GIF filename (relative to logo-dir)",
    )
    args = ap.parse_args(argv)

    logo_dir = Path(args.logo_dir)
    if not logo_dir.exists():
        raise SystemExit(f"Logo directory not found: {logo_dir}")

    if args.which == "clusters":
        frames = sorted(
            logo_dir.glob("logo_cluster*.png"),
            key=lambda p: int("".join(filter(str.isdigit, p.stem)) or "-1"),
        )
        if not frames:
            raise SystemExit(f"No logo_cluster*.png files found in {logo_dir}")
        out_mp4_name = args.out_mp4
        out_gif_name = args.out_gif
    else:
        # Non-cluster logos: full library, cv1-5, test (in that order if present)
        order_names = ["logo_full_library.png"] + [f"logo_cv{i}.png" for i in range(1, 6)] + ["logo_test.png"]
        frames = []
        for name in order_names:
            p = logo_dir / name
            if p.exists():
                frames.append(p)
        if not frames:
            raise SystemExit(f"No non-cluster logos (logo_full_library/logo_cv*/logo_test) found in {logo_dir}")
        out_mp4_name = "split_logos.mp4"
        out_gif_name = "split_logos.gif"

    images = [imageio.imread(p) for p in frames]

    mp4_path = logo_dir / out_mp4_name
    gif_path = logo_dir / out_gif_name

    imageio.mimwrite(mp4_path, images, fps=float(args.fps))
    imageio.mimsave(gif_path, images, duration=1.0 / float(args.fps))

    print(f"Wrote {mp4_path}")
    print(f"Wrote {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

