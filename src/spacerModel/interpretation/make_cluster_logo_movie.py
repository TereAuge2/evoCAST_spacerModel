from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import numpy as np

# MP4 needs the imageio FFMPEG plugin: pip install "imageio[ffmpeg]"  (GIF usually works without it.)

LogoMovieMode = Literal["clusters", "by_locus", "splits"]


def _uniform_frame_shapes(images: list[np.ndarray]) -> list[np.ndarray]:
    """
    Center each frame on a white canvas of (max_h, max_w).

    GIF/MP4 writers require every frame to have the same H×W; per-locus (or mixed) logos
    often differ in size.
    """
    if len(images) <= 1:
        return images
    max_h = max(im.shape[0] for im in images)
    max_w = max(im.shape[1] for im in images)
    if all(im.shape[0] == max_h and im.shape[1] == max_w for im in images):
        return images

    out: list[np.ndarray] = []
    for im in images:
        h, w = im.shape[:2]
        if h == max_h and w == max_w:
            out.append(im)
            continue
        if im.ndim == 2:
            canvas = np.full((max_h, max_w), 255, dtype=im.dtype)
        elif im.shape[2] == 4:
            canvas = np.full((max_h, max_w, 4), 255, dtype=im.dtype)
        else:
            canvas = np.full((max_h, max_w, 3), 255, dtype=im.dtype)
        y0 = (max_h - h) // 2
        x0 = (max_w - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = im
        out.append(canvas)
    return out


def collect_logo_movie_frames(logo_dir: Path, mode: LogoMovieMode) -> list[Path]:
    """Return ordered PNG paths for one movie kind; empty list if nothing to render."""
    logo_dir = Path(logo_dir)
    if mode == "clusters":
        return sorted(
            logo_dir.glob("logo_cluster*.png"),
            key=lambda p: int("".join(filter(str.isdigit, p.stem)) or "-1"),
        )
    if mode == "by_locus":
        locus_dir = logo_dir / "by_locus"
        if not locus_dir.is_dir():
            return []
        return sorted(locus_dir.glob("logo_locus_*.png"))
    # splits: full library, then logo_cv1..logo_cvK (any K), then test — matches analyze_dataset exports
    def _cv_index(p: Path) -> int:
        m = re.search(r"cv(\d+)", p.stem, re.I)
        return int(m.group(1)) if m else -1

    out: list[Path] = []
    full_lib = logo_dir / "logo_full_library.png"
    if full_lib.exists():
        out.append(full_lib)
    out.extend(sorted(logo_dir.glob("logo_cv*.png"), key=_cv_index))
    test_logo = logo_dir / "logo_test.png"
    if test_logo.exists():
        out.append(test_logo)
    return out


def _write_logo_movie_files(
    frames: list[Path],
    gif_path: Path,
    mp4_path: Path,
    *,
    fps: float,
    no_mp4: bool,
    log: Callable[[str], None],
) -> None:
    images = [imageio.imread(p) for p in frames]
    images = _uniform_frame_shapes(images)
    frame_seconds = 1.0 / float(fps)
    imageio.mimsave(gif_path, images, duration=frame_seconds)
    log(f"Wrote {gif_path}")
    if no_mp4:
        return
    try:
        imageio.mimwrite(mp4_path, images, fps=float(fps))
        log(f"Wrote {mp4_path}")
    except (ValueError, OSError) as e:
        log(
            f"Skipped MP4 ({gif_path.parent.name}/{mp4_path.name}): {e}. "
            f"For MP4: pip install 'imageio[ffmpeg]'"
        )


def write_logo_movies(
    logo_dir: Path,
    *,
    fps: float = 3.0,
    no_mp4: bool = False,
    log: Callable[[str], None] | None = None,
    modes: Sequence[LogoMovieMode] | None = None,
) -> None:
    """
    Write cluster_logos / locus_logos / split_logos GIF (+ MP4 unless ``no_mp4``).

    Skips a mode when there are no frames (and logs). ``modes`` defaults to all three.
    """
    _log = log or print
    logo_dir = Path(logo_dir)
    if not logo_dir.is_dir():
        _log(f"Logo movies: skip (not a directory: {logo_dir})")
        return

    want: tuple[LogoMovieMode, ...] = tuple(modes) if modes is not None else ("clusters", "by_locus", "splits")
    out_for: dict[LogoMovieMode, tuple[str, str]] = {
        "clusters": ("cluster_logos.gif", "cluster_logos.mp4"),
        "by_locus": ("locus_logos.gif", "locus_logos.mp4"),
        "splits": ("split_logos.gif", "split_logos.mp4"),
    }

    for mode in want:
        frames = collect_logo_movie_frames(logo_dir, mode)
        gif_name, mp4_name = out_for[mode]
        if not frames:
            _log(f"Logo movies ({mode}): skip (no frames)")
            continue
        _write_logo_movie_files(
            frames,
            logo_dir / gif_name,
            logo_dir / mp4_name,
            fps=fps,
            no_mp4=no_mp4,
            log=_log,
        )


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
        choices=["all", "clusters", "splits", "by_locus"],
        default="all",
        help=(
            "all: cluster + locus + split movies; clusters: logo_cluster*.png; "
            "splits: full/cv/test in --logo-dir; by_locus: logo_locus_*.png in --logo-dir/by_locus"
        ),
    )
    ap.add_argument(
        "--out-mp4",
        default="cluster_logos.mp4",
        help="Output MP4 filename (relative to logo-dir); only for single-mode (not --which all)",
    )
    ap.add_argument(
        "--out-gif",
        default="cluster_logos.gif",
        help="Output GIF filename (relative to logo-dir); only for single-mode (not --which all)",
    )
    ap.add_argument(
        "--no-mp4",
        action="store_true",
        help="Only write GIF (skip MP4; avoids needing imageio[ffmpeg])",
    )
    args = ap.parse_args(argv)

    logo_dir = Path(args.logo_dir)
    if not logo_dir.exists():
        raise SystemExit(f"Logo directory not found: {logo_dir}")

    if args.which == "all":
        write_logo_movies(logo_dir, fps=float(args.fps), no_mp4=bool(args.no_mp4))
        return 0

    frames = collect_logo_movie_frames(logo_dir, args.which)
    if not frames:
        raise SystemExit(_no_frames_exit_message(args.which, logo_dir))

    if args.which == "clusters":
        out_mp4_name = args.out_mp4
        out_gif_name = args.out_gif
    elif args.which == "by_locus":
        out_mp4_name = args.out_mp4 if args.out_mp4 != "cluster_logos.mp4" else "locus_logos.mp4"
        out_gif_name = args.out_gif if args.out_gif != "cluster_logos.gif" else "locus_logos.gif"
    else:
        out_mp4_name = "split_logos.mp4"
        out_gif_name = "split_logos.gif"

    _write_logo_movie_files(
        frames,
        logo_dir / out_gif_name,
        logo_dir / out_mp4_name,
        fps=float(args.fps),
        no_mp4=bool(args.no_mp4),
        log=print,
    )
    return 0


def _no_frames_exit_message(which: str, logo_dir: Path) -> str:
    if which == "clusters":
        return f"No logo_cluster*.png files found in {logo_dir}"
    if which == "by_locus":
        return f"No logo_locus_*.png files found in {logo_dir / 'by_locus'}"
    return f"No non-cluster logos (logo_full_library/logo_cv*/logo_test) found in {logo_dir}"


if __name__ == "__main__":
    raise SystemExit(main())
