"""
Normalize playback timing across all gallery GIFs.

Every animation is retimed to a common ITERS_PER_SEC (placement iterations
per wall-clock second), regardless of the --interval it was captured at,
and the final-frame hold is padded so all GIFs share the same total loop
period — side by side in the README they advance and restart in sync.

Idempotent: re-running on already-retimed GIFs is a no-op (timing is
recomputed from CAPTURE_INTERVAL, not from the current frame durations).

Usage:
    python scripts/retime_gifs.py
"""

from pathlib import Path

from PIL import Image, ImageSequence

PROJECT_ROOT = Path(__file__).parent.parent

ITERS_PER_SEC = 250   # common playback rate
MIN_FINAL_HOLD = 2000  # ms — never hold the punchline frame shorter than this

# gif directory -> --interval the run was captured at (see docs/RUNNING.md)
CAPTURE_INTERVAL = {
    "fft_1_s42": 25,
    "fft_2_lambda_s42": 25,
    "superblue12_showcase": 50,
}


def collect():
    """[(path, frames[RGB], frame_ms)] for every gallery gif."""
    entries = []
    for dirname, interval in CAPTURE_INTERVAL.items():
        frame_ms = round(interval * 1000 / ITERS_PER_SEC)
        for p in sorted((PROJECT_ROOT / "graphs" / "comparisons" / dirname)
                        .glob("*.gif")):
            frames = [f.convert("RGB")
                      for f in ImageSequence.Iterator(Image.open(p))]
            entries.append((p, frames, frame_ms))
    return entries


def main():
    entries = collect()
    # common loop period = longest retimed body + the minimum final hold
    period = max((len(fr) - 1) * ms for _, fr, ms in entries) + MIN_FINAL_HOLD
    print(f"common loop period: {period / 1000:.1f}s "
          f"at {ITERS_PER_SEC} iters/sec")
    for p, frames, ms in entries:
        body = (len(frames) - 1) * ms
        durations = [ms] * len(frames)
        durations[-1] = period - body  # pad hold so loops stay in phase
        images = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
        images[0].save(p, save_all=True, append_images=images[1:],
                       duration=durations, loop=0, optimize=True)
        print(f"  {p.relative_to(PROJECT_ROOT)}: {len(frames)} frames "
              f"x {ms}ms + {durations[-1]}ms hold")


if __name__ == "__main__":
    main()
