"""Render the project's short film: video/film.html drawn at 30 frames a second, each frame
screenshot by a headless browser, the frames encoded to H.264 MP4 and video/sound.py's track
laid under them.

    node video/trace.mjs
    uv run --no-project --with playwright --with imageio-ffmpeg --with numpy python video/render.py

trace.mjs replays the pair from demo/data/*.json (written by `mselect demo build`) with the
page's own code and writes video/trace.json, which the film is drawn from; run it again when
the page's data changes. The browser is the machine's own Chrome (Playwright's `channel`; pass
another, such as msedge, as the first argument) and the ffmpeg is imageio-ffmpeg's, so neither
needs installing.

Writes video/out/film.mp4 (with sound) and video/out/film-poster.jpg, gitignored, with the
page's address on the last screen, for posting. `--site` leaves the address off, for a copy
embedded on the page itself.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FPS = 30
SECONDS = 37.0  # film.html's END
POSTER_AT = 2.0  # the frame the page shows before the film is played
# The names the film prints; the vendors' identifiers are longer than a card.
NAMES = ["DeepSeek-V4.1-Flash", "Jev 1.14"]


def main() -> None:
    import imageio_ffmpeg  # type: ignore[import-untyped]
    from playwright.sync_api import sync_playwright

    site = "--site" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    channel = args[0] if args else "chrome"
    trace = json.loads((HERE / "trace.json").read_text(encoding="utf8"))
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    frames = Path(tempfile.mkdtemp(prefix="film-frames-"))
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=channel)
            page = browser.new_page(viewport={"width": 1080, "height": 1080})
            page.goto((HERE / "film.html").as_uri())
            page.evaluate(
                "d => { window.D = d.trace; window.NAMES = d.names; }",
                {"trace": trace, "names": NAMES},
            )
            page.evaluate("k => { window.TIMING = k; }", {"site": site})
            page.evaluate("document.fonts.ready")
            for i in range(round(SECONDS * FPS)):
                page.evaluate("t => render(t)", i / FPS)
                page.screenshot(path=str(frames / f"f{i:04d}.png"))
            page.evaluate("t => render(t)", POSTER_AT)
            page.screenshot(path=str(out / "film-poster.jpg"), type="jpeg", quality=86)
            browser.close()

        sys.path.insert(0, str(HERE))
        import sound

        adaptive = len(trace["adaptive"]["steps"])
        left, right = sound.track(adaptive, adaptive, len(trace["random"]["steps"]), SECONDS)
        sound.write_wav(frames / "track.wav", left, right)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-framerate",
                str(FPS),
                "-i",
                str(frames / "f%04d.png"),
                "-i",
                str(frames / "track.wav"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                "-preset",
                "slow",
                "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-ar",
                "48000",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(out / "film.mp4"),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(frames, ignore_errors=True)
    print(f"wrote film.mp4 and film-poster.jpg to {out}")


if __name__ == "__main__":
    main()
