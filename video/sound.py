"""The film's sound track, synthesised: a soft pad under the whole film, and a sound for each
thing that happens on screen, timed by the schedule video/film.html draws to (its T).

- the two models appear: a note as each card lands, a wooden knock for "vs";
- the two title screens: a soft bell for the usual way, a brighter one for the better way;
- the wall of questions lights up: a rain of soft ticks that speeds up with the meter;
- the questions fly into the cloud: a sweep, and a low note as the weak ones turn red;
- the race: a wooden note for each adaptive question, climbing as the estimates rise, and a
  soft tick for each random one; a chord when the adaptive test separates the pair, and a
  thud when the random one runs out of budget;
- the result: a swell landing on the headline with a low impact and a bell chord, a note per
  card, and a long bell under the address.

The instruments are the ones projects 10 and 17 use in their films. 48 kHz stereo, peak at
-1 dBFS. numpy only.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

SR = 48_000
Arr = npt.NDArray[np.float64]

# A minor pentatonic from A4 up: the adaptive questions climb it.
SCALE = [440.0, 523.25, 587.33, 659.26, 783.99, 880.0, 1046.5, 1174.66, 1318.51, 1567.98, 1760.0]


def _t(n: int) -> Arr:
    return np.arange(n) / SR


def pluck(freq: float, dur: float = 0.9, decay: float = 5.0, bright: float = 0.35) -> Arr:
    """A plucked note: a fundamental with two soft harmonics and an exponential decay."""
    t = _t(int(dur * SR))
    tone = (
        np.sin(2 * np.pi * freq * t)
        + bright * np.sin(4 * np.pi * freq * t)
        + 0.12 * np.sin(6 * np.pi * freq * t)
    )
    env = np.minimum(t / 0.006, 1.0) * np.exp(-decay * t)
    out: Arr = tone * env
    return out


def thud(freq: float = 82.0, dur: float = 0.45) -> Arr:
    t = _t(int(dur * SR))
    out: Arr = (
        np.sin(2 * np.pi * freq * t * (1 - 0.25 * t)) * np.minimum(t / 0.004, 1) * np.exp(-9 * t)
    )
    return out


def tick(rng: np.random.Generator, dur: float = 0.05) -> Arr:
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    smooth = np.convolve(noise, np.ones(12) / 12, mode="same")  # a soft click, not a hiss
    out: Arr = smooth * np.exp(-60 * _t(n))
    return out


def sweep(f0: float, f1: float, dur: float) -> Arr:
    t = _t(int(dur * SR))
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))
    env = np.sin(np.pi * t / dur) ** 2
    out: Arr = np.sin(phase) * env
    return out


def bell(freq: float, dur: float = 3.0, index: float = 2.2, ratio: float = 1.41) -> Arr:
    """An FM bell: a carrier modulated at an inharmonic ratio, the brightness fading first."""
    t = _t(int(dur * SR))
    mod = index * np.exp(-2.5 * t) * np.sin(2 * np.pi * freq * ratio * t)
    env = np.minimum(t / 0.003, 1.0) * np.exp(-1.6 * t)
    out: Arr = np.sin(2 * np.pi * freq * t + mod) * env
    return out


def marimba(freq: float, dur: float = 0.8) -> Arr:
    """A wooden note: the fundamental, a fourth partial and a little of the tenth, dying fast."""
    t = _t(int(dur * SR))
    tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(8 * np.pi * freq * t) * np.exp(-25 * t)
    tone += 0.06 * np.sin(20 * np.pi * freq * t) * np.exp(-40 * t)
    out: Arr = tone * np.minimum(t / 0.002, 1.0) * np.exp(-7 * t)
    return out


def impact(dur: float = 1.8) -> Arr:
    """A low hit: a sine falling from 62 to 41 Hz under a short soft burst."""
    t = _t(int(dur * SR))
    f = 41 + 21 * np.exp(-6 * t)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-2.2 * t)
    rng = np.random.default_rng(3)
    burst = np.convolve(rng.standard_normal(len(t)), np.ones(24) / 24, mode="same") * np.exp(
        -30 * t
    )
    out: Arr = (body + 0.6 * burst) * np.minimum(t / 0.002, 1.0)
    return out


def riser(dur: float) -> Arr:
    """A swell that grows to its end: soft noise and a tone rising an octave, cut at the top."""
    t = _t(int(dur * SR))
    rng = np.random.default_rng(5)
    noise = np.convolve(rng.standard_normal(len(t)), np.ones(16) / 16, mode="same")
    tone = np.sin(2 * np.pi * (220 * t + 220 * t * t / (2 * dur)))
    out: Arr = (0.7 * noise + 0.5 * tone) * (t / dur) ** 2.5
    return out


def pad(dur: float, chord: list[float], fade: float) -> Arr:
    t = _t(int(dur * SR))
    out = np.zeros_like(t)
    for i, f in enumerate(chord):
        lfo = 0.75 + 0.25 * np.sin(2 * np.pi * (0.13 + 0.05 * i) * t + i)
        out += lfo * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 1.003 * t))
    env = np.minimum(np.minimum(t / fade, 1.0), np.minimum((dur - t) / fade, 1.0))
    res: Arr = out / len(chord) * env
    return res


# film.html's schedule (its T, RUN_A, RUN_B), in film.html's seconds.
ASK, INTRO1, WALL, CLOUD, INTRO2, RACE, RESULT = 0.0, 3.4, 6.6, 10.2, 14.4, 19.6, 28.0
RUN_A, RUN_B = (19.9, 23.5), (24.3, 26.8)


def asked_at(k: int, n: int, run: tuple[float, float], power: float) -> float:
    """When film.html shows question k of n arriving: it eases them in as (u ** power)."""
    return run[0] + (k / n) ** (1 / power) * (run[1] - run[0])


def track(adaptive: int, random_from: int, random_to: int, seconds: float) -> tuple[Arr, Arr]:
    """Left and right channels for a film of `seconds`. `adaptive` is how many questions the
    adaptive run asked; the random run carries on alone from `random_from` to `random_to`."""
    n = int(seconds * SR)
    left, right = np.zeros(n), np.zeros(n)
    rng = np.random.default_rng(2)

    def add(sig: Arr, at: float, gain: float, pan: float = 0.0) -> None:
        i = int(at * SR)
        if i >= n or i < 0:
            return
        j = min(n, i + len(sig))
        left[i:j] += sig[: j - i] * gain * (1 - pan) ** 0.5
        right[i:j] += sig[: j - i] * gain * (1 + pan) ** 0.5

    # The pad: A minor while the story is told, a brighter D major under the result.
    add(pad(RESULT + 0.6, [110.0, 164.81, 220.0, 261.63], 1.0), 0.0, 0.10)
    add(pad(seconds - RESULT, [73.42, 146.83, 220.0, 329.63, 369.99], 1.2), RESULT, 0.11)

    # 1. The two cards land, then "vs".
    add(pluck(SCALE[2], 1.0, 4, 0.4), 0.45, 0.14, -0.4)
    add(pluck(SCALE[4], 1.0, 4, 0.4), 0.8, 0.14, 0.4)
    add(marimba(SCALE[0] / 2), 1.2, 0.2)

    # The two title screens: a soft bell under the question, a brighter one for the better way.
    add(bell(220.0, 3.0, 1.0, 2.0), INTRO1 + 0.15, 0.10)
    add(pluck(SCALE[1] / 2, 1.2, 3, 0.3), INTRO1 + 0.7, 0.08)
    add(riser(0.6), INTRO2 + 0.15 - 0.6, 0.03)
    add(bell(440.0, 3.0, 1.2), INTRO2 + 0.15, 0.08)
    for i, f in enumerate((659.26, 783.99, 880.0)):
        add(marimba(f), INTRO2 + 1.15 + 1.0 * i, 0.10, -0.3 + 0.3 * i)

    # 2. The wall lights up: ticks that speed up as the meter runs.
    t, gap = WALL + 0.3, 0.16
    while t < WALL + 3.2:
        add(tick(rng, 0.03), t, 0.22, float(rng.uniform(-0.5, 0.5)))
        t += gap
        gap = max(0.035, gap * 0.93)
    add(sweep(180, 420, 2.6), WALL + 0.4, 0.04)
    add(thud(70, 0.6), WALL + 3.25, 0.35)

    # 3. The dots fly into the cloud; the weak ones turn red.
    add(sweep(300, 760, 1.3), CLOUD + 0.05, 0.07)
    for i in range(5):
        add(pluck(SCALE[i] / 2, 0.7, 6, 0.3), CLOUD + 0.9 + 0.08 * i, 0.07, -0.6 + 0.3 * i)
    add(pluck(SCALE[0] / 4, 1.2, 3, 0.1), CLOUD + 1.65, 0.16)
    add(pluck(SCALE[0] / 2, 1.0, 4, 0.2), CLOUD + 1.5, 0.08)

    # 4. The race. Adaptive questions: a wooden note each, climbing the scale.
    for k in range(1, adaptive + 1):
        at = asked_at(k, adaptive, RUN_A, 1.6)
        note = SCALE[min(len(SCALE) - 1, int(k / adaptive * (len(SCALE) - 1)))]
        add(marimba(note, 0.5), at, 0.07 if k > 12 else 0.11, -0.25)
    # Random questions, at the same pace up to the adaptive answer, then on their own: a tick
    # each, thinned where they arrive faster than the ear separates them.
    last = -1.0
    for k in range(1, adaptive + 1):
        at = asked_at(k, adaptive, RUN_A, 1.6)
        if at - last > 0.05:
            add(tick(rng, 0.03), at, 0.16, 0.35)
            last = at
    for k in range(random_from + 1, random_to + 1):
        u = (k - random_from) / (random_to - random_from)
        at = RUN_B[0] + u ** (1 / 1.15) * (RUN_B[1] - RUN_B[0])
        if at - last > 0.05:
            add(tick(rng, 0.03), at, 0.16, 0.35)
            last = at
    # The adaptive test separates the pair: a chord and a soft impact.
    add(impact(1.2), RUN_A[1] + 0.05, 0.12)
    for f, pan in ((587.33, -0.3), (739.99, 0.0), (880.0, 0.3)):
        add(marimba(f), RUN_A[1] + 0.1, 0.14, pan)
    # The random run reaches its budget unseparated.
    add(thud(), RUN_B[1] + 0.1, 0.5)
    add(pluck(SCALE[0] / 2, 0.9, 5, 0.1), RUN_B[1] + 0.1, 0.12)

    # The result: the swell lands on the winner (film.html scales it in from 0.4 s in), a
    # second hit for "so did adaptive testing", a note per figure, a bell under the address.
    add(riser(1.1), RESULT + 0.45 - 1.1, 0.05)
    add(impact(), RESULT + 0.45, 0.22)
    for f, pan in ((587.33, -0.4), (739.99, 0.0), (880.0, 0.4)):
        add(bell(f, 3.2, 1.2), RESULT + 0.45, 0.045, pan)
    add(impact(1.2), RESULT + 1.15, 0.12)
    for f, pan in ((739.99, -0.3), (880.0, 0.0), (1108.73, 0.3)):
        add(marimba(f), RESULT + 1.15, 0.12, pan)
    for i, (f, pan) in enumerate(((880.0, -0.35), (1046.5, 0.0), (1174.66, 0.35))):
        add(marimba(f), RESULT + 2.05 + 0.45 * i, 0.15, pan)
    add(bell(293.66, 4.0, 1.4, 2.0), RESULT + 4.1, 0.16)
    add(bell(1174.66, 3.0, 1.0), RESULT + 4.15, 0.04, 0.2)

    # A little room: three soft echoes, a fade at the end, then normalise.
    for d, gain in ((0.083, 0.22), (0.131, 0.15), (0.197, 0.1)):
        k = int(d * SR)
        left[k:] += gain * right[:-k]
        right[k:] += gain * left[:-k]
    tail = np.minimum(1.0, (seconds - _t(n)) / 0.6)
    left, right = left * tail, right * tail
    peak = max(np.abs(left).max(), np.abs(right).max())
    scale = 10 ** (-1 / 20) / peak
    return left * scale, right * scale


def write_wav(path: Path, left: Arr, right: Arr) -> None:
    pcm = (np.stack([left, right], axis=1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
