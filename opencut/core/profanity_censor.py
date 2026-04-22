"""
OpenCut Profanity Censor v1.28.0

Multi-mode profanity censoring: bleep, silence, mute_speaker.
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_DEFAULT_WORDS = [
    "fuck", "shit", "ass", "bitch", "bastard", "damn", "crap",
    "hell", "piss", "dick", "cock", "cunt", "whore", "slut",
]

_CUSTOM_WORDS_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "custom_profanity.txt")


def check_profanity_censor_available() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


@dataclass
class CensorResult:
    output: str = ""
    censor_count: int = 0
    censored_words: List[str] = field(default_factory=list)
    mode: str = "bleep"
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str):
        return getattr(self, k)

    def keys(self):
        return ("output", "censor_count", "censored_words", "mode", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def list_wordlists() -> Dict:
    custom_words = []
    if os.path.isfile(_CUSTOM_WORDS_PATH):
        with open(_CUSTOM_WORDS_PATH, encoding="utf-8") as f:
            custom_words = [l.strip() for l in f if l.strip()]
    return {"default": _DEFAULT_WORDS, "custom": custom_words, "custom_path": _CUSTOM_WORDS_PATH}


def censor(
    audio_path: str,
    transcript_segments: List[Dict],
    mode: str = "bleep",
    custom_words: Optional[List[str]] = None,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> CensorResult:
    """Censor profanity in audio. transcript_segments: list of {word, start, end}"""
    wordlist = set(w.lower() for w in _DEFAULT_WORDS)
    if custom_words:
        wordlist.update(w.lower() for w in custom_words)
    wl = list_wordlists()
    wordlist.update(w.lower() for w in wl.get("custom", []))

    targets = [
        seg for seg in transcript_segments
        if str(seg.get("word", "")).lower().strip(".,!?;:\"'") in wordlist
    ]

    if not targets:
        if output is None:
            output = audio_path
        return CensorResult(output=output, censor_count=0, censored_words=[], mode=mode,
                            notes=["No profanity detected"])

    if output is None:
        base, ext = os.path.splitext(audio_path)
        output = f"{base}_censored{ext or '.wav'}"

    if on_progress:
        on_progress(10, f"Censoring {len(targets)} words via {mode}")

    if mode == "bleep":
        bleep_inputs = []
        for i, seg in enumerate(targets):
            start_ms = int(float(seg["start"]) * 1000)
            dur = max(0.05, float(seg["end"]) - float(seg["start"]))
            bleep_inputs.append(
                f"aevalsrc=0.3*sin(2*PI*1000*t):d={dur:.4f}:s=44100[b{i}]; "
                f"[b{i}]adelay={start_ms}|{start_ms}[bd{i}]; "
            )
        fc = "".join(bleep_inputs)
        labels = "".join(f"[bd{i}]" for i in range(len(targets)))
        fc += f"[0:a]{labels}amix=inputs={len(targets)+1}:normalize=0[outa]"
        cmd = ["ffmpeg", "-y", "-i", audio_path, "-filter_complex", fc, "-map", "[outa]", output]
    else:
        vol_expr = "+".join(
            f"between(t,{float(s['start']):.4f},{float(s['end']):.4f})" for s in targets
        )
        fc = (f"[0:a]volume=0:enable='{vol_expr}'[outa]" if len(targets) == 1
              else f"[0:a]volume='if(gt({vol_expr},0),0,1)'[outa]")
        cmd = ["ffmpeg", "-y", "-i", audio_path, "-filter_complex", fc, "-map", "[outa]", output]

    subprocess.run(cmd, capture_output=True, check=True)

    if on_progress:
        on_progress(100, "Done")

    return CensorResult(output=output, censor_count=len(targets),
                        censored_words=[str(s.get("word", "")) for s in targets], mode=mode, notes=[])


__all__ = ["check_profanity_censor_available", "CensorResult", "list_wordlists", "censor"]
