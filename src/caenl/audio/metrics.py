"""Captioning metrics: official ``pycocoevalcap`` values (authoritative) cross-checked against native code.

The native CIDEr-D, BLEU-1..4 and ROUGE-L implementations below follow coco-caption (CIDEr-D with
sigma 6, corpus BLEU with the closest reference length, ROUGE-L with beta 1.2) and are unit-tested
against ``pycocoevalcap``; they exist so that a job can run where Java is unavailable.  Whenever
``pycocoevalcap`` is importable the official values are the ones reported; METEOR and SPICE always
come from the official Java implementations and a job whose protocol requires them fails loudly if
they cannot be computed (no silent NaN).
"""
from __future__ import annotations

import math
import re
import shutil
from collections import Counter, defaultdict
from typing import Iterable, Sequence

_PUNCT = re.compile(r"[^a-z0-9' ]+")


def tokenize(s: str) -> list[str]:
    s = s.lower().replace("-", " ")
    s = _PUNCT.sub(" ", s)
    return s.split()


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


# ----------------------------------------------------------------------------- BLEU (corpus level)
def bleu(candidates: Sequence[str], references: Sequence[Sequence[str]], max_n: int = 4) -> dict[str, float]:
    cand_tok = [tokenize(c) for c in candidates]
    ref_tok = [[tokenize(r) for r in refs] for refs in references]
    out = {}
    for N in range(1, max_n + 1):
        match = [0] * N
        total = [0] * N
        cand_len = 0
        ref_len = 0
        for c, refs in zip(cand_tok, ref_tok):
            cand_len += len(c)
            # closest reference length
            ref_len += min((abs(len(r) - len(c)), len(r)) for r in refs)[1] if refs else 0
            for n in range(1, N + 1):
                cn = _ngrams(c, n)
                maxref: Counter = Counter()
                for r in refs:
                    for g, k in _ngrams(r, n).items():
                        maxref[g] = max(maxref[g], k)
                match[n - 1] += sum(min(k, maxref[g]) for g, k in cn.items())
                total[n - 1] += max(len(c) - n + 1, 0)
        if any(t == 0 for t in total) or any(m == 0 for m in match):
            out[f"bleu{N}"] = 0.0
            continue
        logp = sum(math.log(m / t) for m, t in zip(match, total)) / N
        bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / max(cand_len, 1))
        out[f"bleu{N}"] = bp * math.exp(logp)
    return out


# ----------------------------------------------------------------------------- ROUGE-L
def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(candidates: Sequence[str], references: Sequence[Sequence[str]], beta: float = 1.2) -> float:
    """ROUGE-L as in coco-caption: F-measure of the *maximum* precision and the *maximum* recall over references."""
    scores = []
    for c, refs in zip(candidates, references):
        ct = tokenize(c)
        precs, recs = [], []
        for r in refs:
            rt = tokenize(r)
            l = _lcs(ct, rt)
            precs.append(l / len(ct) if ct else 0.0)
            recs.append(l / len(rt) if rt else 0.0)
        pm, rm = (max(precs) if precs else 0.0), (max(recs) if recs else 0.0)
        scores.append((1 + beta**2) * pm * rm / (rm + beta**2 * pm) if pm > 0 and rm > 0 else 0.0)
    return float(sum(scores) / max(len(scores), 1))


# ----------------------------------------------------------------------------- CIDEr-D
def cider_d(candidates: Sequence[str], references: Sequence[Sequence[str]], n_max: int = 4, sigma: float = 6.0) -> float:
    cand_tok = [tokenize(c) for c in candidates]
    ref_tok = [[tokenize(r) for r in refs] for refs in references]
    # document frequencies over reference sets
    df = [Counter() for _ in range(n_max)]
    for refs in ref_tok:
        for n in range(1, n_max + 1):
            seen = set()
            for r in refs:
                seen.update(_ngrams(r, n).keys())
            for g in seen:
                df[n - 1][g] += 1
    n_img = len(ref_tok)
    log_ref = math.log(max(n_img, 1))

    def vec(tokens, n):
        cnt = _ngrams(tokens, n)
        v = {}
        norm = 0.0
        for g, k in cnt.items():
            w = k * (log_ref - math.log(max(df[n - 1][g], 1)))
            v[g] = w
            norm += w * w
        return v, math.sqrt(norm), len(tokens)

    scores = []
    for c, refs in zip(cand_tok, ref_tok):
        total = 0.0
        for n in range(1, n_max + 1):
            vc, nc, lc = vec(c, n)
            s_n = 0.0
            for r in refs:
                vr, nr, lr = vec(r, n)
                # clipped dot product
                dot = sum(min(vc[g], vr.get(g, 0.0)) * vr.get(g, 0.0) for g in vc)
                val = dot / (nc * nr) if nc > 0 and nr > 0 else 0.0
                val *= math.exp(-((lc - lr) ** 2) / (2 * sigma**2))
                s_n += val
            total += s_n / max(len(refs), 1)
        scores.append(total / n_max * 10.0)
    return float(sum(scores) / max(len(scores), 1))


class CaptionMetricError(RuntimeError):
    """A metric the protocol requires could not be computed (no silent NaN)."""


def _coco_inputs(candidates: Sequence[str], references: Sequence[Sequence[str]], ptb: bool) -> tuple[dict, dict]:
    """Build pycocoevalcap ``gts``/``res`` dicts; with ``ptb`` the official Java PTBTokenizer is used."""
    gts = {i: [{"caption": r} for r in refs] for i, refs in enumerate(references)}
    res = {i: [{"caption": c}] for i, c in enumerate(candidates)}
    if ptb:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # type: ignore

        tok = PTBTokenizer()
        return tok.tokenize(gts), tok.tokenize(res)
    return {i: [" ".join(tokenize(r["caption"])) for r in refs] for i, refs in gts.items()}, {i: [" ".join(tokenize(c["caption"])) for c in cs] for i, cs in res.items()}


def official_metrics(candidates: Sequence[str], references: Sequence[Sequence[str]], java: bool = True, spice: bool = False) -> dict[str, float]:
    """BLEU-1..4, ROUGE-L and CIDEr-D from ``pycocoevalcap`` (the coco-caption reference code), plus METEOR/SPICE when ``java``.

    Raises :class:`CaptionMetricError` if ``pycocoevalcap`` is missing, or if a Java metric was requested
    but cannot be computed.  Tokenisation uses the official PTBTokenizer when Java is available and the
    native tokenizer otherwise (recorded in the result under ``tokenizer``).
    """
    import contextlib
    import io

    try:
        from pycocoevalcap.bleu.bleu import Bleu  # type: ignore
        from pycocoevalcap.cider.cider import Cider  # type: ignore
        from pycocoevalcap.rouge.rouge import Rouge  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise CaptionMetricError(f"pycocoevalcap is not importable ({type(exc).__name__}: {exc}); install it (pip install pycocoevalcap) or set audio.captioning.official_metrics: false") from exc
    have_java = shutil.which("java") is not None
    if java and not have_java:
        raise CaptionMetricError("METEOR/SPICE requested (audio.captioning.java_metrics: true) but no Java runtime is installed")
    gts, res = _coco_inputs(candidates, references, ptb=have_java)
    out: dict[str, float] = {"tokenizer": "ptb" if have_java else "native"}
    with contextlib.redirect_stdout(io.StringIO()):
        b, _ = Bleu(4).compute_score(gts, res)
        for n in range(4):
            out[f"bleu{n + 1}"] = float(b[n])
        out["rouge_l"] = float(Rouge().compute_score(gts, res)[0])
        out["cider_d"] = float(Cider().compute_score(gts, res)[0])
        if java:
            try:
                from pycocoevalcap.meteor.meteor import Meteor  # type: ignore

                out["meteor"] = float(Meteor().compute_score(gts, res)[0])
            except Exception as exc:  # noqa: BLE001
                raise CaptionMetricError(f"METEOR failed: {type(exc).__name__}: {exc}") from exc
            if spice:
                try:
                    from pycocoevalcap.spice.spice import Spice  # type: ignore

                    out["spice"] = float(Spice().compute_score(gts, res)[0])
                except Exception as exc:  # noqa: BLE001
                    raise CaptionMetricError(f"SPICE failed: {type(exc).__name__}: {exc}") from exc
    for k, v in out.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise CaptionMetricError(f"official metric {k} is not finite")
    return out


def caption_metrics(candidates: Sequence[str], references: Sequence[Sequence[str]], java: bool = True, spice: bool = False, official: bool = True, tolerance: float = 0.05) -> dict[str, float]:
    """Caption metrics with the official implementation as the authority.

    * native CIDEr-D / BLEU / ROUGE-L are always computed (keys ``native_*``);
    * when ``official`` (default), ``pycocoevalcap`` values are computed and stored under the canonical
      keys (``cider_d``, ``bleu1..4``, ``rouge_l``; ``meteor``/``spice`` when ``java``); the absolute
      native-vs-official discrepancy is recorded (``discrepancy_*``) and ``metrics_source`` says which
      implementation produced the canonical values;
    * a required metric that cannot be computed raises :class:`CaptionMetricError` (jobs fail instead
      of reporting NaN); with ``official=False`` and ``java=False`` the native values are canonical.
    """
    native = {"cider_d": cider_d(candidates, references), "rouge_l": rouge_l(candidates, references)}
    native.update(bleu(candidates, references))
    out: dict[str, float] = {f"native_{k}": v for k, v in native.items()}
    if official or java:
        off = official_metrics(candidates, references, java=java, spice=spice)
        out["metrics_source"] = "pycocoevalcap"
        out["tokenizer"] = off.pop("tokenizer")
        out.update(off)
        for k in native:
            if k in off:
                out[f"discrepancy_{k}"] = abs(float(off[k]) - float(native[k]))
        out["native_within_tolerance"] = all(out.get(f"discrepancy_{k}", 0.0) <= tolerance for k in ("bleu4", "rouge_l")) and out.get("discrepancy_cider_d", 0.0) <= 10 * tolerance
    else:
        out["metrics_source"] = "native"
        out.update(native)
    if "spice" in out and "cider_d" in out:
        out["spider"] = 0.5 * (out["cider_d"] + out["spice"])
    for k, v in out.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise CaptionMetricError(f"caption metric {k} is not finite")
    return out
