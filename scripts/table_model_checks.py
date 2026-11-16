"""Table of the checks of the 64-pixel model of the whole training band (LaTeX).

Rows: the 64-pixel model (two training seeds), the same network with 1.5 times the channels, the same network trained
for 1.5 times the steps, and the 64-pixel model sampled with uniform time steps and with more steps. Columns:
parameters, training time with nothing else on the GPU, held-out flow-matching loss at the end of training, and
distances to 1024 held-out 64 x 64 px windows (mean over training seeds and noise sets where there are several).
Reads results/sic/single/*.json and .pt, results/sic/windows/eval_*.json and references.json.
Writes tables/table_model_checks.tex.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
S, W = ROOT / "results/sic/single", ROOT / "results/sic/windows"


def ev(tag):
    f = W / f"eval_{tag}.json"
    return json.loads(f.read_text()) if f.exists() else None


def mean_fid(tags):
    es = [e for e in (ev(t) for t in tags) if e]
    return {k: np.mean([e["fidelity"][k] for e in es]) for k in ("fid", "two_point", "chords", "pore")}, len(es)


def train_info(name, timed=None, steps=None):
    """Parameters, training minutes and final held-out loss. Minutes come from the step time of `timed` (a run made
    with nothing else on the GPU, from its second log entry on: the first includes compilation) times `steps`."""
    ck = torch.load(S / f"{name}.pt", map_location="cpu", weights_only=False)
    params = sum(v.numel() for v in ck["ema"].values()) / 1e6
    loss = np.mean(json.loads((S / f"{name}.json").read_text())["log"][-1]["held_out"])
    log = json.loads((S / f"{timed or name}.json").read_text())["log"]
    per_step = (log[-1]["seconds"] - log[0]["seconds"]) / (log[-1]["step"] - log[0]["step"])
    return params, per_step * (steps or ck["steps"]) / 60, loss


# (label, model for parameters and loss, run timed alone, steps, models evaluated); seed 1 of the 64-pixel model
# shared the GPU, so its time comes from the same network trained alone for 12000 steps, scaled to 8000 steps
rows = [
    ("64-pixel model (2 seeds)", "single64_seed1", "single64_steps12000", 8000, ["single64", "single64_seed1"]),
    ("1.5$\\times$ channels", "single64_wide", None, None, ["single64_wide"]),
    ("1.5$\\times$ training steps", "single64_steps12000", None, None, ["single64_steps12000"]),
]
lines = [r"\begin{tabular}{@{}lrrrlrrrrr@{}}", r"\toprule",
         r"Model & Param.\ (M) & Train (min) & Held-out loss & Sampler & Eval. & FID & Two-point ($10^{-3}$) & Chords ($10^{-3}$) & Pore ($10^{-2}$) \\",
         r"\midrule"]
for label, name, timed, steps, evals in rows:
    sampler = "32 steps, uniform"
    params, minutes, loss = train_info(name, timed, steps)
    tags = [n + s for n in evals for s in ("", "@32n1", "@32n2")]
    d, n = mean_fid(tags)
    lines.append(f"{label} & {params:.1f} & {minutes:.0f} & {loss:.4f} & {sampler} & {n} & {d['fid']:.0f} & "
                 f"{d['two_point'] * 1e3:.1f} & {d['chords'] * 1e3:.1f} & {d['pore'] * 1e2:.1f} \\\\")
tags = [n + s for n in ("A4scratch", "B4scratch") for s in ("", "@32n1", "@32n2")]
d, n = mean_fid(tags)
if n:
    lines.append(f"Paired models A, B at 3.2\,mm$^2$, from scratch & 13.8 & 13 & -- & 32 steps, uniform & {n} & {d['fid']:.0f} & "
                 f"{d['two_point'] * 1e3:.1f} & {d['chords'] * 1e3:.1f} & {d['pore'] * 1e2:.1f} \\\\")
# other samplers: one evaluation each, so only the FID is given (descriptor distances of one evaluation vary too much)
for tag, sampler in (("single64@8", "8 steps, uniform"), ("single64@16", "16 steps, uniform"),
                     ("single64@64", "64 steps, uniform"), ("single64@128", "128 steps, uniform"),
                     ("single64@16r0.7", "16 steps, bunched near the noise"),
                     ("single64@32r0.7", "32 steps, bunched near the noise"),
                     ("single64@64r0.7", "64 steps, bunched near the noise")):
    d, n = mean_fid([tag])
    if n:
        lines.append(f"64-pixel model (seed 64) & & & & {sampler} & {n} & {d['fid']:.0f} & -- & -- & -- \\\\")
ref = json.loads((W / "references.json").read_text())
t, f = ref["training_band"], ref["floor"]
lines += [r"\midrule",
          f"Real windows of the training band & & & & & & {t['fid']:.0f} & {t['two_point'] * 1e3:.1f} & {t['chords'] * 1e3:.1f} & {t['pore'] * 1e2:.1f} \\\\",
          f"Two sets of held-out windows & & & & & & {f['fid']:.0f} & {f['two_point'] * 1e3:.1f} & {f['chords'] * 1e3:.1f} & {f['pore'] * 1e2:.1f} \\\\",
          r"\bottomrule", r"\end{tabular}"]
out = ROOT / "tables/table_model_checks.tex"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
