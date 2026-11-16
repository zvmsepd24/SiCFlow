"""Print the numbers quoted in the paper, from the result files.

Paired models (results/sic/convergence), same-noise agreement per sample (results/sic/windows/disagreement.json),
checks of the 64-pixel model (results/sic/windows) and whole-band models (results/sic/bandmodels). The tow-scale numbers
are printed by scripts/key_numbers_tow.py.
Usage: python scripts/key_numbers.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/sic"
KEYS = ("fid", "two_point", "chords", "pore", "diameter")


def paired():
    reps = [json.loads((R / f"convergence/rep{r}.json").read_text()) for r in range(3) if (R / f"convergence/rep{r}.json").exists()]
    print(f"== paired models ({len(reps)} replicates)")
    for lev in range(5):
        rows = [rep[lev] for rep in reps if len(rep) > lev]
        if not rows:
            continue
        m = lambda f: np.mean([f(x) for x in rows])
        print(f"area {rows[0]['area_mm2']:.3f}: agreement {m(lambda x: x['agreement']):.3f}  copy excess "
              f"{m(lambda x: x['copy']['A']['generated'] - x['copy']['A']['held_out']):.3f}")
        for k in KEYS:
            print(f"   {k:9s} fidelity {m(lambda x: (x['fidelity']['A'][k] + x['fidelity']['B'][k]) / 2):.4g}"
                  f"  A-B {m(lambda x: x['convergence'][k]):.4g}  tiles-held {m(lambda x: x['real_fidelity'][k]):.4g}"
                  f"  tilesA-B {m(lambda x: x['real_convergence'][k]):.4g}  floor {rows[0]['floor'][k]:.4g}")
        d = {c: m(lambda x: (x['dimension']['generated']['A'][c] + x['dimension']['generated']['B'][c]) / 2)
             for c in ("pixels", "two_point", "chords")}
        print("   dimension generated", {k: round(v, 1) for k, v in d.items()}, "held",
              {k: round(rows[0]['dimension']['held_out'][k], 1) for k in d})


def disagreement():
    f = R / "windows/disagreement.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    print("== same-noise agreement per sample")
    for p, v in d.items():
        r = np.array(v["agreement"])
        line = f"{p:28s} mean {r.mean():.3f} median {np.median(r):.3f} below0.5 {(r < 0.5).mean():.3f} worst {np.sort(r)[:3].round(2)}"
        if "pore1" in v:
            a, b = np.array(v["pore1"]), np.array(v["pore2"])
            big = np.abs(a - b) > 0.2
            line += (f" | pore>0.2 windows A {(a > 0.2).mean():.3f} B {(b > 0.2).mean():.3f}; differ {big.mean():.3f};"
                     f" share of r<0.5 {big[r < 0.5].mean() if (r < 0.5).any() else 0:.2f}; median r of those {np.median(r[big]) if big.any() else 0:.2f};"
                     f" B has the pore {(b[big] > a[big]).mean() if big.any() else 0:.2f}")
        print(line)


def windows():
    W = R / "windows"
    ev = lambda t: json.loads((W / f"eval_{t}.json").read_text())["fidelity"] if (W / f"eval_{t}.json").exists() else None
    print("== 64-pixel model and checks (32 uniform steps; noise sets 0-2)")
    for name, models in (("64-pixel", ["single64", "single64_seed1"]), ("wide", ["single64_wide"]), ("steps12000", ["single64_steps12000"])):
        vals = [e for e in (ev(m + s) for m in models for s in ("", "@32n1", "@32n2")) if e]
        if vals:
            print(f"{name:11s} n={len(vals)} " + " ".join(f"{k} {np.mean([v[k] for v in vals]):.4g} ({min(v[k] for v in vals):.3g}-{max(v[k] for v in vals):.3g})" for k in ("fid", "two_point", "chords", "pore")))
    print("sweep uniform:", {s: round(ev('single64' if s == 32 else f'single64@{s}')['fid'], 1) for s in (8, 16, 32, 64, 128) if ev('single64' if s == 32 else f'single64@{s}')})
    print("sweep near-noise grid:", {s: round(ev(f'single64@{s}r0.7')['fid'], 1) for s in (8, 16, 32, 64, 128) if ev(f'single64@{s}r0.7')})
    ref = json.loads((W / "references.json").read_text())
    print("references", {k: {m: round(v, 4) for m, v in ref[k].items()} for k in ("training_band", "floor")})


def band():
    B = R / "bandmodels"
    print("== whole-band models")
    for n in ("held_out_band", "training_band", "single64", "single128", "single256", "single512", "twoscale_stateaug_s1",
              "twoscale_stateaug_s2", "twoscale_stateaug_s3", "fibers_on_real_layouts"):
        f = B / f"{n}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        sc, po = d.get("scores", {}), d.get("pores", {})
        print(f"{n:14s} FID128 {sc.get('inception_128', float('nan')):.1f} KID128 {sc.get('kid_128', float('nan')):.4f}"
              f" DINOv2 {sc.get('dinov2_128', float('nan')):.0f} pore fraction {po.get('fraction', float('nan')):.4f}"
              f" fiber fraction {np.mean(d['curves']['fiber_fraction']) if 'curves' in d else float('nan'):.3f}")


if __name__ == "__main__":
    for part in (sys.argv[1:] or ["paired", "disagreement", "windows", "band"]):
        globals()[part]()
