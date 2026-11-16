"""Numbers quoted in the tow-scale part of the paper, from the saved results.

Copying: median NCC of 0.53 mm windows (64 layout px) of the fiber-fraction layout with the most similar window of
the training region, generated against held-out windows, and the fraction of generated windows above the largest
held-out value. Tow structure: correlation of the fiber-fraction layout along the bands at 0.27 and 0.53 mm.
Usage: python scripts/key_numbers_tow.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TS = ROOT / "results/sic/towscale"
PX = 8 * 1.04


def load(name):
    p = TS / f"{name.replace('/', '_')}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    real = json.loads((TS / "real.json").read_text())
    print("== tow structure (fiber-fraction layout): correlation along at 0.27 / 0.53 mm, across min; sd")
    for k in ("training band", "held-out band", "left half", "right half"):
        t = real[k]
        print(f"  real {k:14s} along {t['corr_along'][32]:.3f} / {t['corr_along'][64]:.3f}  across min "
              f"{min(t['corr_across']):.3f} at {np.argmin(t['corr_across']) * PX:.0f} um  sd {t['sd']:.3f}")
    names = ["single64", "single128", "single256", "single512", "single512_steps24000", "twoscale_s1", "twoscale_state_s1",
             "twoscale_earlyshort_s1", "twoscale_earlyshort_s2", "twoscale_earlyshort_s3",
             "twoscale_early_2k_s1", "twoscale_early_2k_s2", "twoscale_early_2k_s3", "twoscale_early_6k_s1", "twoscale_early_long_s1",
             "twoscale_early_s1", "twoscale_early_s2", "twoscale_early_s3",
             "twoscale_stateaug_s1", "twoscale_stateaug_s2", "twoscale_stateaug_s3", "twoscale/layout_state_s1",
             "layouts/steps500_s1", "layouts/steps500_s2", "layouts/steps1000_s1", "layouts/steps1000_s2", "layouts/steps2000_s1",
             "layouts/A500_s1", "layouts/A500_s2", "layouts/B500_s1", "layouts/A1000_s1", "layouts/A1000_s2", "layouts/B1000_s1",
             "layouts/crop32_A_s1", "layouts/crop32_A_s2", "layouts/crop32_B_s1",
             "layouts/full_s2", "layouts/A_s1", "layouts/A_s2", "layouts/B_s1", "layouts/quarter_s1",
             "layouts/crop32_s1", "layouts/crop16_s1"]
    print("== per model: tow structure and copying")
    for n in names:
        d = load(n)
        if d is None:
            continue
        t = d["tow"]
        c0, c1 = d["cols"]
        line = (f"  {n:26s} along {t['corr_along'][32]:.3f} / {t['corr_along'][64]:.3f}  across min "
                f"{min(t['corr_across']):.3f}  sd {t['sd']:.3f}  mean {t['mean']:.3f}")
        for K, g in d["copy"].items():
            h = np.array(real["copy_held"][f"{c0}:{c1} {K}"])
            g = np.array(g)
            line += f" | copy{K}: gen {np.median(g):.2f} held {np.median(h):.2f} above-max {np.mean(g > h.max()):.2f}"
        for K, g in d.get("copy_other", {}).items():
            line += f" | other{K}: {np.median(g):.2f}"
        for K, g in d.get("copy_image", {}).items():
            h = np.array(real["copy_image_held"][str(K)])
            g = np.array(g)
            line += f" | image{K}: gen {np.median(g):.2f} (max {g.max():.2f}) held {np.median(h):.2f} (max {h.max():.2f})"
        print(line)
    p = TS / "agreement.json"
    if p.exists():
        print("== same-noise agreement of layouts")
        for k, v in json.loads(p.read_text()).items():
            print(f"  {k}: {v:.3f}")
    for steps, f in ((2000, "fiber_pairs_stateaug2k.json"), (4000, "fiber_pairs_stateaugshort.json"), (8000, "fiber_pairs.json"),
                     (16000, "fiber_pairs_stateauglong.json")):
        p = TS / f
        if p.exists():
            print(f"== fiber models of the two halves, same layout and noise, {steps} steps")
            d = json.loads(p.read_text())
            print("  agreement (layout scale removed):", {k: round(v, 3) for k, v in d["agreement_high"].items()})
            print("  copying of each half:", {k: {r: round(v, 3) for r, v in c.items()} for k, c in d["copying"].items()})
    p = TS / "given_layout.json"
    if p.exists():
        print("== fiber models given real layouts")
        for k, v in json.loads(p.read_text()).items():
            keys = ("copy_training_high", "corr_real_high", "corr_real_blur", "corr_pair_blur", "dark_fraction",
                    "fiber_fraction", "diameter_um")
            print(f"  {k:24s} " + "  ".join(f"{q} {np.mean(v[q]):.3f}" for q in keys if q in v))
    p = TS / "dimension.json"
    if p.exists():  # scripts/evaluate_tow_dimension.py
        d = json.loads(p.read_text())
        print("== intrinsic dimension of 0.53 mm layout windows (two-point autocovariance; pixels)")
        for k, v in d.get("real", {}).items():
            print(f"  real {k:14s} crops {v['crops']:5.1f}  two-point {v['two_point']['mean']:.2f}  pixels {v['pixels']['mean']:.1f}")
        for k, v in d.get("generated", {}).items():
            print(f"  {k:24s} two-point {v['two_point']['mean']:.2f} +- {v['two_point']['sd']:.2f}  pixels {v['pixels']['mean']:.1f}")
    bm = ROOT / "results/sic/bandmodels"
    print("== band models: FID 128 px, fiber fraction, pore fraction, diameter W1 (um)")
    for n in ("training_band", "single64", "single128", "single256", "single512", "twoscale_s1", "twoscale_state_s1",
              "twoscale_early_1k_s1", "twoscale_earlyshort_s1", "twoscale_earlyshort_s2", "twoscale_earlyshort_s3",
              "twoscale_early_2k_s1", "twoscale_early_2k_s2", "twoscale_early_2k_s3", "twoscale_early_6k_s1",
              "twoscale_early_s1", "twoscale_early_s2", "twoscale_early_s3", "twoscale_early_long_s1",
              "twoscale_early_crop256_s1", "twoscale_stateaug_s1", "twoscale_stateaug_s2",
              "twoscale_stateaug_s3", "twoscale_stateblur_s1", "twoscale_mix_l3", "twoscale_mix_l5", "twoscale_mix_l7",
              "fibers_on_real_layouts", "fibers_blur_on_real_layouts"):
        p = bm / f"{n}.json"
        if p.exists():
            d = json.loads(p.read_text())
            print(f"  {n:22s} FID {d['scores']['inception_128']:.1f}  fiber {d['curves']['fiber_fraction']:.3f}  "
                  f"pore {d['pores']['fraction']:.4f}  diameter W1 {d['fiber_w1_um']['diameter']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
