"""Supplementary tables, written as LaTeX from the result files.

S1: image-level scores of models trained on the whole training band (FID, KID, Fréchet distance with DINOv2
    features), from results/sic/bandmodels.
S2: fiber and pore statistics of the same models.
S3: Levina-Bickel intrinsic dimension of the checkerboard ensembles (results/sic/convergence), k = 10.
S4: Levina-Bickel intrinsic dimension of ensembles of 128 px windows of the whole-band models, k = 10 and 20.
S5: fiber models of the two-scale model given real layouts: copying of the training band, rims and fiber statistics,
    trained on exact layouts and with noise on the layout (results/sic/towscale/given_layout.json).
S6: tow statistics and copying at the tow scale of every model (results/sic/towscale).
Writes tables/supplement_tables.tex.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BM = ROOT / "results/sic/bandmodels"
CONV = ROOT / "results/sic/convergence"
UM = 1.04
TS = ROOT / "results/sic/towscale"
MODELS = [("Training band (real)", ["training_band"]), ("Single scale, 64-pixel crops", ["single64"]),
          ("Single scale, 128-pixel crops", ["single128"]), ("Single scale, 256-pixel crops", ["single256"]),
          ("Single scale, 512-pixel crops", ["single512"]),
          ("Two-scale", ["twoscale_early_2k_s1", "twoscale_early_2k_s2", "twoscale_early_2k_s3"])]


def load(n):
    p = BM / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(vals, digits=0):
    m = np.mean(vals)
    if len(vals) == 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ({min(vals):.{digits}f}--{max(vals):.{digits}f})"


def table_scores():
    keys = ["inception_128", "kid_128", "dinov2_128"]  # 512 px windows: the held-out band holds only four
    rows = []
    for label, names in MODELS:
        rs = [r for r in map(load, names) if r is not None and "scores" in r]
        if rs:
            rows.append(label + " & " + " & ".join(fmt([r["scores"][k] for r in rs]) for k in keys) + r"\\")
    return "\n".join([
        r"\begin{table}[h]", r"\centering\small",
        r"\caption{\textbf{Image-level scores against the held-out band.} Fréchet Inception distance (FID), kernel inception distance (KID, $\times10^3$) and Fréchet distance with DINOv2 ViT-B/14 features, on 1000 windows of $128\times128$ pixels per ensemble. For three seeds, the mean and the range are given.}",
        r"\label{tab:scores}", r"\begin{tabular}{lrrr}", r"\toprule",
        r" & FID & KID & DINOv2\\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def table_phases():
    rows = []
    held = load("held_out_band")
    for label, names in [("Held-out band (real)", ["held_out_band"])] + MODELS:
        rs = [r for r in map(load, names) if r is not None]
        if not rs:
            continue
        ff = [r["curves"]["fiber_fraction"] for r in rs]
        pf = [r["pores"]["fraction"] for r in rs]
        dm = [np.median(r["fibers"]["diameter"]) * UM for r in rs]
        ed = [np.median(r["fibers"]["edge_length"]) * UM for r in rs]
        w1 = [r["fiber_w1_um"]["diameter"] for r in rs if "fiber_w1_um" in r]
        w1s = fmt(w1, 2) if w1 else "--"
        rows.append(f"{label} & {fmt(ff, 3)} & {fmt(pf, 3)} & {fmt(dm, 1)} & {fmt(ed, 1)} & {w1s}\\\\")
    return "\n".join([
        r"\begin{table}[h]", r"\centering\small",
        r"\caption{\textbf{Phase fractions and fiber statistics.} Fiber and pore fractions of the segmented images, median fiber diameter and median distance between neighboring fibers ($\mu$m), and the Wasserstein-1 distance between the fiber-diameter distributions of the images and of the held-out band ($\mu$m).}",
        r"\label{tab:phases}", r"\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrr}", r"\toprule",
        r" & fiber fraction & pore fraction & diameter & neighbor distance & diameter, W$_1$\\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


def table_lb_checkerboard():
    reps = [json.loads(p.read_text()) for p in sorted(CONV.glob("rep*.json"))]
    if not reps:
        return ""
    n = min(len(r) for r in reps)
    rows = []
    for lev in range(n):
        area = reps[0][lev]["area_mm2"]
        cells = []
        for kind in ("generated", "training"):
            for key in ("pixels", "two_point", "chords"):
                v = [np.mean([r[lev]["dimension"][kind][s][key] for s in "AB"]) for r in reps]
                cells.append(f"{np.mean(v):.1f}")
        rows.append(f"{area:.2g} mm$^2$ & " + " & ".join(cells) + r"\\")
    h = [np.mean([r[0]["dimension"]["held_out"][k] for r in reps]) for k in ("pixels", "two_point", "chords")]
    rows.append(r"Held-out windows & " + " & ".join(f"{v:.1f}" for v in h) + r" & & & \\")
    return "\n".join([
        r"\begin{table}[h]", r"\centering\small",
        r"\caption{\textbf{Intrinsic dimension of the checkerboard ensembles.} Levina--Bickel estimate ($k=10$) for clouds of 512 windows of $64\times64$ pixels, described by their pixel values, their two-point vectors and their chord-length distributions; generated windows (mean of models A and B) and windows of the training tiles; means over three replicates.}",
        r"\label{tab:lbcheck}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r" & \multicolumn{3}{c}{generated} & \multicolumn{3}{c}{training tiles}\\",
        r"Training area & pixels & two-point & chords & pixels & two-point & chords\\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def table_lb_band():
    rows = []
    cell = lambda v: f"{v['mean']:.1f} $\\pm$ {v['sd']:.1f}"
    for label, names in [("Held-out band (real)", ["held_out_band"])] + MODELS:
        rs = [r for r in map(load, names) if r is not None]
        if not rs:
            continue
        r = rs[0]
        rows.append(label + (" (seed 1)" if len(rs) > 1 else "") + " & " +
                    " & ".join(cell(r["dimension"][f"{k}_k{kk}"]) for k in ("pixels", "two_point", "chords") for kk in (10, 20)) + r"\\")
    return "\n".join([
        r"\begin{table}[h]", r"\centering\small",
        r"\caption{\textbf{Intrinsic dimension of ensembles of $128\times128$-pixel windows.} Levina--Bickel estimate for clouds of 100 non-overlapping windows, described by their pixel values, their two-point vectors and their chord-length distributions, for $k=10$ and 20 nearest neighbors; mean and standard deviation over 20 random subsets. The held-out band contains only 108 such windows.}",
        r"\label{tab:lbband}", r"\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrr}", r"\toprule",
        r" & \multicolumn{2}{c}{pixels} & \multicolumn{2}{c}{two-point} & \multicolumn{2}{c}{chords}\\",
        r" & $k=10$ & $k=20$ & $k=10$ & $k=20$ & $k=10$ & $k=20$\\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


def table_given_layout():
    p = TS / "given_layout.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    names = [("Held-out band (real)", "real"), ("Earlier layout, exact layouts in training", "fiber_s1"),
             ("Exact layouts in training", "fiber_state_s1"), ("Noise on the layout, sampled at 0.25", "fiber_stateaug_s1@0.25"),
             ("Noise on the layout, sampled at 0.5", "fiber_stateaug_s1@0.5"), ("Noise on the layout, sampled at 1", "fiber_stateaug_s1@1.0"), ("Blurred noise, sampled at 0.25", "fiber_stateblur_s1@0.25"), ("Noise on the layout, 4000 steps, sampled at 0.5", "fiber_stateaugshort_s1@0.5"), ("Noise on the layout, 2000 steps, sampled at 0.5", "fiber_stateaug2k_s1@0.5")]
    rows = []
    for label, k in names:
        if k not in d:
            continue
        r = d[k]
        c = lambda key: f"{np.mean(r[key]):.2f}" if key in r else ""
        rows.append(f"{label} & {c('copy_training_high')} & {c('corr_real_high')} & {c('corr_real_blur')} & "
                    f"{100 * r['dark_fraction']:.1f} & {r['fiber_fraction']:.3f} & {r['diameter_um']:.1f}\\\\")
    return "\n".join([
        r"\begin{table}[h]", r"\centering\small",
        r"\caption{\textbf{Fiber models given real layouts.} Copying: correlation of the image generated from the layout of a region of the training band (1024 pixels) with that region, after removing the layout scale (Gaussian of 12 pixels). Held-out layout: correlation of images generated from the layout of the held-out band with the held-out image, after removing the layout scale (fiber scale) and after the Gaussian (layout scale); rim pixels (gray levels 45--89), fiber fraction and median fiber diameter ($\mu$m) of the generated image. The earlier layout (gray-level thresholds, Gaussian of 6 pixels) keeps the positions of single fibers.}",
        r"\label{tab:given}", r"\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrr}", r"\toprule",
        r" & copying & \multicolumn{2}{c}{held-out layout} & & & \\",
        r"Fiber model & training region & fiber scale & layout scale & rim pixels (\%) & fiber fraction & diameter\\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


def table_tow():
    p = TS / "real.json"
    if not p.exists():
        return ""
    real = json.loads(p.read_text())
    lab = {"single64": "Single scale, 64-pixel crops", "single128": "Single scale, 128-pixel crops",
           "single256": "Single scale, 256-pixel crops", "single512": "Single scale, 512-pixel crops",
           "twoscale_early_2k_s1": "Two-scale (seed 1)", "twoscale_earlyshort_s1": "Two-scale, fiber model after 4000 steps", "twoscale_early_s1": "Two-scale, fiber model after 8000 steps", "twoscale_stateaug_s1": "Two-scale, layout model after 6000 steps", "twoscale_layout_state_s1": "Layout model, band, 0.53 mm (seed 1)",
           "layouts_full_s2": "Layout model, band, 0.53 mm (seed 2)", "layouts_A_s1": "Layout model, left half, 0.53 mm",
           "layouts_A_s2": "Layout model, left half, 0.53 mm (seed 2)", "layouts_B_s1": "Layout model, right half, 0.53 mm",
           "layouts_quarter_s1": "Layout model, quarter, 0.53 mm", "layouts_crop32_s1": "Layout model, band, 0.27 mm",
           "layouts_crop32_A_s1": "Layout model, left half, 0.27 mm", "layouts_crop32_B_s1": "Layout model, right half, 0.27 mm",
           "layouts_crop16_s1": "Layout model, band, 0.13 mm"}
    rows = [f"Training band (real) & {real['training band']['corr_along'][32]:.2f} & {real['training band']['corr_along'][64]:.2f} & "
            f"{min(real['training band']['corr_across']):.2f} & & \\\\",
            f"Held-out band (real) & {real['held-out band']['corr_along'][32]:.2f} & {real['held-out band']['corr_along'][64]:.2f} & "
            f"{min(real['held-out band']['corr_across']):.2f} & {np.median(real['copy_held']['0:2640 64']):.2f} & \\\\"]
    for k, name in lab.items():
        q = TS / f"{k}.json"
        if not q.exists():
            continue
        d = json.loads(q.read_text())
        t = d["tow"]
        g = np.array(d["copy"]["64"])
        h = np.array(real["copy_held"][f"{d['cols'][0]}:{d['cols'][1]} 64"])
        rows.append(f"{name} & {t['corr_along'][32]:.2f} & {t['corr_along'][64]:.2f} & {min(t['corr_across']):.2f} & "
                    f"{np.median(g):.2f} ({np.median(h):.2f}) & {np.mean(g > h.max()):.2f}\\\\")
    return "\n".join([
        r"\begin{table}[h]", r"\centering\small",
        r"\caption{\textbf{Tow statistics and copying at the tow scale.} Fiber-fraction layout (segmented fibers, Gaussian of 12 pixels, 8.3\,$\mu$m per pixel) of 24 fields or layouts per model. Correlation along the bands at 0.27 and 0.53\,mm and smallest correlation across them. Copying: median NCC of 128 windows of 0.53\,mm with the most similar window of the model's training region (held-out windows against the same region in parentheses), and the fraction of generated windows more similar than any held-out window.}",
        r"\label{tab:tow}", r"\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrr}", r"\toprule",
        r" & \multicolumn{2}{c}{along} & across & \multicolumn{2}{c}{copying} \\",
        r"Model & 0.27\,mm & 0.53\,mm & minimum & median NCC & fraction above held-out\\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


def main():
    parts = [table_scores(), table_phases(), table_lb_checkerboard(), table_lb_band(), table_given_layout(), table_tow()]
    out = "% Generated by scripts/supplement_tables.py; do not edit by hand.\n\n" + "\n\n".join(p for p in parts if p) + "\n"
    (ROOT / "tables/supplement_tables.tex").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    sys.exit(main())
