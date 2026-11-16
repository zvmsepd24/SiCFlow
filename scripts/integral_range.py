"""Integral range of the fiber and pore phases of the section: how much area represents the volume fractions.

For a stationary two-phase medium with volume fraction p, the variance of the fraction measured over a window of
area A falls as D^2(A) = p (1 - p) A3 / A once A is much larger than the integral range A3 (Lantuejoul 1991; Kanit et
al. 2003). A3 = D^2(A) A / (p (1 - p)) is estimated for square windows of 8 to 1024 px tiling the whole section; it
levels off at the integral range when the windows are large enough. The area needed to know p to a relative error
eps (two standard deviations) from one region is A_RVE = 4 p (1 - p) A3 / (eps p)^2.
Writes results/sic/integral_range.json and figures/figS_integral_range.pdf/.png.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro.phases import phases  # noqa: E402
from scripts.figstyle import INK, INK2, SERIES, SINGLE, panel_label, plain_log_axis, plt  # noqa: E402

UM = 1.04
WINDOWS = [8, 16, 32, 64, 128, 256, 512, 1024]


def main():
    plane = np.load(ROOT / "data/sic/section_plane.npy").astype(np.float32) / 127.5 - 1
    ph = phases(plane)
    H, W = ph.shape
    res = {}
    for name, ind in (("fiber", ph == 1), ("pore", ph == 2)):
        p = float(ind.mean())
        rows = []
        for w in WINDOWS:
            ny, nx = H // w, W // w
            m = ind[:ny * w, :nx * w].reshape(ny, w, nx, w).mean((1, 3))
            a = (w * UM) ** 2 / 1e6  # mm^2
            rows.append(dict(window_px=w, area_mm2=a, n=int(ny * nx), d2=float(m.var()),
                             a3_mm2=float(m.var() * a / (p * (1 - p)))))
        a3 = rows[-1]["a3_mm2"]
        res[name] = dict(fraction=p, windows=rows, a3_mm2=a3,
                         rve_mm2={str(e): 4 * (1 - p) * a3 / (e * e * p) for e in (0.05, 0.1, 0.2)})
        print(name, f"p {p:.4f}  A3 {a3 * 1e6:.0f} um^2 (sqrt {np.sqrt(a3) * 1e3:.0f} um)  RVE for 5/10/20%:",
              {k: round(v, 2) for k, v in res[name]["rve_mm2"].items()}, "mm^2")
    (ROOT / "results/sic/integral_range.json").write_text(json.dumps(res, indent=1))
    fig, ax = plt.subplots(figsize=(SINGLE, 0.7 * SINGLE))
    plt.subplots_adjust(left=0.2, right=0.97, bottom=0.2, top=0.95)
    for name, col, lab in (("fiber", SERIES[0], "fibers"), ("pore", SERIES[1], "pores")):
        r = res[name]["windows"]
        ax.plot([x["window_px"] * UM for x in r], [x["a3_mm2"] * 1e6 for x in r], "o-", color=col, ms=3, label=lab)
    for x, t in ((64 * UM, "64 px crop"), (512 * UM, "0.53 mm crop")):
        ax.axvline(x, color=INK2, lw=0.6, ls=":")
        ax.text(x * 1.06, 2e4, t, fontsize=5.6, color=INK2, rotation=90, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_axis(ax, "x", [10, 30, 100, 300, 1000])
    plain_log_axis(ax, "y", [10, 100, 1000, 10000])
    ax.set_xlabel("window side (µm)")
    ax.set_ylabel("integral range estimate (µm$^2$)")
    ax.legend(fontsize=6, loc="upper left")
    fig.savefig(ROOT / "figures/figS_integral_range.pdf", dpi=300)
    fig.savefig(ROOT / "figures/figS_integral_range.png", dpi=220)


if __name__ == "__main__":
    main()
