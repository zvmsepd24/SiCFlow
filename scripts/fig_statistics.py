"""Figure 7: FID, fiber and pore statistics and intrinsic dimension of models trained on the whole training band.

a: Fréchet Inception distance to the held-out band, 1000 windows of 128 px per ensemble (two-scale: mean of
   three seeds, bar = range). 512 px windows are not used: the held-out band holds only four non-overlapping ones.
b, c: distributions of fiber diameter and of the distance between neighboring fibers (Delaunay edges),
   from 256 px windows.
d: pore fraction of 256 px windows (box: quartiles; whiskers: 5th and 95th percentiles).
e: standard deviation of the pore fraction between non-overlapping W x W windows, against W; the held-out
   band is shown up to the largest W with at least 16 windows.
Data: results/sic/bandmodels/*.json (scripts/evaluate_band_models.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import BLUES, DOUBLE, INK, INK2, SERIES, TWO_SCALE, panel_label, plain_log_axis, plt  # noqa: E402

BM = ROOT / "results/sic/bandmodels"
UM = 1.04
MODELS = [("single scale, 64 px", ["single64"], BLUES[1]), ("single scale, 128 px", ["single128"], BLUES[2]),
          ("single scale, 256 px", ["single256"], BLUES[3]), ("single scale, 512 px", ["single512"], BLUES[4]),
          ("two-scale", TWO_SCALE, SERIES[2])]
SHORT = {"single scale, 64 px": "single,\n64 px", "single scale, 128 px": "single,\n128 px",
         "single scale, 256 px": "single,\n256 px", "single scale, 512 px": "single,\n512 px", "two-scale": "two-\nscale"}


def load(n):
    p = BM / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def runs(names):
    return [r for r in (load(n) for n in names) if r is not None]


def main():
    held, train = load("held_out_band"), load("training_band")
    out = {}
    fig = plt.figure(figsize=(DOUBLE, 0.62 * DOUBLE))
    boxes = [[0.07, 0.6, 0.25, 0.3], [0.41, 0.6, 0.25, 0.3], [0.74, 0.6, 0.25, 0.3],
             [0.07, 0.1, 0.4, 0.33], [0.58, 0.1, 0.41, 0.33]]
    ax_fid, ax_d, ax_e, ax_box, ax_sd = (fig.add_axes(b) for b in boxes)
    present = [(lab, runs(names), col) for lab, names, col in MODELS]
    present = [p for p in present if p[1]]
    # a: FID
    labels = ["training\nband"] + [SHORT[lab] for lab, _, _ in present]
    colors = [INK] + [col for _, _, col in present]
    x = np.arange(len(labels))
    runs_fid = [[train["scores"]["inception_128"]]] + [[r["scores"]["inception_128"] for r in rs] for _, rs, _ in present]
    vals = [np.mean(v) for v in runs_fid]
    ax_fid.bar(x, vals, 0.6, color=colors, lw=0)
    for xi, m, v in zip(x, vals, runs_fid):
        if len(v) > 1:  # range over training seeds
            ax_fid.errorbar(xi, m, yerr=[[m - min(v)], [max(v) - m]], color=INK2, lw=0.8, capsize=2)
    out["fid_128"] = dict(zip([lab.replace("\n", " ") for lab in labels], map(float, vals)))
    ax_fid.set_xticks(x, labels, fontsize=5.8)
    ax_fid.tick_params(axis="x", length=0)
    ax_fid.set_ylabel("FID to held-out band\n(128 px windows)")
    panel_label(ax_fid, "a", x=-0.3)
    # b, c: fiber statistics
    for ax, key, xlabel, bins in ((ax_d, "diameter", "fiber diameter (µm)", np.linspace(8, 22, 29)),
                                  (ax_e, "edge_length", "distance to neighboring fiber (µm)", np.linspace(8, 40, 33))):
        ax.hist(np.array(held["fibers"][key]) * UM, bins=bins, density=True, color="#d9d8d4", lw=0, label="held-out band")
        out[key] = {"held-out band": float(np.median(held["fibers"][key]) * UM)}
        for lab, rs, col in present:
            v = np.concatenate([np.array(r["fibers"][key]) for r in rs]) * UM
            h, e = np.histogram(v, bins=bins, density=True)
            ax.plot((e[1:] + e[:-1]) / 2, h, color=col, lw=1.3, label=lab)
            out[key][lab] = float(np.median(v))
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    ax_d.set_ylabel("frequency")
    h, lab = ax_d.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.7, 1.0), ncol=len(lab), fontsize=6, handlelength=1.4,
               columnspacing=1.2)
    panel_label(ax_d, "b", x=-0.12)
    panel_label(ax_e, "c", x=-0.12)
    # d: pore fraction of 256 px windows
    data = [(np.array(train["pores"]["window256"]), "training\nband", INK), (np.array(held["pores"]["window256"]), "held-out\nband", INK)]
    for lab, rs, col in present:
        data.append((np.concatenate([np.array(r["pores"]["window256"]) for r in rs]), SHORT[lab], col))
    bp = ax_box.boxplot([d for d, _, _ in data], widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True,
                        medianprops=dict(color="white", lw=1.2), whiskerprops=dict(color=INK2, lw=0.8),
                        capprops=dict(color=INK2, lw=0.8))
    for patch, (_, _, col) in zip(bp["boxes"], data):
        patch.set_facecolor(col)
        patch.set_edgecolor(col)
    ax_box.set_xticks(range(1, len(data) + 1), [lab for _, lab, _ in data], fontsize=5.8)
    ax_box.tick_params(axis="x", length=0)
    ax_box.set_ylabel("pore fraction of\n0.27 mm windows")
    out["pore_fraction_mean"] = {lab.replace("\n", " "): float(d.mean()) for d, lab, _ in data}
    panel_label(ax_box, "d", x=-0.18)
    # e: spread of pore fraction between windows
    for res, lab, ls in ((train, "real, training band", "-"), (held, "real, held-out band", ":")):
        sd = res["pores"]["sd_between_windows"]
        w = np.array(sorted(int(k) for k in sd))
        ax_sd.plot(w * UM, [sd[str(k)] for k in w], ls, color=INK, lw=1.4, marker="o", ms=2.5, label=lab)
    for lab, rs, col in present:
        w = np.array(sorted(int(k) for k in rs[0]["pores"]["sd_between_windows"]))
        v = np.array([[r["pores"]["sd_between_windows"][str(k)] for k in w] for r in rs])
        if len(rs) > 1:
            ax_sd.fill_between(w * UM, v.min(0), v.max(0), color=col, alpha=0.25, lw=0)
        ax_sd.plot(w * UM, v.mean(0), color=col, lw=1.3, marker="o", ms=2.5, label=lab)
    ax_sd.set_xscale("log")
    ax_sd.set_yscale("log")
    plain_log_axis(ax_sd, "x", [10, 30, 100, 300])
    plain_log_axis(ax_sd, "y", [0.005, 0.01, 0.02, 0.05, 0.1])
    ax_sd.set_xlabel("window size (µm)")
    ax_sd.set_ylabel("spread of pore fraction\nbetween windows (s.d.)")
    ax_sd.legend(fontsize=5.3, loc="lower left")
    panel_label(ax_sd, "e", x=-0.18)
    fig.savefig(ROOT / "figures/fig7_statistics.pdf")
    fig.savefig(ROOT / "figures/fig7_statistics.png", dpi=220)
    (ROOT / "results/sic/fig7_statistics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
