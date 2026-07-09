"""Live DDPO training diagnostic dashboard. Reads monitoring/ddpo_ckpts/metrics.jsonl and writes a
multi-panel PNG. Re-run any time to refresh as training progresses.

    python -m src.ddpo_ft.plot_metrics [metrics.jsonl] [out.png]

Per-iter values are noisy (fresh random inputs each outer iter) so every panel overlays a moving
average (bold) on the raw series (faint). Diagnosis cues are printed and annotated:
  * spec_highk DOWN  = the hi-k deficit shrinking (THE objective)
  * pde flat/down while spec_highk falls = genuine (physical) improvement
  * pde UP while spec_highk falls = wrong-phase high-k energy (spectral reward being gamed)
  * reward UP, reward_std healthy (advantage signal), loss meaningfully != 0 (updates biting)
"""
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

W = 10  # moving-average window


def mov(a, w=W):
    if len(a) < 2:
        return a
    w = min(w, len(a))
    k = np.ones(w) / w
    return np.convolve(a, k, mode="valid")


def slope(a):
    return float(np.polyfit(np.arange(len(a)), a, 1)[0]) if len(a) > 1 else 0.0


def main(path="monitoring/ddpo_ckpts/metrics.jsonl", out="monitoring/ddpo_ckpts/dashboard.png"):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    it = np.array([r["iter"] for r in rows])
    col = lambda k: np.array([r.get(k, np.nan) for r in rows])
    mit = it[W - 1:] if n >= W else it

    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 4, figsize=(21, 8))

    def panel(a, keys, title, colors, ylog=False, invert_good=None):
        for k, c in zip(keys, colors):
            y = col(k)
            a.plot(it, y, color=c, alpha=0.22, lw=0.9)
            if n >= 2:
                a.plot(mit if n >= W else it, mov(y), color=c, lw=2.0,
                       label=f"{k.replace('c_','')} (slope {slope(y):+.4f})")
        a.set_title(title, fontsize=10); a.set_xlabel("outer iter")
        if ylog:
            a.set_yscale("log")
        a.legend(fontsize=7.5, frameon=False)

    # 1. THE objective — spectral steering signals (want DOWN)
    panel(ax[0, 0], ["c_spec_highk", "c_spec"], "Spectral deficit  (want ↓ = objective)",
          ["#3ca951", "#4269d0"])
    # 2. PDE guard (want flat/down; UP while spec↓ = wrong-phase hacking)
    panel(ax[0, 1], ["c_pde"], "PDE residual  (guard — UP+spec↓ = hacking)", ["#a463f2"])
    # 3. reward (want UP)
    a = ax[0, 2]
    rm, rs = col("reward_mean"), col("reward_std")
    a.fill_between(it, rm - rs, rm + rs, color="#efb118", alpha=0.12)
    a.plot(it, rm, color="#efb118", alpha=0.3, lw=0.9)
    a.plot(mit if n >= W else it, mov(rm), color="#efb118", lw=2.2, label=f"reward (slope {slope(rm):+.4f})")
    a.set_title("Reward  (want ↑; band = ±std)", fontsize=10); a.set_xlabel("outer iter")
    a.legend(fontsize=8, frameon=False)
    # 4. weak guards
    panel(ax[1, 0], ["c_energy", "c_w1"], "Energy / W1 guards (weak sensors)", ["#efb118", "#ff725c"])
    # 5. diversity + advantage (advantage signal health)
    a = ax[1, 1]
    a.plot(it, col("reward_std"), color="#6cc5b0", alpha=0.3, lw=0.9)
    a.plot(mit if n >= W else it, mov(col("reward_std")), color="#6cc5b0", lw=2.0, label="reward_std (diversity)")
    a.plot(it, col("adv_abs_mean"), color="#9c6b4e", alpha=0.3, lw=0.9)
    a.plot(mit if n >= W else it, mov(col("adv_abs_mean")), color="#9c6b4e", lw=2.0, label="|advantage|")
    a.set_title("Advantage signal health  (want reward_std high)", fontsize=10); a.set_xlabel("outer iter")
    a.legend(fontsize=8, frameon=False)
    # 6. loss (updates biting?)
    panel(ax[1, 2], ["loss_first", "loss_last"], "PPO surrogate loss  (last≪0 = updates biting)",
          ["#9498a0", "#e45756"])
    # 7. THE MONEY METRIC — live GT hi-k retention probe (held-out, GT=1.0)
    a = ax[0, 3]
    g = col("gt_hik_ret"); m = ~np.isnan(g)
    if m.any():
        a.plot(it[m], g[m], "o-", color="#e45756", lw=2.0, ms=4, label="GT hik-retention (probe)")
        a.axhline(1.0, color="k", lw=0.9, ls="--", label="GT (=1.0)")
        base_lvl = float(g[m][:2].mean())
        a.axhline(base_lvl, color="gray", lw=0.9, ls=":", label=f"~start {base_lvl:.2f}")
        a.set_ylim(min(base_lvl, g[m].min()) - 0.03, max(1.05, g[m].max() + 0.03))
    else:
        a.text(0.5, 0.5, "no GT probe\n(added mid-project)", ha="center", va="center", transform=a.transAxes)
    a.set_title("GT hi-k retention  (held-out probe — the real metric)", fontsize=10)
    a.set_xlabel("outer iter"); a.legend(fontsize=8, frameon=False)
    ax[1, 3].axis("off")   # spare

    # verdict line
    sh, pd, rw = slope(col("c_spec_highk")), slope(col("c_pde")), slope(col("reward_mean"))
    if sh < -0.001 and pd < 0.005:
        verdict = "GOOD: deficit ↓, physics stable"
    elif sh < -0.001 and pd > 0.01:
        verdict = "WATCH: deficit ↓ but pde ↑ (possible wrong-phase / spectral gaming)"
    elif abs(sh) < 0.001:
        verdict = "FLAT: no deficit movement (under-powered / plateau)"
    else:
        verdict = "WATCH: deficit rising"
    fig.suptitle(f"DDPO Re=1000 — {n} iters | spec_highk slope {sh:+.4f}  pde {pd:+.4f}  reward {rw:+.4f}\n{verdict}",
                 y=1.02, fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"{n} iters -> {out}")
    print(f"  spec_highk slope {sh:+.4f} | pde slope {pd:+.4f} | reward slope {rw:+.4f}")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args)
