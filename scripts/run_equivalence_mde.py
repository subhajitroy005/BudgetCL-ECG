#!/usr/bin/env python3
"""Paper 1 Task T2 Step 4 — pre-registered equivalence (TOST), MDE, effect
sizes, and declared sensitivities.

HARD GATE: this script recomputes SHA-256 of the pre-registration plan and
compares it to preregistration/PREREG_LOCK.txt. If they differ, or the lock is
absent, it aborts WITHOUT computing any test statistic. This enforces that no
equivalence or MDE result predates the locked plan.

Everything follows preregistration/paper1_stat_plan.md exactly:
  - unit = patient, n = 21, seeds aggregated by MEDIAN before any test
  - family of 6: A4-A1, A5-A1, A4-A2, A4-A3, A4-A0, A5-A0
  - superiority: two-sided Wilcoxon, Holm over the 6
  - equivalence: nonparametric TOST via 90% BCa CI of the median paired
    difference, contained in [-0.02, +0.02]
  - effect size: rank-biserial + BCa CI
  - MDE: Monte Carlo, power >= 0.80, Holm alpha for the comparison's position
  - sensitivities: (a) drop record 232; (b) mean aggregation

Run:  PYTHONNOUSERSITE=1 python3 scripts/run_equivalence_mde.py
Deterministic (fixed seed 20260719).
"""

from __future__ import annotations

import csv
import hashlib
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from budget_stats import holm_correct, paired_test, rank_biserial  # noqa: E402
from budget_stats.bca import BCA_SEED, bca_ci  # noqa: E402

warnings.filterwarnings("ignore", message="Exact p-value calculation")

PLAN = ROOT / "preregistration" / "paper1_stat_plan.md"
LOCK = ROOT / "preregistration" / "PREREG_LOCK.txt"
DATA = ROOT / "results" / "primary" / "E7_patient_seed_results.csv"

PRIMARY_ARMS = ["A0", "A1", "A2", "A3", "A4", "A5"]
FAMILY = ["A4_vs_A1", "A5_vs_A1", "A4_vs_A2", "A4_vs_A3", "A4_vs_A0", "A5_vs_A0"]
DELTA = 0.02
M = len(FAMILY)
DELTA_GRID = np.round(np.arange(0.0, 0.2001, 0.005), 4)
MC_ITERS = 10_000
SEED = BCA_SEED


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def assert_lock() -> None:
    if not LOCK.exists() or not PLAN.exists():
        sys.exit("ABORT: pre-registration lock or plan missing — the T2 gate is not satisfied.")
    want = None
    for line in LOCK.read_text().splitlines():
        if line.strip().startswith("plan_sha256:"):
            want = line.split(":", 1)[1].strip()
    got = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    if want != got:
        sys.exit(f"ABORT: plan hash mismatch.\n  locked: {want}\n  actual: {got}\n"
                 "The statistical plan changed after the lock — refusing to run.")
    print(f"lock OK: plan sha256 {got} matches PREREG_LOCK.txt")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_cells() -> dict[str, dict[str, list[float]]]:
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with DATA.open() as fh:
        for row in csv.DictReader(fh):
            v = row["macro_f1_present"]
            if v == "":
                continue
            per[row["arm"]][row["record"]].append(float(v))
    return per


def aggregate(per, fn, drop=()):
    return {a: {r: float(fn(v)) for r, v in d.items() if r not in drop}
            for a, d in per.items()}


def paired_diffs(scores, comparison):
    left, right = comparison.split("_vs_")
    recs = sorted(set(scores[left]) & set(scores[right]))
    a = np.array([scores[left][r] for r in recs])
    b = np.array([scores[right][r] for r in recs])
    return a, b, (a - b)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def superiority_family(scores):
    raw = {}
    cache = {}
    for c in FAMILY:
        a, b, d = paired_diffs(scores, c)
        res = paired_test(a, b)
        raw[c] = res.p_value
        cache[c] = (a, b, d, res)
    holm = holm_correct(raw)
    return raw, holm, cache


def run_primary(scores):
    raw, holm, cache = superiority_family(scores)
    paired_rows, tost_rows = [], []
    for c in FAMILY:
        a, b, d, res = cache[c]
        assert len(d) == 21, f"{c}: n={len(d)} (pseudoreplication guard: must be 21)"
        ci95 = bca_ci(d, np.median, alpha=0.05)
        rb_ci = bca_ci(d, rank_biserial, alpha=0.05)
        ci90 = bca_ci(d, np.median, alpha=0.10)      # TOST 90% CI
        equivalent = (ci90[0] >= -DELTA) and (ci90[1] <= DELTA)
        pct_imp = round(100.0 * float(np.mean(d > 0)), 1)
        from scipy.stats import wilcoxon
        wstat = float(wilcoxon(d, zero_method="wilcox").statistic)
        paired_rows.append({
            "comparison": c, "n": len(d),
            "median_difference": round(float(np.median(d)), 4),
            "mean_difference": round(float(d.mean()), 4),
            "ci95_bca_low": round(ci95[0], 4), "ci95_bca_high": round(ci95[1], 4),
            "wilcoxon_stat": round(wstat, 1),
            "p_wilcoxon": round(raw[c], 6), "p_holm": round(holm[c], 6),
            "rank_biserial": round(res.effect_size, 3),
            "rb_ci_low": round(rb_ci[0], 3), "rb_ci_high": round(rb_ci[1], 3),
            "pct_improved": pct_imp,
        })
        tost_rows.append({
            "comparison": c, "delta": DELTA,
            "median_difference": round(float(np.median(d)), 4),
            "ci90_bca_low": round(ci90[0], 4), "ci90_bca_high": round(ci90[1], 4),
            "equivalent": equivalent,
        })
    return paired_rows, tost_rows, raw, holm


def holm_alpha_by_position(raw):
    """Holm critical alpha per comparison, from observed p-value ranking."""
    order = sorted(FAMILY, key=lambda c: raw[c])
    out = {}
    for k, c in enumerate(order, start=1):     # k = 1..M ascending
        out[c] = 0.05 / (M - k + 1)
    return out


def _wilcoxon_p_vec(samples: np.ndarray) -> np.ndarray:
    """Vectorised two-sided Wilcoxon signed-rank p-value, normal approximation
    (zeros dropped, scipy convention, no continuity correction). samples is
    (M, n); returns (M,) p-values. At n=21 with dropped zeros this is exactly
    the branch SciPy selects. Ties among nonzero |d| use ordinal ranks (tied
    magnitudes are negligibly rare in continuous macro-F1 differences)."""
    from scipy.stats import norm
    absv = np.abs(samples)
    nz = absv > 0
    r = nz.sum(axis=1).astype(float)                       # nonzero count/row
    # ordinal ranks 1..n per row with zeros pushed to the end (then masked).
    order = np.argsort(np.where(nz, absv, np.inf), axis=1)
    ranks = np.argsort(order, axis=1) + 1                  # rank of each element
    sign = np.sign(samples)
    r_plus = np.where(nz & (sign > 0), ranks, 0).sum(axis=1)
    r_minus = np.where(nz & (sign < 0), ranks, 0).sum(axis=1)
    T = np.minimum(r_plus, r_minus)
    mn = r * (r + 1) / 4.0
    var = r * (r + 1) * (2 * r + 1) / 24.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (T - mn) / np.sqrt(var)
        p = 2.0 * norm.cdf(-np.abs(z))
    p = np.where(r > 0, p, 1.0)                            # degenerate row: no reject
    return p


def run_mde(scores, raw):
    alphas = holm_alpha_by_position(raw)
    rng = np.random.default_rng(SEED)
    mde_rows, curve_rows = [], []
    for c in FAMILY:
        _, _, d = paired_diffs(scores, c)
        n = len(d)
        centered = d - np.median(d)            # impose the null
        alpha = alphas[c]
        mde = None
        for delta in DELTA_GRID:
            shifted = centered + delta
            idx = rng.integers(0, n, size=(MC_ITERS, n))
            samples = shifted[idx]                          # (MC_ITERS, n)
            p = _wilcoxon_p_vec(samples)
            power = float(np.mean(p < alpha))
            curve_rows.append({"comparison": c, "delta": float(delta),
                               "power": round(power, 4), "holm_alpha": round(alpha, 5)})
            if mde is None and power >= 0.80:
                mde = float(delta)
        mde_rows.append({"comparison": c, "holm_alpha": round(alpha, 5),
                         "mde_power80": mde if mde is not None else ">0.20"})
    return mde_rows, curve_rows


def run_sensitivity(scores, tag):
    raw, holm, cache = superiority_family(scores)
    rows = []
    for c in FAMILY:
        a, b, d, res = cache[c]
        ci90 = bca_ci(d, np.median, alpha=0.10)
        rows.append({
            "comparison": c, "n": len(d),
            "median_difference": round(float(np.median(d)), 4),
            "mean_difference": round(float(d.mean()), 4),
            "p_wilcoxon": round(raw[c], 6), "p_holm": round(holm[c], 6),
            "ci90_bca_low": round(ci90[0], 4), "ci90_bca_high": round(ci90[1], 4),
            "equivalent": (ci90[0] >= -DELTA) and (ci90[1] <= DELTA),
        })
    return rows


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #
def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def write_table_v(paired, tost, mde, path):
    tost_by = {r["comparison"]: r for r in tost}
    mde_by = {r["comparison"]: r for r in (mde or [])}
    lines = [
        r"% Auto-generated by scripts/run_equivalence_mde.py (T2). Do not hand-edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{The six pre-registered paired comparisons under primary "
        r"median-seed aggregation ($n = 21$). Superiority is two-sided Wilcoxon "
        r"with Holm correction; equivalence is the nonparametric TOST "
        r"($90\%$ BCa interval of the median paired difference contained in "
        r"$[-\delta,+\delta]$, $\delta = 0.02$). MDE is the minimum median shift "
        r"detectable at $80\%$ power. Bold marks comparisons surviving Holm.}",
        r"\label{tab:pairwise_tests}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrcr}",
        r"\toprule",
        r"Comparison & Median $\Delta$ & 95\% BCa CI & Holm $p$ & $r_{rb}$ & "
        r"90\% CI (TOST) & Equiv. & MDE \\",
        r"\midrule",
    ]
    for r in paired:
        c = r["comparison"].replace("_vs_", "--")
        t = tost_by[r["comparison"]]
        m = mde_by.get(r["comparison"], {})
        eq = r"\checkmark" if t["equivalent"] else r"--"
        mde = m.get("mde_power80", "--")
        mde_s = f"{mde:.3f}" if isinstance(mde, (int, float)) else str(mde)
        holm = f"\\textbf{{{r['p_holm']:.3f}}}" if r["p_holm"] < 0.05 else f"{r['p_holm']:.3f}"
        lines.append(
            f"{c} & {r['median_difference']:+.4f} & "
            f"[{r['ci95_bca_low']:+.3f}, {r['ci95_bca_high']:+.3f}] & "
            f"{holm} & {r['rank_biserial']:+.2f} & "
            f"[{t['ci90_bca_low']:+.3f}, {t['ci90_bca_high']:+.3f}] & {eq} & {mde_s} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def make_fig05(scores, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "figures" / "scripts"))
    from _style import apply_style
    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.axvspan(-DELTA, DELTA, color="0.85", label=f"equivalence band ±{DELTA}")
    ax.axvline(0, color="k", lw=0.8)
    for i, c in enumerate(FAMILY):
        _, _, d = paired_diffs(scores, c)
        med = float(np.median(d))
        lo, hi = bca_ci(d, np.median, alpha=0.10)
        ax.errorbar(med, i, xerr=[[med - lo], [hi - med]], fmt="o", color="C0", capsize=3)
    ax.set_yticks(range(len(FAMILY)))
    ax.set_yticklabels([c.replace("_vs_", "–") for c in FAMILY])
    ax.set_xlabel("Median paired difference in macro-F1 (90% BCa CI)")
    ax.set_title("Paired differences vs. pre-registered ±δ equivalence band")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.savefig(path, metadata={"CreationDate": None}); plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    assert_lock()
    per = load_cells()

    primary = aggregate(per, np.median)                 # pre-registered primary
    paired_rows, tost_rows, raw, holm = run_primary(primary)
    write_csv(ROOT / "results" / "e8_paired_tests.csv", paired_rows)
    write_csv(ROOT / "results" / "e8_tost_equivalence.csv", tost_rows)

    mde_rows, curve_rows = run_mde(primary, raw)
    write_csv(ROOT / "results" / "e8_mde.csv", mde_rows)
    write_csv(ROOT / "results" / "e8_mde_power_curves.csv", curve_rows)

    sens_232 = run_sensitivity(aggregate(per, np.median, drop={"232"}), "no232")
    write_csv(ROOT / "results" / "e8_sensitivity_record232.csv", sens_232)
    sens_mean = run_sensitivity(aggregate(per, np.mean), "mean")
    write_csv(ROOT / "results" / "e8_sensitivity_mean_aggregation.csv", sens_mean)

    write_table_v(paired_rows, tost_rows, mde_rows,
                  ROOT / "manuscript" / "tables" / "table_V_paired_comparisons.tex")
    try:
        make_fig05(primary, ROOT / "figures" / "paper" / "fig05_paired_differences.pdf")
    except Exception as exc:
        print(f"WARNING: fig05 skipped: {exc}", file=sys.stderr)

    print("\n=== TOST equivalence verdicts (delta=0.02) ===")
    for r in tost_rows:
        print(f"  {r['comparison']:9s} medΔ={r['median_difference']:+.4f} "
              f"90%CI=[{r['ci90_bca_low']:+.4f},{r['ci90_bca_high']:+.4f}] "
              f"equivalent={r['equivalent']}")
    print("\n=== MDE @80% power ===")
    for r in mde_rows:
        print(f"  {r['comparison']:9s} holm_alpha={r['holm_alpha']:.5f} MDE={r['mde_power80']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
