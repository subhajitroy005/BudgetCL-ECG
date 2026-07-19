#!/usr/bin/env python3
"""Generate every LaTeX result table from the released CSVs.

Step 5 of the reproduction chain. Result values are NEVER typed into
``main.tex`` by hand -- ``manuscript/tables/*.tex`` is generated output, and
regenerating it is how a changed result propagates into the paper.

Outputs:
    manuscript/tables/table_primary_arms.tex     arm means, CIs (E7)
    manuscript/tables/table_pairwise_tests.tex   six comparisons (E8)
    manuscript/tables/table_arm_budget.tex       byte accounting
    manuscript/tables/table_splitfirst.tex       preprocessing sensitivity (E17)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_cl.memory import calculate_arm_memory  # noqa: E402
from budget_cl.utils import get_logger, repo_root  # noqa: E402

LOG = get_logger("make_tables")

ARM_LABELS = {
    "A0": "A0 (frozen)",
    "A1": r"A1 (head $+$ max replay)",
    "A2": r"A2 (post-pool $r{=}1$)",
    "A3": r"A3 (post-pool $r{=}2$)",
    "A4": r"A4 (LoRA $r{=}1$)",
    "A5": r"A5 (LoRA $r{=}2$)",
}


def read_csv(path: Path) -> list[dict]:
    """Read a CSV into a list of dicts.

    Raises:
        FileNotFoundError: pointing at the regeneration command, because a
            missing input here means the statistics step has not been run.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `make statistics` first")
    with path.open() as fh:
        return list(csv.DictReader(fh))


def write(path: Path, body: str) -> None:
    """Write a generated table, with a do-not-edit banner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    banner = (
        "% GENERATED FILE -- do not edit by hand.\n"
        "% Regenerate with: make tables  (scripts/make_tables.py)\n"
    )
    path.write_text(banner + body)
    LOG.info("wrote %s", path.relative_to(repo_root()))


def table_primary_arms(rows: list[dict]) -> str:
    """Per-arm mean, SD, median and patient-bootstrap CI."""
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Per-patient \macrof{} over the 21-record patient-disjoint "
        r"cohort, five seeds averaged within patient. Intervals are patient "
        r"bootstraps ($10{,}000$ resamples).}",
        r"\label{tab:primary_arms}",
        r"\begin{tabular}{lrrrl}", r"\toprule",
        r"Arm & Mean & SD & Median & 95\% CI \\", r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{ARM_LABELS.get(r['arm'], r['arm'])} & ${float(r['mean']):.3f}$ & "
            f"${float(r['sd']):.3f}$ & ${float(r['median']):.3f}$ & "
            f"$[{float(r['ci_low']):.3f}, {float(r['ci_high']):.3f}]$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_pairwise(rows: list[dict]) -> str:
    """The six pre-specified comparisons; Holm survivors in bold."""
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{The six pre-specified paired comparisons ($n = 21$). Bold "
        r"marks the comparisons surviving Holm correction at $\alpha = 0.05$. "
        r"\textbf{Only the two against the frozen model survive.}}",
        r"\label{tab:pairwise_tests}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrcrrrr}", r"\toprule",
        r"Comparison & Mean $\Delta$ & Med. $\Delta$ & 95\% CI & Wilcoxon $p$ "
        r"& Holm $p$ & $r_{\mathrm{rb}}$ & Impr./Wors. \\",
        r"\midrule",
    ]
    for r in rows:
        name = r["comparison"].replace("_vs_", " vs.\\ ")
        holm = float(r["p_holm"])
        bold = holm < 0.05
        fmt = (lambda s: rf"\textbf{{{s}}}") if bold else (lambda s: s)
        lines.append(
            f"{fmt(name)} & ${float(r['mean_difference']):+.3f}$ & "
            f"${float(r['median_difference']):+.3f}$ & "
            f"$[{float(r['ci_low']):+.3f},{float(r['ci_high']):+.3f}]$ & "
            f"${float(r['p_wilcoxon']):.4f}$ & {fmt(f'${holm:.4f}$')} & "
            f"${float(r['rank_biserial']):.2f}$ & "
            f"{r['improved']} / {r['worsened']} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        "",
        r"\vspace{0.4em}",
        r"{\small Improved/worsened counts are threshold-free, matching the "
        r"pre-specified plan; Table~\ref{tab:harm} repeats them at a "
        r"meaningful-change threshold of $0.02$. The A4--A1 and A5--A1 intervals "
        r"include zero, which is the basis for this paper's refusal to claim "
        r"superiority over maximum-volume replay."
        "\n\n"
        r"\emph{Why an interval can span zero while the test is significant.} "
        r"Two rows pair a bootstrap interval containing zero with a raw Wilcoxon "
        r"$p$ below $0.05$. This is not an inconsistency: the procedures target "
        r"\emph{different estimands}. The bootstrap interval estimates the "
        r"\emph{mean} paired difference, which a few large negative patients can "
        r"drag toward zero, whereas the signed-rank test evaluates the "
        r"distribution of signed \emph{ranks} and is sensitive to a consistent "
        r"direction of change even when magnitudes are small. Neither row is "
        r"claimed as a positive result, since neither survives Holm correction.}",
        r"\end{table}", "",
    ]
    return "\n".join(lines)


def table_arm_budget() -> str:
    """Byte accounting, computed live from the solver rather than transcribed."""
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Persistent-state accounting per arm. Replay counts are "
        r"\emph{derived} from the budget, not chosen. ``Reserved'' re-solves "
        r"each arm against a $15{,}360$\,B ceiling (1\,KiB firmware reserve).}",
        r"\label{tab:arm_definitions}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Arm & Trainable & Replay & Items & Items & Bytes \\",
        r"    & params    & record & (16\,KiB) & (reserved) & used \\",
        r"\midrule",
    ]
    for arm in ("A0", "A1", "A2", "A3", "A4", "A5"):
        primary = calculate_arm_memory(arm)
        reserved = calculate_arm_memory(arm, reserve_bytes=1024)
        record = "---" if primary.replay_items == 0 else (
            r"203\,B" if arm in ("A4", "A5") else r"21\,B"
        )
        # Apply the LaTeX thousands separator to the byte figure ONLY -- doing
        # it to the whole line corrupts the "\,B" unit spacing.
        used = f"{primary.used_bytes:,}".replace(",", "{,}")
        lines.append(
            f"{arm} & {primary.trainable_parameters} & {record} & "
            f"{primary.replay_items} & {reserved.replay_items} & {used} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_splitfirst(primary: list[dict], split: list[dict]) -> str:
    """E17 whole-record versus split-first arm means."""
    by_arm = {r["arm"]: float(r["mean"]) for r in primary}
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Boundary-leakage sensitivity (E17). The split-first rerun "
        r"cuts the raw signal at the boundary, discards a $2$\,s guard each "
        r"side, and filters the pieces independently, so no adaptation-side "
        r"sample can depend on a test-side raw sample. \textbf{Every "
        r"conclusion is preserved.}}",
        r"\label{tab:splitfirst}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Arm & Whole-record & Split-first & $\Delta$ \\", r"\midrule",
    ]
    for r in split:
        arm = r["arm"]
        base, new = by_arm.get(arm), float(r["mean"])
        delta = f"${new - base:+.3f}$" if base is not None else "---"
        base_s = f"${base:.3f}$" if base is not None else "---"
        lines.append(f"{ARM_LABELS.get(arm, arm)} & {base_s} & ${new:.3f}$ & {delta} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_splitfirst_contrasts(rows: list[dict]) -> str:
    """E17 paired contrasts, with raw and family-corrected p-values."""
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Paired contrasts recomputed on the split-first (E17) cohort, "
        r"$n = 21$ patients, five seeds averaged within patient. \textbf{These "
        r"four sensitivity contrasts form their own family and are not pooled "
        r"with the six pre-specified primary comparisons}; the Holm column "
        r"corrects within these four only. Both frozen-model contrasts remain "
        r"significant and neither encoder arm clears maximum head-only replay, "
        r"matching the primary cohort.}",
        r"\label{tab:splitfirst_contrasts}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrcrrr}", r"\toprule",
        r"Contrast & Mean $\Delta$ & Med. $\Delta$ & 95\% CI & Raw $p$ "
        r"& Holm $p$ & Impr./Wors. \\",
        r"\midrule",
    ]
    for r in rows:
        name = r["comparison"].replace("_vs_", "$-$")
        holm = float(r["p_holm_within_sensitivity_family"])
        bold = holm < 0.05
        fmt = (lambda s: rf"\textbf{{{s}}}") if bold else (lambda s: s)
        lines.append(
            f"{fmt(name)} & ${float(r['mean_difference']):+.3f}$ & "
            f"${float(r['median_difference']):+.3f}$ & "
            f"$[{float(r['ci_low']):+.3f},{float(r['ci_high']):+.3f}]$ & "
            f"${float(r['p_wilcoxon_raw']):.4f}$ & {fmt(f'${holm:.4f}$')} & "
            f"{r['improved']} / {r['worsened']} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        "",
        r"\vspace{0.4em}",
        r"{\small ``Raw $p$'' is the unadjusted paired Wilcoxon signed-rank "
        r"value; ``Holm $p$'' corrects across these four sensitivity contrasts. "
        r"Intervals are patient bootstraps ($10{,}000$ resamples).}",
        r"\end{table}", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(repo_root() / "results"))
    parser.add_argument("--out-dir", default=str(repo_root() / "manuscript" / "tables"))
    args = parser.parse_args()

    results, out = Path(args.results_dir), Path(args.out_dir)

    arm_rows = read_csv(results / "primary" / "E7_arm_summary.csv")
    write(out / "table_primary_arms.tex", table_primary_arms(arm_rows))
    write(
        out / "table_pairwise_tests.tex",
        table_pairwise(read_csv(results / "primary" / "E8_paired_tests.csv")),
    )
    write(out / "table_arm_budget.tex", table_arm_budget())

    split_path = results / "preprocessing_sensitivity" / "E17_arm_summary.csv"
    if split_path.exists():
        write(out / "table_splitfirst.tex", table_splitfirst(arm_rows, read_csv(split_path)))
    else:
        LOG.warning("E17 summary missing; skipping table_splitfirst.tex")

    contrasts_path = results / "preprocessing_sensitivity" / "E17_paired_tests.csv"
    if contrasts_path.exists():
        write(
            out / "table_splitfirst_contrasts.tex",
            table_splitfirst_contrasts(read_csv(contrasts_path)),
        )
    else:
        LOG.warning("E17 contrasts missing; skipping table_splitfirst_contrasts.tex")

    LOG.info("tables complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
