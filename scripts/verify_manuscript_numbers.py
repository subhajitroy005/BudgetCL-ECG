#!/usr/bin/env python3
"""Check that what the manuscript SAYS matches what the artifacts CONTAIN.

This closes the traceability loop. Everything upstream generates numbers; this
script reads the LaTeX literally and diffs it against the released CSVs and the
byte solver.

It has caught real defects, including a figure that kept shipping record-level
external results months after the subject-level analysis existed. A number that
is merely "probably still right" is not verified.

Exit code is non-zero on any FAIL.

Policy: a missing required artifact and a reappearing retired claim are both
FAIL, not WARN. WARN is reserved for genuinely non-blocking conditions, so that
CI cannot go green while a publication artifact is absent or an unsupported
claim has crept back into the manuscript.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_cl.data import MITBIH_DS2_PRIMARY_21  # noqa: E402
from budget_cl.memory import calculate_arm_memory  # noqa: E402
from budget_cl.utils import repo_root  # noqa: E402

REPO = repo_root()
PASS: list[str] = []
WARN: list[str] = []
FAIL: list[str] = []

# Phrases the paper must not contain: each was retired for a stated reason.
FORBIDDEN = {
    "measured sram": "no hardware SRAM was measured",
    "measured latency": "no latency was measured",
    "ordering is preserved": "the reserve replication does NOT preserve arm ordering",
    "minority-class-strengthened": "the alternative checkpoints were never shown to be stronger",
    "held untouched": "the test signal is not strictly unseen near the boundary",
    "provably does not change": "TOST supports equivalence within a margin, not proof",
    # Retired in the causal-language pass: this is a controlled-ablation design,
    # not a formal causal-identification design.
    "allocation causally": "controlled ablations, not causal identification",
    "causal matrix": "controlled ablation matrix",
    "four causal contrasts": "four controlled contrasts",
    "not itself limiting": "exceeds the measured 12-record/3-seed panel scope",
    "run-to-run variation": "E17 reuses the same seeds; the changes are procedural",
    "pathologically weak": "no frozen unseen-patient baseline supports this",
    "pathological baseline": "no frozen unseen-patient baseline supports this",
    "no benefit at all": "scope to the tested selector, ratio, optimizer, cohort",
    "exceeds head-only": "mean difference only; no paired interval reported",
    "analytical compute": "the numerical compute model was withdrawn",
    "mac counts": "the numerical compute model was withdrawn",
}


def load_tex() -> dict[str, str]:
    """Read every manuscript .tex file."""
    files = {}
    manuscript = REPO / "manuscript"
    for path in list(manuscript.glob("*.tex")) + list(manuscript.rglob("sections/*.tex")) + list(
        manuscript.rglob("tables/*.tex")
    ):
        files[path.name] = path.read_text()
    return files


def read_csv(rel: str) -> list[dict]:
    path = REPO / rel
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def check(condition: bool, message: str) -> None:
    (PASS if condition else FAIL).append(message)


def verify_cohort(tex: dict[str, str]) -> None:
    """The paper must claim the corrected 21-record cohort throughout."""
    body = "\n".join(tex.values())
    check(len(MITBIH_DS2_PRIMARY_21) == 21, f"cohort is {len(MITBIH_DS2_PRIMARY_21)} records")
    check("21-record" in body or "21 patient-disjoint" in body,
          "manuscript states the 21-record cohort")
    check("630" in body, "manuscript states the 630-cell primary grid")


def verify_arm_means(tex: dict[str, str]) -> None:
    """Arm means in the generated table must equal the released CSV."""
    rows = read_csv("results/primary/E7_arm_summary.csv")
    if not rows:
        FAIL.append("required artifact missing: results/primary/E7_arm_summary.csv")
        return
    table = tex.get("table_primary_arms.tex", "")
    for row in rows:
        mean = f"{float(row['mean']):.3f}"
        check(mean in table, f"table_primary_arms contains {row['arm']} mean {mean}")


def verify_paired_tests(tex: dict[str, str]) -> None:
    """Every p-value in the comparison table must come from the CSV."""
    rows = read_csv("results/primary/E8_paired_tests.csv")
    if not rows:
        FAIL.append("required artifact missing: results/primary/E8_paired_tests.csv")
        return
    table = tex.get("table_pairwise_tests.tex", "")
    survivors = set()
    for row in rows:
        holm = float(row["p_holm"])
        check(f"{holm:.4f}" in table,
              f"table_pairwise_tests contains {row['comparison']} Holm p={holm:.4f}")
        if holm < 0.05:
            survivors.add(row["comparison"])
    check(survivors == {"A4_vs_A0", "A5_vs_A0"},
          f"only frozen-model comparisons survive Holm (found {sorted(survivors)})")


def verify_prose_matches_tables(tex: dict[str, str]) -> None:
    """Every CI quoted in PROSE must match the released CSV.

    The verifier previously checked only the generated tables, so a stale
    interval sitting in the running text could survive indefinitely -- and one
    did: the prose quoted [0.052,0.239] while the table and CSV said
    [0.055,0.239]. Both came from valid bootstrap runs with different RNG
    seeds, which is exactly why a machine check is needed rather than a reading.

    Bootstrap endpoints move in the third decimal across seeds, so the CSV
    (fixed seed 20260719) is the single source of truth.
    """
    rows = read_csv("results/primary/E8_paired_tests.csv")
    if not rows:
        FAIL.append(
            "required artifact missing: results/primary/E8_paired_tests.csv (prose CI check)"
        )
        return
    prose = "\n".join(
        text for name, text in tex.items() if name.startswith(("05_", "06_", "08_"))
    )
    for row in rows:
        if row["comparison"] not in ("A4_vs_A0", "A5_vs_A0"):
            continue  # only these two are quoted numerically in the prose
        lo, hi = float(row["ci_low"]), float(row["ci_high"])
        # Compare with whitespace stripped so LaTeX line wrapping cannot hide a
        # mismatch, and accept both the bare and sign-prefixed renderings.
        flat = "".join(prose.split())
        wanted = [f"[{lo:.3f},{hi:.3f}]", f"[+{lo:.3f},+{hi:.3f}]"]
        check(
            any(w in flat for w in wanted),
            f"prose quotes {row['comparison']} CI [{lo:.3f},{hi:.3f}] matching the CSV",
        )


def verify_tost_count(tex: dict[str, str]) -> None:
    """The "N of 18" claim must equal the equivalence count in the TOST CSV.

    The generated table and the prose previously disagreed: one table cell
    marked rank-2 encoder LoRA at 32 KiB as NOT equivalent because its Wilcoxon
    p was significant. Significance and equivalence are different questions --
    that contrast is both significantly different from zero AND equivalent
    within the margin -- so the table said 13 while the CSV and prose said 14.
    """
    rows = read_csv("results/budget_sweep/E10_tost.csv")
    if not rows:
        FAIL.append("required artifact missing: results/budget_sweep/E10_tost.csv")
        return
    n_equiv = sum(1 for r in rows if r["equivalent_at_margin"].strip().lower() == "true")
    body = "\n".join(tex.values())
    check(
        f"{n_equiv} of {len(rows)}" in body,
        f"manuscript states '{n_equiv} of {len(rows)}' TOST equivalences, matching the CSV",
    )
    table = tex.get("table_e10_paired.tex", "")
    yes = table.count("& yes \\\\")
    no = table.count("& no \\\\")
    check(
        yes == n_equiv and no == len(rows) - n_equiv,
        f"TOST table has {yes} yes / {no} no cells, matching the CSV "
        f"({n_equiv} / {len(rows) - n_equiv})",
    )


def verify_byte_totals(tex: dict[str, str]) -> None:
    """Replay counts and byte totals must match the solver, not a transcript."""
    table = tex.get("table_arm_budget.tex", "")
    body = "\n".join(tex.values())
    for arm in ("A1", "A4", "A5"):
        primary = calculate_arm_memory(arm)
        reserved = calculate_arm_memory(arm, reserve_bytes=1024)
        check(str(primary.replay_items) in table,
              f"budget table contains {arm} replay count {primary.replay_items}")
        check(str(reserved.replay_items) in table,
              f"budget table contains {arm} reserved count {reserved.replay_items}")
        check(primary.used_bytes <= 16384, f"{arm} fits 16 KiB ({primary.used_bytes} B)")
    check("203" in body and "21" in body, "manuscript states serialized record sizes")


def verify_splitfirst(tex: dict[str, str]) -> None:
    """E17 arm means in the paper must match the released sensitivity CSV."""
    rows = read_csv("results/preprocessing_sensitivity/E17_arm_summary.csv")
    if not rows:
        FAIL.append(
            "required artifact missing: "
            "results/preprocessing_sensitivity/E17_arm_summary.csv"
        )
        return
    table = tex.get("table_splitfirst.tex", "")
    for row in rows:
        mean = f"{float(row['mean']):.3f}"
        check(mean in table, f"table_splitfirst contains {row['arm']} split-first mean {mean}")


def verify_forbidden(tex: dict[str, str]) -> None:
    """Retired claims must not reappear."""
    for phrase, reason in FORBIDDEN.items():
        hits = [name for name, text in tex.items() if phrase in text.lower()]
        if hits:
            # A retired claim reappearing is a release blocker, not a note: the
            # whole point of the list is that these statements were removed
            # because the evidence does not support them.
            FAIL.append(f"forbidden phrase '{phrase}' found in {hits}: {reason}")
        else:
            PASS.append(f"forbidden phrase '{phrase}' absent ({reason})")


def verify_generated_tables_are_marked() -> None:
    """Generated tables must carry the do-not-edit banner."""
    for name in ("table_primary_arms.tex", "table_pairwise_tests.tex",
                 "table_arm_budget.tex", "table_splitfirst.tex"):
        path = REPO / "manuscript" / "tables" / name
        if not path.exists():
            FAIL.append(f"required generated table missing: manuscript/tables/{name}")
            continue
        check("GENERATED FILE" in path.read_text(), f"{name} is marked as generated")


def main() -> int:
    tex = load_tex()
    if not tex:
        print("error: no manuscript .tex files found", file=sys.stderr)
        return 1

    verify_cohort(tex)
    verify_arm_means(tex)
    verify_paired_tests(tex)
    verify_prose_matches_tables(tex)
    verify_tost_count(tex)
    verify_byte_totals(tex)
    verify_splitfirst(tex)
    verify_forbidden(tex)
    verify_generated_tables_are_marked()

    print("\n" + "-" * 78)
    for item in PASS:
        print(f"  PASS  {item}")
    for item in WARN:
        print(f"  WARN  {item}")
    for item in FAIL:
        print(f"  FAIL  {item}")
    print("-" * 78)
    print(f"PASS {len(PASS)}   WARN {len(WARN)}   FAIL {len(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
