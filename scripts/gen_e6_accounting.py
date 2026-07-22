#!/usr/bin/env python3
"""E6 two-axis memory accounting (Paper 1, Experiment Block 1, Task T1).

Produces, per arm, TWO independent resource axes:

  1. Persistent state  -- what the device must retain *between* adaptation steps.
                          Reuses the verified module ``budget_cl.memory.accounting``
                          (the single source of truth for every byte in the paper).
  2. Peak transient     -- the forward+backward working set during ONE adaptation
                          step, under three implementation regimes R1/R2/R3.

The transient axis is the one the 16 KiB budget *excludes*; this task replaces
the paper's assumption ("peak SRAM is out of scope") with an argued, itemised
analytical model. Nothing here is measured on hardware -- it is byte arithmetic
over a statically known graph. The word "measured" does not appear.

Architecture (verified in PAPER1_STATUS.md, param count sums to 6,643):
    T=66 tokens, d_model=16, d_ff=128, n_heads=1, key_dim=16,
    raw beat = 198 samples + 2 RR, 5 AAMI classes, 1 encoder block.

n_heads=1/key_dim=16 is fixed by the spec's own arithmetic
("attention matrix ~ T^2 * 4 B ~ 17 KiB" == 66^2*4 = 17,424 B) and is invariant
in the param count (heads*key_dim=16 either way).

Run:  PYTHONNOUSERSITE=1 python3 scripts/gen_e6_accounting.py
Deterministic, config-free, stdlib-only for the accounting; matplotlib only for
the two figures.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from budget_cl.memory.accounting import account_arm  # noqa: E402

# --------------------------------------------------------------------------- #
# Architecture constants
# --------------------------------------------------------------------------- #
T = 66            # tokens
D_MODEL = 16      # embed dim / pooled width
D_FF = 128        # feed-forward hidden
N_HEADS = 1       # heads*key_dim = d_model = 16 (fixed by spec arithmetic)
KEY_DIM = 16
RAW_SAMPLES = 198
RR = 2
CLASSES = 5
PROTO_DIM = 18    # pooled(16) + RR(2), the post-pool replay vector

ARMS = ["A0", "A1", "A2", "A3", "A4", "A5"]
REGIMES = ["R1", "R2", "R3"]
BUDGET = 16_384

# Activation dtype width per regime (bytes). R1 is FP32; R2/R3 store FP16.
REGIME_ACT_BYTES = {"R1": 4, "R2": 2, "R3": 2}

# --------------------------------------------------------------------------- #
# Forward activation tensors (element counts) for the FULL-model path (A4/A5).
# One encoder block, batch 1. Each entry is a distinct allocation under the
# "no in-place" naive model. Sizes are exact.
# --------------------------------------------------------------------------- #
FULL_FORWARD = [
    ("ecg_in", RAW_SAMPLES),          # 198  raw beat
    ("rr_in", RR),                    # 2
    ("patch_embed", T * D_MODEL),     # 1056 conv tokens
    ("pos_embed", T * D_MODEL),       # 1056
    ("ln1", T * D_MODEL),             # 1056
    ("q", T * D_MODEL),               # 1056
    ("k", T * D_MODEL),               # 1056
    ("v", T * D_MODEL),               # 1056
    ("scores", N_HEADS * T * T),      # 4356 attention matrix
    ("attn_weights", N_HEADS * T * T),# 4356 softmax(scores)
    ("context", T * D_MODEL),         # 1056
    ("attn_out", T * D_MODEL),        # 1056 output projection
    ("add1", T * D_MODEL),            # 1056 residual
    ("ln2", T * D_MODEL),             # 1056
    ("ffn1", T * D_FF),               # 8448 FFN hidden (pre-activation)
    ("relu", T * D_FF),               # 8448 FFN hidden (post-activation)
    ("ffn2", T * D_MODEL),            # 1056
    ("add2", T * D_MODEL),            # 1056 residual (block output)
    ("final_ln", T * D_MODEL),        # 1056
    ("rr_proj", RR),                  # 2
    ("pool", D_MODEL),                # 16  global average pool
    ("fused", PROTO_DIM),             # 18  concat(pool, rr)
    ("logits", CLASSES),              # 5
    ("probs", CLASSES),               # 5
]

# Post-pool path (A1/A2/A3): the STORED replay is the 18-d post-pool vector, so
# the encoder never runs during adaptation. Forward is head (+ optional tiny
# rank-r residual adapter on the 18-d vector).
def postpool_forward(rank: int) -> list[tuple[str, int]]:
    fwd = [("proto_in", PROTO_DIM)]           # 18 stored post-pool vector
    if rank > 0:
        fwd += [("adapter_down", rank),        # 18 -> r
                ("adapter_up", PROTO_DIM)]      # r -> 18 (residual)
    fwd += [("logits", CLASSES), ("probs", CLASSES)]
    return fwd


# --------------------------------------------------------------------------- #
# Transient peak model (analytical). Documented, hand-verifiable formulas.
# --------------------------------------------------------------------------- #
def sum_elems(fwd: list[tuple[str, int]]) -> int:
    return sum(n for _, n in fwd)


def transient_full(regime: str) -> dict:
    """Peak transient working set (bytes) for the full-model path (A4/A5)."""
    b = REGIME_ACT_BYTES[regime]
    s_fwd = sum_elems(FULL_FORWARD)  # 39,582 elements

    if regime in ("R1", "R2"):
        # Naive retain-all: the full forward completes with every activation
        # live; the forward->backward boundary is the peak because backward
        # frees monotonically as it consumes, reusing freed space for each
        # (smaller) gradient temporary. Peak = saved-activation bytes.
        peak_elems = s_fwd
        detail = {"policy": "retain-all", "saved_forward_elems": s_fwd}
        mult = 1.0
    else:  # R3 -- checkpoint at attention/FFN sub-block granularity + recompute
        # Retain only the residual-stream boundary tensors across backward;
        # recompute each sub-block's internals transiently during its backward.
        retained = (
            T * D_MODEL   # pos_embed  (block input, base for recompute)
            + T * D_MODEL # add1       (attention/FFN boundary)
            + T * D_MODEL # add2       (block output)
            + D_MODEL + PROTO_DIM + RR  # pool + fused + rr_proj (head path)
        )
        # FFN backward working set (dominant): recompute ln2 -> ffn1 -> relu from
        # the retained add1, then form g_relu; relu and g_relu coexist.
        ffn_ws = (T * D_MODEL)      # ln2 recomputed
        ffn_ws += (T * D_FF)        # relu (stores the mask)
        ffn_ws += (T * D_FF)        # g_relu gradient
        # Attention backward working set: recompute ln1 -> q,k,v -> scores ->
        # attn_weights, then g_scores; attn_weights and g_scores coexist.
        attn_ws = (T * D_MODEL)             # ln1 recomputed
        attn_ws += 3 * (T * D_MODEL)        # q, k, v
        attn_ws += (N_HEADS * T * T)        # attn_weights
        attn_ws += (N_HEADS * T * T)        # g_scores
        attn_ws += (T * D_MODEL)            # context (for output-proj / LoRA-O grad)
        subblock_ws = max(ffn_ws, attn_ws)
        peak_elems = retained + subblock_ws
        detail = {
            "policy": "checkpoint+recompute",
            "retained_elems": retained,
            "ffn_recompute_ws_elems": ffn_ws,
            "attn_recompute_ws_elems": attn_ws,
            "subblock_ws_elems": subblock_ws,
        }
        mult = recompute_multiplier()

    return {"peak_bytes": peak_elems * b, "peak_elems": peak_elems,
            "act_bytes": b, "compute_multiplier": mult, **detail}


def transient_postpool(regime: str, rank: int) -> dict:
    """Peak transient for A1/A2/A3 (post-pool). No encoder pass at all."""
    b = REGIME_ACT_BYTES[regime]
    fwd = postpool_forward(rank)
    s_fwd = sum_elems(fwd)
    # Backward temporaries on an 18-d path are <= the forward activations; add
    # the single largest gradient tensor (18) to be conservative. Parameter
    # gradients (head/adapter weights) are charged on the PERSISTENT axis, not
    # here. All regimes identical up to dtype width; nothing to rematerialise.
    peak_elems = s_fwd + PROTO_DIM
    return {"peak_bytes": peak_elems * b, "peak_elems": peak_elems,
            "act_bytes": b, "compute_multiplier": 1.0,
            "policy": "no-encoder-pass", "saved_forward_elems": s_fwd}


def transient_frozen(regime: str) -> dict:
    """A0: frozen, no adaptation step. Report the forward-only INFERENCE arena
    (free-after-use), which is the region A4/A5 transient can alias onto.
    Peak forward-only is at the FFN: ln2 feeds ffn1; ffn1 feeds relu."""
    b = REGIME_ACT_BYTES[regime]
    # Streaming forward, free-after-use. Peak = largest live pair on the FFN:
    # relu(8448) is produced while ffn1(8448) is its input (not yet freed).
    peak_elems = (T * D_FF) + (T * D_FF)   # ffn1 + relu coexist at the peak
    return {"peak_bytes": peak_elems * b, "peak_elems": peak_elems,
            "act_bytes": b, "compute_multiplier": 1.0,
            "policy": "inference-arena (no adaptation)",
            "saved_forward_elems": 0}


def recompute_multiplier() -> float:
    """Extra forward FLOPs from R3 recompute / original forward FLOPs (MACs)."""
    patch = T * D_MODEL * 3                       # conv k=3, 1 in-ch
    attn = (3 * T * D_MODEL * D_MODEL             # q,k,v projections
            + T * T * D_MODEL                     # scores
            + T * T * D_MODEL                     # context
            + T * D_MODEL * D_MODEL)              # output projection
    ffn = 2 * T * D_MODEL * D_FF                  # two FFN matmuls
    head = PROTO_DIM * CLASSES
    total_fwd = patch + attn + ffn + head
    recompute = attn + ffn                        # block is recomputed once
    return round((total_fwd + recompute) / total_fwd, 3)


def arm_transient(arm: str, regime: str) -> dict:
    if arm == "A0":
        return transient_frozen(regime)
    if arm in ("A1", "A2", "A3"):
        rank = {"A1": 0, "A2": 1, "A3": 2}[arm]
        return transient_postpool(regime, rank)
    return transient_full(regime)  # A4, A5


# --------------------------------------------------------------------------- #
# Persistent-state itemisation (Step 1) via the verified accounting module.
# Packed alignment (1 B) is the paper's headline basis; the 4-B-aligned variant
# is reported as a stated sensitivity in the report, not the CSV total.
# --------------------------------------------------------------------------- #
def persistent(arm: str) -> dict:
    r = account_arm(arm, budget_bytes=BUDGET, optimizer="adam",
                    weight_precision="fp32", replay_precision="int8",
                    alignment=1)
    return r


# --------------------------------------------------------------------------- #
# Emit CSV, report, figures
# --------------------------------------------------------------------------- #
def build_rows() -> list[dict]:
    rows = []
    for arm in ARMS:
        p = persistent(arm)
        for regime in REGIMES:
            t = arm_transient(arm, regime)
            rows.append({
                "arm": arm,
                "regime": regime,
                # persistent components (identical across regimes)
                "trainable_params": p["trainable_params"],
                "trainable_weight_bytes": p["trainable_weight_bytes"],
                "gradient_bytes": p["gradient_bytes"],
                "optimizer_bytes": p["optimizer_bytes"],
                "replay_items": p["prototypes"],
                "replay_value_bytes": p["replay_value_bytes"],
                "replay_label_bytes": p["label_bytes"],
                "replay_flag_bytes": p["valid_flag_bytes"],
                "replay_classid_bytes": p["class_id_bytes"],
                "fixed_metadata_bytes": p["fixed_metadata_bytes"],
                "padding_bytes": p["padding_bytes"],
                "persistent_total_bytes": p["total_bytes"],
                "persistent_fits_16384": p["fits"],
                # transient
                "transient_act_bytes_per_elem": t["act_bytes"],
                "transient_peak_elems": t["peak_elems"],
                "transient_peak_bytes": t["peak_bytes"],
                "transient_policy": t["policy"],
                "compute_multiplier": t["compute_multiplier"],
            })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def kib(n: int) -> str:
    return f"{n/1024:.1f}"


def write_report(rows: list[dict], path: Path) -> None:
    byarm = {}
    for r in rows:
        byarm.setdefault(r["arm"], {})[r["regime"]] = r
    lines: list[str] = []
    A = lines.append
    A("# E6 Two-Axis Memory Accounting (Task T1)\n")
    A("Analytical byte accounting. **No hardware runtime, SRAM, latency, or "
      "energy quantity is reported here** -- every number is arithmetic over "
      "the statically known 6,643-parameter graph, not an observed device "
      "quantity.\n")
    A("## Assumptions\n")
    A("- Architecture: T=66, d_model=16, d_ff=128, n_heads=1, key_dim=16, "
      "198-sample beat + 2 RR, 5 classes, 1 encoder block (PAPER1_STATUS.md).\n")
    A("- Optimizer Adam (2 moments); trainable weights FP32; replay INT8; "
      "packed 1-byte alignment (paper headline basis).\n")
    A("- Persistent axis from `budget_cl.memory.accounting.account_arm` "
      "(the paper's verified single source of truth).\n")
    A("- Transient axis, batch 1. R1: FP32, naive retain-all. R2: FP16 "
      "retain-all. R3: FP16 + checkpoint at attention/FFN granularity with "
      "recompute.\n")
    A("- Attention score tensor = n_heads*T^2 = 4,356 elements (n_heads=1); "
      "this reproduces the spec's ~17 KiB estimate exactly (4,356*4 = 17,424 B).\n")

    A("\n## Two-axis summary (persistent vs peak transient)\n")
    A("| Arm | Trainable | Replay items | Persistent (B) | Fits 16 KiB | "
      "Transient R1 (KiB) | R2 (KiB) | R3 (KiB) |")
    A("|---|---:|---:|---:|:--:|---:|---:|---:|")
    for arm in ARMS:
        r1 = byarm[arm]["R1"]; r2 = byarm[arm]["R2"]; r3 = byarm[arm]["R3"]
        A(f"| {arm} | {r1['trainable_params']} | {r1['replay_items']} | "
          f"{r1['persistent_total_bytes']:,} | "
          f"{'yes' if r1['persistent_fits_16384'] else 'NO'} | "
          f"{kib(r1['transient_peak_bytes'])} | {kib(r2['transient_peak_bytes'])} | "
          f"{kib(r3['transient_peak_bytes'])} |")

    A("\nThe asymmetry the 16 KiB budget excludes: post-pool arms (A1-A3) run "
      "no encoder pass during adaptation, so their peak transient is a fraction "
      "of a KiB; the raw-replay encoder arms (A4/A5) run the full forward+"
      "backward and pay tens of KiB. This is a direct consequence of the "
      "replay-point cliff -- replay location decides whether the encoder graph "
      "is live during adaptation at all.\n")

    A(f"\n**R3 compute multiplier (A4/A5): "
      f"{byarm['A4']['R3']['compute_multiplier']}x** forward FLOPs "
      "(recomputes the encoder block once during backward).\n")

    # Hand verification A1 and A4
    A("\n## Hand verification (must agree with the CSV to the byte)\n")
    _hand_verify_A1(A, byarm["A1"])
    _hand_verify_A4(A, byarm["A4"])

    # Fit check
    A("\n## Fit check (Step 4): persistent <= 16,384 B\n")
    A("| Arm | Persistent (B) | Headroom (B) | Verdict |")
    A("|---|---:|---:|:--:|")
    all_fit = True
    for arm in ARMS:
        r = byarm[arm]["R1"]
        head = BUDGET - r["persistent_total_bytes"]
        ok = r["persistent_fits_16384"]
        all_fit = all_fit and ok
        A(f"| {arm} | {r['persistent_total_bytes']:,} | {head:,} | "
          f"{'PASS' if ok else 'FAIL'} |")
    A(f"\n**All arms fit: {all_fit}.**\n")
    path.write_text("\n".join(lines) + "\n")
    return all_fit


def _hand_verify_A1(A, regimes: dict) -> None:
    r = regimes["R1"]
    A("### A1 (head-only, post-pool replay)\n")
    A("Persistent: head = 18*5+5 = 95 params. "
      f"weights {95*4}=380 + grads 380 + Adam 760 = {r['trainable_weight_bytes']+r['gradient_bytes']+r['optimizer_bytes']} B; "
      f"replay {r['replay_items']} items. CSV persistent total = "
      f"**{r['persistent_total_bytes']:,} B**.\n")
    A("Transient (R1): stored 18-d vector -> logits(5) -> probs(5); forward = "
      "18+5+5 = 28 elems, +18 backward temp = 46 elems * 4 B = **184 B** "
      f"(< 1 KiB). CSV = **{r['transient_peak_bytes']} B**. Encoder never runs.\n")


def _hand_verify_A4(A, regimes: dict) -> None:
    r1 = regimes["R1"]; r3 = regimes["R3"]
    A("\n### A4 (rank-1 encoder LoRA, raw replay)\n")
    A("Persistent: 95 head + 128 LoRA (4 attn projections * 2*16*1) = 223 "
      f"params. trainable-state {223*16}=3,568 B (16 B/param, Adam) + replay "
      f"{r1['replay_items']} raw records. CSV persistent total = "
      f"**{r1['persistent_total_bytes']:,} B**.\n")
    A("Transient R1 (retain-all, FP32): sum of forward activations = 39,582 "
      "elems. Dominant terms: ffn1 8,448 + relu 8,448 + scores 4,356 + "
      "attn_weights 4,356 + thirteen 1,056-tensors 13,728 + small 246. "
      f"39,582 * 4 B = **{39582*4:,} B** = {kib(39582*4)} KiB. CSV = "
      f"**{r1['transient_peak_bytes']:,} B**.\n")
    A("Transient R3 (checkpoint + recompute, FP16): retained residual-stream "
      "boundaries pos+add1+add2 (3*1,056) + head path (36) = 3,204 elems; FFN "
      "backward working set ln2(1,056)+relu(8,448)+g_relu(8,448) = 17,952 elems "
      "dominates attention (12,936). Peak = (3,204+17,952) = 21,156 elems * 2 B "
      f"= **{21156*2:,} B** = {kib(21156*2)} KiB. CSV = "
      f"**{r3['transient_peak_bytes']:,} B**.\n")
    if r3["transient_peak_bytes"] > 24 * 1024:
        A("\n> **ESCALATION (spec failure criterion 2):** A4 R3 peak transient "
          f"exceeds ~24 KiB ({kib(r3['transient_peak_bytes'])} KiB). The FFN "
          "hidden gradient is the driver. See BLOCKERS.md; the T3 "
          "aliasing-onto-inference-arena argument must be stated with this "
          "caveat (the marginal transient beyond the inference arena is the "
          "FFN-hidden gradient, ~8,448 elems).\n")


def make_figures(rows: list[dict]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "figures" / "scripts"))
    from _style import apply_style  # noqa
    apply_style()

    byarm = {}
    for r in rows:
        byarm.setdefault(r["arm"], {})[r["regime"]] = r
    out = ROOT / "figures" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # fig03: persistent-state stacked bar composition per arm
    comps = ["trainable_weight_bytes", "gradient_bytes", "optimizer_bytes",
             "replay_value_bytes", "replay_label_bytes", "replay_flag_bytes",
             "replay_classid_bytes", "fixed_metadata_bytes", "padding_bytes"]
    labels = ["weights", "gradients", "optimizer", "replay values",
              "labels", "valid flags", "class ids", "metadata", "padding"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bottoms = [0] * len(ARMS)
    for comp, lab in zip(comps, labels):
        vals = [byarm[a]["R1"][comp] for a in ARMS]
        ax.bar(ARMS, vals, bottom=bottoms, label=lab)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.axhline(BUDGET, ls="--", lw=1, color="k")
    ax.text(len(ARMS) - 0.5, BUDGET, " 16 KiB", va="bottom", ha="right", fontsize=8)
    ax.set_ylabel("Persistent state (bytes)")
    ax.set_title("Persistent-state composition per arm")
    ax.legend(fontsize=6, ncol=3, loc="lower center", frameon=False)
    p = out / "fig03_budget_composition.pdf"
    fig.savefig(p, metadata={"CreationDate": None}); plt.close(fig)
    paths.append(p)

    # fig13: two-axis Pareto (persistent x, peak transient y, log), per (arm,regime)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    markers = {"R1": "o", "R2": "s", "R3": "^"}
    for arm in ARMS:
        for regime in REGIMES:
            r = byarm[arm][regime]
            x = max(r["persistent_total_bytes"], 1)
            y = max(r["transient_peak_bytes"], 1)
            ax.scatter(x, y, marker=markers[regime], s=42)
            ax.annotate(f"{arm}/{regime}", (x, y), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.axvline(BUDGET, ls="--", lw=1, color="k")
    ax.axhline(BUDGET, ls="--", lw=1, color="0.5")
    ax.text(BUDGET, ax.get_ylim()[0], " 16 KiB persistent", rotation=90,
            va="bottom", ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("Persistent state (bytes)")
    ax.set_ylabel("Peak transient working set (bytes, log)")
    ax.set_title("Two-axis budget: persistent vs peak transient")
    p = out / "fig13_two_axis_pareto.pdf"
    fig.savefig(p, metadata={"CreationDate": None}); plt.close(fig)
    paths.append(p)
    return paths


def write_latex_table(rows: list[dict], path: Path) -> None:
    """Two-axis byte table for the embedded appendix (all arms, all regimes)."""
    byarm = {}
    for r in rows:
        byarm.setdefault(r["arm"], {})[r["regime"]] = r
    L = [
        r"% Auto-generated by scripts/gen_e6_accounting.py (T1). Do not hand-edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Two-axis analytical byte accounting (no hardware quantity is "
        r"observed). Persistent state is bounded at 16{,}384\,B; peak transient "
        r"is the forward+backward working set under three regimes: R1 naive FP32, "
        r"R2 batch-1 FP16, R3 batch-1 FP16 with block rematerialisation "
        r"($1.99\times$ forward FLOPs for the encoder arms). Source: "
        r"\texttt{results/e6\_two\_axis\_accounting.csv}.}",
        r"\label{tab:two_axis}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Arm & Trainable & Replay & Persistent (B) & Transient R1/R2/R3 (KiB) & Fits \\",
        r"\midrule",
    ]
    for arm in ARMS:
        r1 = byarm[arm]["R1"]; r2 = byarm[arm]["R2"]; r3 = byarm[arm]["R3"]
        L.append(
            f"{arm} & {r1['trainable_params']} & {r1['replay_items']} & "
            f"{r1['persistent_total_bytes']:,} & "
            f"{kib(r1['transient_peak_bytes'])} / {kib(r2['transient_peak_bytes'])} / "
            f"{kib(r3['transient_peak_bytes'])} & "
            f"{'yes' if r1['persistent_fits_16384'] else 'NO'} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    rows = build_rows()
    csv_path = ROOT / "results" / "e6_two_axis_accounting.csv"
    write_csv(rows, csv_path)
    write_latex_table(rows, ROOT / "manuscript" / "tables" / "table_two_axis.tex")
    report_path = ROOT / "results" / "e6_accounting_report.md"
    all_fit = write_report(rows, report_path)
    print(f"wrote {csv_path.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"wrote {report_path.relative_to(ROOT)}  all_fit={all_fit}")
    try:
        figs = make_figures(rows)
        for p in figs:
            print(f"wrote {p.relative_to(ROOT)}")
    except Exception as exc:  # figures are optional to the numeric outputs
        print(f"WARNING: figure generation skipped: {exc}", file=sys.stderr)
    return 0 if all_fit else 2


if __name__ == "__main__":
    raise SystemExit(main())
