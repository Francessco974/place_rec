"""Evaluate one run_eval output folder against TUM ground truth.

Sprint 1 (tiny baseline, no verifier): sweep the retrieval score of the rank-0 candidate.
  python3 eval_loops.py --run out/fr3_loh --gt data/fr3_loh/groundtruth.txt --plot
From v3 (verifier): sweep the inlier count of the verified loops.
  python3 eval_loops.py --run out/fr1_room --gt data/fr1_room/groundtruth.txt \
                        --sweep inliers --op-thr 25 --plot
--op-thr is the frozen operating point: choose it on the dev sequence only, then reuse it unchanged.
"""
import argparse
import csv
import json
import os

import numpy as np

from gt_labels import POS, NEG, load_tum_trajectory, poses_at, label_matrix, quat_to_R


def read_rows(path):
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def top1(cands):
    """Option A: one proposal per query = its rank-0 candidate."""
    return [r for r in cands if r["rank"] == 0]


def recall_at_k(cands, L, loop_queries, ks=(1, 5, 10)):
    """Fraction of loop queries with at least one POS among their top-k candidates."""
    if not loop_queries:
        return {f"R@{k}": None for k in ks}
    ranked = {}
    for r in cands:
        ranked.setdefault(int(r["query"]), []).append((int(r["rank"]), int(r["cand"])))
    out = {}
    for k in ks:
        hit = sum(any(L[q, c] == POS for rank, c in ranked.get(q, []) if rank < k)
                  for q in loop_queries)
        out[f"R@{k}"] = hit / len(loop_queries)
    return out


def counts(props, L, n_loop_queries, thr, key):
    """TP / FP / neutral among proposals with props[key] >= thr (at most one per query)."""
    tp = fp = neutral = 0
    for r in props:
        if r[key] < thr:
            continue
        lab = L[int(r["query"]), int(r["cand"])]
        if lab == POS: tp += 1
        elif lab == NEG: fp += 1
        else: neutral += 1          # IGNORE band or UNKNOWN (no GT): not judged
    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall = tp / n_loop_queries if n_loop_queries > 0 else None
    return dict(thr=thr, tp=tp, fp=fp, neutral=neutral, precision=precision, recall=recall)


def pr_curve(props, L, n_loop_queries, key):
    """Sweep the threshold on `key` from strict to permissive. No int cast: scores are floats."""
    thrs = sorted({r[key] for r in props} | {10**9}, reverse=True)
    return [counts(props, L, n_loop_queries, t, key) for t in thrs]


def summary(curve, n_loop_queries):
    if n_loop_queries == 0:                   # negative control: recall is undefined
        return {"R@100P": None, "AP": None}
    perfect = [p["recall"] for p in curve if p["fp"] == 0]
    r100 = max(perfect) if perfect else 0.0
    ap, prev_r = 0.0, 0.0                     # step-wise area under the PR curve
    for p in curve:                           # recall is non-decreasing along the sweep
        ap += (p["recall"] - prev_r) * p["precision"]
        prev_r = p["recall"]
    return {"R@100P": r100, "AP": ap}


def max_neg(props, L, key):
    """Highest sweep value among NEG proposals: the threshold needed for zero false loops."""
    neg = [r[key] for r in props if L[int(r["query"]), int(r["cand"])] == NEG]
    return max(neg) if neg else None


def pose_errors(loops, L, T, thr):
    """Error of accepted TRUE loops: E = (T_c_q^gt)^-1 * T_c_q^est."""
    errs = []
    for r in loops:
        i, j = int(r["query"]), int(r["cand"])
        if r["inliers"] < thr or L[i, j] != POS:
            continue
        T_est = np.eye(4)
        T_est[:3, :3] = quat_to_R(r["qx"], r["qy"], r["qz"], r["qw"])
        T_est[:3, 3] = (r["tx"], r["ty"], r["tz"])
        T_gt = np.linalg.inv(T[j]) @ T[i]     # query pose in candidate frame
        E = np.linalg.inv(T_gt) @ T_est
        ang = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))
        errs.append((np.linalg.norm(E[:3, 3]), ang))
    if not errs:
        return {}
    e = np.array(errs)
    return {"t_median_m": float(np.median(e[:, 0])), "t_max_m": float(e[:, 0].max()),
            "r_median_deg": float(np.median(e[:, 1])), "r_max_deg": float(e[:, 1].max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--sweep", choices=["score", "inliers"], default="score",
                    help="score: rank-0 candidates (no verifier); inliers: verified loops (v3)")
    ap.add_argument("--op-thr", type=float, default=None,
                    help="frozen operating point, chosen on the dev sequence")
    ap.add_argument("--t-excl", type=float, default=20.0)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    kf = read_rows(os.path.join(a.run, "keyframes.csv"))
    stamps = np.array([r["stamp"] for r in kf])
    gt_stamps, gt_T = load_tum_trajectory(a.gt)
    T, valid = poses_at(stamps, gt_stamps, gt_T)
    L = label_matrix(stamps, T, valid, t_excl=a.t_excl)
    loop_queries = [i for i in range(len(stamps)) if (L[i] == POS).any()]
    nq = len(loop_queries)

    loops = read_rows(os.path.join(a.run, "loops.csv"))
    cands = read_rows(os.path.join(a.run, "candidates.csv"))
    timing = np.array([[r["extract_ms"], r["query_ms"], r["verify_ms"]]
                       for r in read_rows(os.path.join(a.run, "timing.csv"))])

    # The recogniser's exclusion window (run_eval --exclude) must match the labels' t_excl.
    gaps = [stamps[int(r["query"])] - stamps[int(r["cand"])] for r in cands]
    if gaps and min(gaps) < a.t_excl - 1e-6:
        raise SystemExit(f"candidate only {min(gaps):.3f} s older than its query, but "
                         f"--t-excl is {a.t_excl}: run_eval --exclude and --t-excl differ")

    props = top1(cands) if a.sweep == "score" else loops
    if not props:
        raise SystemExit(f"no proposals to sweep with --sweep {a.sweep} (is loops.csv empty?)")

    curve = pr_curve(props, L, nq, a.sweep)
    res = {
        "keyframes": len(stamps), "with_gt": int(valid.sum()),
        "loop_queries": nq, "proposals": len(props), "sweep": a.sweep,
        **recall_at_k(cands, L, loop_queries),
        **summary(curve, nq),
        "max_neg": max_neg(props, L, a.sweep),
        "operating_point": (counts(props, L, nq, a.op_thr, a.sweep)
                            if a.op_thr is not None else None),
        "pose_error": (pose_errors(loops, L, T, a.op_thr)
                       if a.sweep == "inliers" and a.op_thr is not None else {}),
        "p95_ms": dict(zip(["extract", "query", "verify"],
                           np.percentile(timing, 95, axis=0).round(2).tolist())),
    }
    with open(os.path.join(a.run, "metrics.json"), "w") as f:
        json.dump({**res, "pr_curve": curve}, f, indent=2)
    print(json.dumps(res, indent=2))

    if a.plot and nq > 0:
        import matplotlib.pyplot as plt
        # where="pre": precision of point k covers (r_{k-1}, r_k], the same steps AP sums.
        plt.step([p["recall"] for p in curve], [p["precision"] for p in curve], where="pre")
        plt.xlabel("recall"); plt.ylabel("precision"); plt.ylim(0, 1.05); plt.grid(alpha=.3)
        plt.title(f"{os.path.basename(os.path.normpath(a.run))}  (sweep: {a.sweep})")
        plt.savefig(os.path.join(a.run, "pr.png"), dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()