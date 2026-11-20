"""按执行步数回放固定采样行号，统计冻结候选的训练来源。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    cfg = json.loads((ROOT / "exps/v480/config.json").read_text())
    loc = ROOT / json.loads((ROOT / "exps/v493/config.json").read_text())["strong"]
    parent = ROOT / cfg["parent_run"]
    report = {}
    for side, paths in cfg["metadata"].items():
        meta = [r for path in paths for r in json.loads((ROOT / path).read_text())]
        groups = np.array([r["src"] + "/" + r["lang"] for r in meta])
        counts = {
            name: np.zeros(len(meta), dtype=np.int64)
            for name in ("parent_final", "student_main", "student_aux")
        }
        for fold in cfg["folds"]:
            for seed in cfg["seeds"]:
                tag = f"B_s{seed}_f{fold}"
                with np.load(
                    parent / "training" / (tag + "_final") / "trace.npz"
                ) as trace:
                    steps = len(trace["loss"])
                with (
                    np.load(parent / f"plans/s{seed}_f{fold}_final.npz") as plan,
                    np.load(loc / (tag + "_auxiliary_plan.npz")) as aux,
                ):
                    chosen = plan[side + "_indices"]
                    assert (
                        len(chosen) >= steps and len(aux[side]) == cfg["student_steps"]
                    )
                    for name, ix in (
                        ("parent_final", chosen[:steps]),
                        ("student_main", chosen[: cfg["student_steps"]]),
                        ("student_aux", aux[side]),
                    ):
                        assert ix.min() >= 0 and ix.max() < len(meta)
                        counts[name] += np.bincount(ix.ravel(), minlength=len(meta))
        report[side] = {}
        for name, ct in counts.items():
            report[side][name] = {
                k: {
                    "exposures": int(ct[groups == k].sum()),
                    "unique_rows": int(np.sum((groups == k) & (ct > 0))),
                }
                for k in np.unique(groups)
                if ct[groups == k].any()
            }
    (HERE / "output/冻结候选训练来源.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["voice"], indent=2))


if __name__ == "__main__":
    main()
