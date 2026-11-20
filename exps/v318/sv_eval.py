"""v318：有标签说话人验证对比 crop6/full/multi（全部语音对；按语言、同性别、至少一方>6秒分组）。
python -m exps.v318.sv_eval outputs/v318/<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from exps.v303.main import eer

CFG88 = json.loads(Path("exps/v088/config.json").read_text())
MODES = ("crop6", "full", "multi")


def main():
    run = Path(sys.argv[1])
    src = {s["name"]: s for s in CFG88["sources"]}
    report = []
    for name in ("official_train", "historical_expanded"):
        z = np.load(run / f"{name}.npz")
        recs = json.loads(Path(src[name]["manifest"]).read_text())
        lk = z["lookup"]
        first = {}
        for i, j in enumerate(lk):
            if j >= 0 and j not in first:
                first[int(j)] = recs[i]
        n = len(first)
        ident = np.array([first[j]["identity"] for j in range(n)])
        lang = np.array([first[j]["language"] for j in range(n)])
        gender = np.array([str(first[j].get("gender", "?")) for j in range(n)])
        video = np.array([str(first[j].get("voice_video")) for j in range(n)])
        long_ = z["duration"] > 6
        iu = np.triu_indices(n, 1)
        y = ident[iu[0]] == ident[iu[1]]
        groups = {}
        for L in np.unique(lang):
            base = (lang[iu[0]] == L) & (lang[iu[1]] == L)
            if name == "historical_expanded":
                base &= ~(y & (video[iu[0]] == video[iu[1]]))  # 目标对要求跨视频
            groups[f"{L}"] = base
            if (gender != "?").all():
                groups[f"{L}/same_gender"] = base & (gender[iu[0]] == gender[iu[1]])
            groups[f"{L}/any_long"] = base & (long_[iu[0]] | long_[iu[1]])
        for g, mask in groups.items():
            row = {"source": name, "group": g, "targets": int((y & mask).sum()), "nontargets": int((~y & mask).sum())}
            for m in MODES:
                E = z[m]
                s = np.sum(E[iu[0][mask]] * E[iu[1][mask]], 1)
                row[m] = eer(y[mask].astype(int), s)
            report.append(row)
            print(row, flush=True)
        print(name, "voices", n, "over6s", float(long_.mean()), {L: float(long_[lang == L].mean()) for L in np.unique(lang)})
    (run / "sv_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
