"""v303：碎片合并闭集转导分配（探索性）。

python3 -m exps.v303.main --mode history   # 历史native20/new25同查询评估（冻结配置后一次）
python3 -m exps.v303.main --mode official  # 官方开发集无标签打包，不上传
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from scipy.stats import rankdata

from .closed_set import closed_merged, physical
from .competition import messages
from .speaker_probability import probability

LOG = logging.getLogger("v303")


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eer(y: np.ndarray, score: np.ndarray) -> float:
    """降序并列组ROC，在FPR=FNR处线性插值（与v300 independent_eer同定义）。"""
    order = np.argsort(-score, kind="stable")
    ranked = y[order]
    end = np.r_[np.where(np.diff(score[order]) != 0)[0], len(y) - 1]
    tpr = np.r_[0, np.cumsum(ranked)[end]] / np.sum(y)
    fpr = np.r_[0, np.cumsum(1 - ranked)[end]] / np.sum(1 - y)
    return float(np.interp(0, fpr + tpr - 1, fpr)) * 100


def eer_bruteforce(y: np.ndarray, score: np.ndarray) -> float:
    """独立实现：逐阈值显式计数FAR/FRR，取交叉线性插值。"""
    thresholds = np.r_[np.inf, np.unique(score)[::-1]]
    pos, neg = score[y == 1], score[y == 0]
    far = np.array([(neg >= t).mean() for t in thresholds])
    frr = np.array([(pos < t).mean() for t in thresholds])
    d = far - frr  # 从负到正单调
    i = int(np.flatnonzero(d >= 0)[0])
    if d[i] == 0 or i == 0:
        return float(far[i]) * 100
    a = -d[i - 1] / (d[i] - d[i - 1])
    return float(far[i - 1] + a * (far[i] - far[i - 1])) * 100


def final_score(parent: np.ndarray, m: np.ndarray, head: dict[str, Any], competition: bool) -> np.ndarray:
    x = np.c_[parent, m, m[:, 1] * np.maximum(1 - m[:, 0], 0)]
    if not competition:
        x[:, 2:] = 0
    return x[:, head["columns"]] @ np.asarray(head["coefficient"]) + head["intercept"]


def cell_ranks(values: np.ndarray, sizes: list[int]) -> np.ndarray:
    out, offset = np.empty_like(values), 0
    for n in sizes:
        out[offset : offset + n] = rankdata(values[offset : offset + n]) / n
        offset += n
    return out


def history_contexts(cfg: dict[str, Any]):
    for item in read(cfg["native_index"]):
        with np.load(item["queries"]) as q:
            cells = {k: q[k] for k in q.files}
        yield "native20", item["context"], Path(item["geometry"]), cells, "parent"
    with np.load(cfg["transfer_queries"]) as q:
        for c in cfg["contexts"]:
            cells = {k: q[c + "_" + k] for k in cfg["transfer_conditions"]}
            yield "new25", c, Path(cfg["transfer_root"]) / c / "storage.npz", cells, "coherent_parent"


def run_history(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    head, edge, final = (read(cfg[k]) for k in ("local_model", "edge_model", "final_model"))
    rows, audits, inputs = [], [], {}
    replayed: dict[str, int] = {}
    for source, context, geometry, cells, parent_key in history_contexts(cfg):
        inputs[str(geometry)] = digest(geometry)
        with np.load(geometry) as z:
            s = {k: z[k] for k in ("face", "edges", "mapped", "voice_probability", "voice_labels", parent_key, "face_probability")}
        ref_path = Path(cfg["reference_run"]) / "predictions" / source / context / "scores.npz"
        inputs[str(ref_path)] = digest(ref_path)
        with np.load(ref_path) as r:
            ref = {k: r[k] for k in ("labels", "baseline", "no_competition_current")}
        labels = np.concatenate([q[:, 2] for q in cells.values()]).astype(int)
        if not np.array_equal(labels, ref["labels"]):
            raise AssertionError("标签顺序与v300参照不同")
        t0 = time.monotonic()
        pv = probability(s, head, cfg)
        audit: dict[str, Any] = {"source": source, "context": context, "pv_seconds": time.monotonic() - t0}
        if replayed.get(source, 0) < cfg["replay_check_contexts"]:
            m = messages(s["face_probability"], pv, s["edges"], s["mapped"], edge, cfg)
            for name, comp, key in [("v219", True, "baseline"), ("v219_no_competition", False, "no_competition_current")]:
                err = float(np.max(abs(final_score(s[parent_key], m, final, comp) - ref[key])))
                audit[f"replay_{name}_max_error"] = err
                if err > cfg["numerical_tolerance"]:
                    raise AssertionError(f"PV/v219回放失败 {name}: {err}")
            replayed[source] = replayed.get(source, 0) + 1
        values = {"v219": ref["baseline"], "v219_no_competition": ref["no_competition_current"]}
        args = (cfg["positive_prior"], cfg["min_edges"], cfg["merge_threshold"])
        values["closed_merged"], info = closed_merged(s["face"], s["face_probability"], pv, s["edges"], s["mapped"], *args)
        audit["closed_merged"] = info
        rng = np.random.default_rng(303)
        worst = 0.0
        for i in rng.choice(len(s["mapped"]), cfg["physical_checks"], replace=False):
            d = physical(s["face"], s["face_probability"], pv, s["edges"], s["mapped"][i], *args)
            worst = max(worst, abs(d - values["closed_merged"][i]))
        audit["physical_closed_merged_max_error"] = worst
        if worst > 1e-8:
            raise AssertionError(f"闭集物理留边核验失败: {worst}")
        del s["face_probability"]
        sizes = [len(q) for q in cells.values()]
        values["rank_fusion"] = (cell_ranks(values["v219"], sizes) + cell_ranks(values["closed_merged"], sizes)) / 2
        if set(values) != set(cfg["variants"]):
            raise AssertionError("变体不完整")
        (out / "predictions").mkdir(exist_ok=True)
        np.savez_compressed(out / "predictions" / f"{source}_{context}.npz", labels=labels, **values)
        offset = 0
        for cell, q in cells.items():
            n = len(q)
            y = labels[offset : offset + n]
            for name, v in values.items():
                a, b = eer(y, v[offset : offset + n]), eer_bruteforce(y, v[offset : offset + n])
                if abs(a - b) > 1e-9:
                    raise AssertionError(f"EER两实现不符 {a} {b}")
                rows.append({"source": source, "context": context, "cell": cell, "model": name, "n": n, "eer_percent": a})
            offset += n
        audits.append(audit)
        LOG.info("完成 %s %s %s", source, context, audit)
    # 与v300已核验EER交叉比对
    v300 = {(r["source"], r["context"], r["cell"], r["model"]): r["eer_percent"] for r in read(Path(cfg["reference_run"]) / "metrics.json")}
    cross = 0.0
    for r in rows:
        key = {"v219": "baseline", "v219_no_competition": "no_competition_current"}.get(r["model"])
        if key:
            cross = max(cross, abs(v300[(r["source"], r["context"], r["cell"], key)] - r["eer_percent"]))
    if cross > 1e-9:
        raise AssertionError(f"与v300 EER不一致 {cross}")
    save(out / "metrics.json", rows)
    save(out / "audits.json", audits)
    save(out / "inputs.json", inputs)
    summary = summarize(rows, cfg)
    summary["v300_eer_cross_check_max_error"] = cross
    save(out / "summary.json", summary)
    return summary


def summarize(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    table: dict[str, dict[str, dict[str, float]]] = {}
    for src in ("native20", "new25"):
        table[src] = {}
        for name in cfg["variants"]:
            sel = [r for r in rows if r["source"] == src and r["model"] == name]
            cells = sorted({r["cell"] for r in sel})
            mean = {c: float(np.mean([r["eer_percent"] for r in sel if r["cell"] == c])) for c in cells}
            eng = [v for c, v in mean.items() if c.startswith("English")]
            mean["English_mean"] = float(np.mean(eng))
            mean["four_mean"] = float(np.mean([v for c, v in mean.items() if c in cells]))
            table[src][name] = mean
    decision = None
    for cand in ("closed_merged", "rank_fusion"):
        if all(table[s][cand]["English_mean"] < table[s]["v219"]["English_mean"] for s in table):
            decision = cand
            break
    return {"table": table, "decision": decision, "decision_rule": cfg["decision_rule"]}


def run_official(cfg: dict[str, Any], out: Path, history_run: Path) -> dict[str, Any]:
    hist = read(history_run / "summary.json")
    if read(history_run / "config.json") != cfg:
        raise AssertionError("历史评估后配置改变")
    if digest(cfg["official_v219_zip"]) != cfg["official_v219_zip_sha256"]:
        raise AssertionError("v219 ZIP改变")
    final = read(cfg["final_model"])
    with np.load(cfg["official_storage"]) as z:
        s = {k: z[k] for k in ("face", "edges", "mapped", "coherent_parent", "face_probability")}
    run = Path(cfg["official_v276_run"])
    with np.load(run / "new_geometry.npz") as z:
        pv, m, v219 = z["voice_probability"], z["messages"], z["similarities"]
    with np.load(run / "research_scores.npz") as z:
        cached = {k: z[k] for k in z.files}
    manifest = read(cfg["feature_manifest"])
    audit: dict[str, Any] = {}
    audit["v219_replay_max_error"] = float(np.max(abs(final_score(s["coherent_parent"], m, final, True) - v219)))
    # 与已提交ZIP逐行比对
    zip_err, offset = 0.0, 0
    with ZipFile(cfg["official_v219_zip"]) as zf:
        for cell, n in cfg["expected_cells"].items():
            track, lang = cell.split("/")
            name = f"{track}/sub_score_v4_{lang}_{'heard' if lang == 'English' else 'unheard'}.txt"
            lines = [x.split() for x in zf.read(name).decode().splitlines()]
            prefix = cell.replace("/", "_")
            if [x[0] for x in lines] != cached[prefix + "_pair_id"].tolist():
                raise AssertionError("ZIP行序不同")
            zip_err = max(zip_err, float(np.max(abs(-np.array([float(x[1]) for x in lines]) - v219[offset : offset + n]))))
            offset += n
    audit["v219_zip_max_error"] = zip_err
    if max(audit["v219_replay_max_error"], zip_err) > 1e-9:
        raise AssertionError(f"官方v219回放失败 {audit}")
    sizes = list(cfg["expected_cells"].values())
    scores = {
        "v219": v219,
        "v219_no_competition": final_score(s["coherent_parent"], m, final, False),
    }
    args = (cfg["positive_prior"], cfg["min_edges"], cfg["merge_threshold"])
    scores["closed_merged"], info = closed_merged(s["face"], s["face_probability"], pv, s["edges"], s["mapped"], *args)
    audit["closed_merged"] = info
    scores["rank_fusion"] = (cell_ranks(scores["v219"], sizes) + cell_ranks(scores["closed_merged"], sizes)) / 2
    rng = np.random.default_rng(303)
    worst = 0.0
    for i in rng.choice(len(s["mapped"]), cfg["physical_checks"], replace=False):
        d = physical(s["face"], s["face_probability"], pv, s["edges"], s["mapped"][i], *args)
        worst = max(worst, abs(d - scores["closed_merged"][i]))
    audit["physical_closed_merged_max_error"] = worst
    if worst > 1e-8:
        raise AssertionError("官方闭集物理核验失败")
    # 无标签排序变化
    audit["spearman_vs_v219"] = {}
    offset = 0
    for cell, n in cfg["expected_cells"].items():
        sl = slice(offset, offset + n)
        audit["spearman_vs_v219"][cell] = {
            k: float(np.corrcoef(rankdata(scores["v219"][sl]), rankdata(v[sl]))[0, 1]) for k, v in scores.items() if k != "v219"
        }
        offset += n
    np.savez_compressed(out / "official_scores.npz", **scores)
    packages = {}
    if hist["decision"]:
        packages["closed"] = package(cfg, out, "closed", scores[hist["decision"]], cached, manifest)
        packages["closed"]["variant"] = hist["decision"]
    audit["packages"] = packages
    audit["history_decision"] = hist["decision"]
    audit["uploaded"] = False
    audit["official_labels_used"] = False
    save(out / "verification.json", audit)
    return audit


def package(cfg, out, key, values, cached, manifest) -> dict[str, Any]:
    archive = out / f"{key}.zip"
    expected, offset = {}, 0
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=cfg["compression_level"]) as z:
        for cell, n in cfg["expected_cells"].items():
            track, lang = cell.split("/")
            prefix = cell.replace("/", "_")
            ids = cached[prefix + "_pair_id"].tolist()
            true_ids = [r["pair_id"] for r in manifest if r["track"] == track and r["language"] == lang]
            dist = -values[offset : offset + n]
            if ids != true_ids or len(ids) != n or len(set(ids)) != n or not np.isfinite(dist).all():
                raise AssertionError("行数/ID/数值无效 " + cell)
            name = f"{track}/sub_score_v4_{lang}_{'heard' if lang == 'English' else 'unheard'}.txt"
            info = ZipInfo(name, date_time=tuple(cfg["archive_timestamp"]))
            info.compress_type = ZIP_DEFLATED
            z.writestr(info, "".join(f"{i} {v:.{cfg['score_decimals']}f}\n" for i, v in zip(ids, dist, strict=True)))
            expected[name] = (ids, dist)
            offset += n
    with ZipFile(archive) as z:
        if set(z.namelist()) != set(expected) or z.testzip() is not None:
            raise AssertionError("ZIP文件名/CRC失败")
        for name, (ids, dist) in expected.items():
            lines = [x.split() for x in z.read(name).decode().splitlines()]
            if [x[0] for x in lines] != ids or not np.allclose([float(x[1]) for x in lines], dist, atol=1e-11, rtol=0):
                raise AssertionError("ZIP内容回读失败")
    dest = Path(cfg["deliveries"][key])
    with dest.open("xb") as w, archive.open("rb") as r:
        shutil.copyfileobj(r, w)
    if digest(dest) != digest(archive):
        raise AssertionError("交付哈希不符")
    return {"delivery": str(dest), "sha256": digest(dest), "rows": offset}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["history", "official"], required=True)
    p.add_argument("--history-run", type=Path)
    a = p.parse_args()
    cfg = read(Path(__file__).with_name("config.json"))
    run_id = datetime.now().strftime("%m-%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out = Path(cfg["output_root"]) / cfg["experiment_id"] / run_id
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log", encoding="utf-8")])
    save(out / "config.json", cfg)
    (out / "source").mkdir()
    hashes = {}
    for f in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(f, out / "source" / f.name)
        hashes[f.name] = digest(f)
    save(out / "source_hashes.json", hashes)
    save(out / "environment.json", {"executable": sys.executable, "python": sys.version, "mode": a.mode, "no_training": True})
    t = time.monotonic()
    try:
        if a.mode == "history":
            result = run_history(cfg, out)
        else:
            if a.history_run is None:
                raise ValueError("需要--history-run")
            for f, h in read(a.history_run / "source_hashes.json").items():
                if digest(Path(__file__).with_name(f)) != h:
                    raise AssertionError("历史评估后源码改变: " + f)
            result = run_official(cfg, out, a.history_run)
        result["seconds"] = time.monotonic() - t
        save(out / "result.json", result)
        print(out)
    except Exception as e:
        save(out / "failure.json", {"error": repr(e), "seconds": time.monotonic() - t})
        raise


if __name__ == "__main__":
    main()
