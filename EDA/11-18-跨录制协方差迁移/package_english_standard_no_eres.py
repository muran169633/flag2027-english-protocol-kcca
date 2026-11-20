"""将冻结逐对评分器的 Evaluation 距离写成本地研究候选包。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from pair_score import PairScorer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_package(
    package: Path,
    scorers: dict[str, Callable[[np.ndarray, np.ndarray, str], np.ndarray]],
    *,
    resource_root: Path = ROOT,
) -> dict[str, Any]:
    cfg = json.loads((ROOT / "exps/v494/config.json").read_text())
    arrays = []
    manifests = []
    inputs = {}
    for side in ("face", "voice"):
        mp = resource_root / f"outputs/v379/features/eval_{side}s.json"
        records = json.loads(mp.read_text())
        manifests.append(records)
        inputs[str(mp.relative_to(resource_root))] = digest(mp)
        blocks = []
        for file, key in cfg[side + "_features"]:
            path = resource_root / file.replace("/train_", "/eval_")
            inputs[str(path.relative_to(resource_root))] = digest(path)
            with np.load(path) as z:
                blocks.append(z[key])
        arrays.append(np.column_stack(blocks))
        assert len(arrays[-1]) == len(records)
    buffer = BytesIO()
    cells = {}
    expected = {}
    consumed = []
    with ZipFile(buffer, "w") as archive:
        for track in ("no_gender", "gender"):
            folder = resource_root / "data/FLAG_2027/evaluation_phase" / track
            for lang, cond in (("English", "heard"), ("Bangla", "unheard")):
                official = folder / f"{lang}_valid.txt"
                inputs[str(official.relative_to(resource_root))] = digest(official)
                ids = []
                idx = [
                    i
                    for i, r in enumerate(manifests[0])
                    if (r["track"], r["lang"]) == (track, lang)
                ]
                for i, line in zip(idx, official.read_text().splitlines(), strict=True):
                    pair, voice, *face_parts = line.split()
                    face = (
                        "".join(face_parts)
                        if len(face_parts) == 1
                        else str(Path(face_parts[0]).joinpath(*face_parts[1:]))
                    )
                    for records, path in zip(manifests, (face, voice), strict=True):
                        r = records[i]
                        relative = (folder / path).relative_to(resource_root)
                        assert (
                            r["pair_id"],
                            r["track"],
                            r["lang"],
                            Path(r["path"]).parts[-len(relative.parts) :],
                        ) == (pair, track, lang, relative.parts)
                    ids.append(pair)
                assert len(set(ids)) == len(ids)
                consumed.extend(idx)
                distance = scorers[track](arrays[0][idx], arrays[1][idx], track)
                bounded = 0.5 + np.arctan(distance) / np.pi
                contents = "".join(
                    f"{k} {v:.15f}\n" for k, v in zip(ids, bounded, strict=True)
                )
                saved = np.array(
                    [float(line.split()[1]) for line in contents.splitlines()]
                )
                order = np.argsort(distance)
                assert np.isfinite(saved).all() and np.all((saved > 0) & (saved < 1))
                np.testing.assert_array_equal(
                    np.sign(np.diff(saved[order])), np.sign(np.diff(distance[order]))
                )
                member = f"{track}/sub_score_v4_{lang}_{cond}.txt"
                archive.writestr(ZipInfo(member), contents, compress_type=ZIP_DEFLATED)
                expected[member] = contents.encode()
                cells[member] = len(idx)
    assert sorted(consumed) == list(range(len(manifests[0]))) and len(consumed) == len(
        manifests[1]
    )
    blob = buffer.getvalue()
    if package.exists():
        if package.read_bytes() != blob:
            raise FileExistsError(package)
    else:
        with package.open("xb") as f:
            f.write(blob)
    with ZipFile(package) as archive:
        assert archive.testzip() is None and archive.namelist() == list(cells)
        for member, content in expected.items():
            assert archive.read(member) == content
    return {
        "package": str(
            package.relative_to(ROOT) if package.is_relative_to(ROOT) else package
        ),
        "sha256": digest(package),
        "pairs": cells,
        "input_sha256": inputs,
        "platform_submission": None,
    }


def main() -> None:
    weights = HERE / "output/纯英语评分参数.npz"
    standard_weights = HERE / "output/纯英语移除eres评分参数.npz"
    package = (
        ROOT
        / "evaluation/research_candidates/submission_evaluation_english_only_standard_no_eres.zip"
    )
    report = write_package(
        package,
        {
            "no_gender": PairScorer(standard_weights).distance,
            "gender": PairScorer(weights).distance,
        },
    )
    report.update(
        gender_parameters=digest(weights), standard_parameters=digest(standard_weights)
    )
    (HERE / "output/纯英语标准组移除eres验收.json").write_text(
        json.dumps(report, indent=2)
    )
    print(report["pairs"], report["sha256"])


if __name__ == "__main__":
    main()
