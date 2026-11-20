"""在许可人物上拟合冻结评分器，保留独立单对二次型复核。"""

from __future__ import annotations

import numpy as np
import torch
from media_common import REF, SIDES
from media_eval_data import Media
from media_gaussian import align, embed, fit, score

State = dict[str, torch.Tensor]
KEYS = ("0.weight", "0.bias", "1.weight", "1.bias", "1.running_mean", "1.running_var")


def project(ds: Media, state: State) -> list[np.ndarray]:
    return [
        embed(x, state, s, REF) for s, x in zip(SIDES, (ds.face, ds.voice), strict=True)
    ]


@torch.inference_mode()
def precise_project(ds: Media, state: State) -> list[np.ndarray]:
    projected = []
    for side, arr in zip(SIDES, (ds.face, ds.voice), strict=True):
        p = {
            k: state[f"{side}.net.{k}"].to(REF["device"], dtype=torch.float64)
            for k in KEYS
        }
        chunks = []
        for start in range(0, len(arr), REF["batch_size"]):
            x = torch.as_tensor(
                arr[start : start + REF["batch_size"]],
                device=REF["device"],
                dtype=torch.float64,
            )
            z = torch.nn.functional.linear(x, p["0.weight"], p["0.bias"])
            z = (z - p["1.running_mean"]) / torch.sqrt(
                p["1.running_var"] + REF["bn_epsilon"]
            )
            z = torch.relu(z * p["1.weight"] + p["1.bias"])
            chunks.append(torch.nn.functional.normalize(z, dim=1).cpu().numpy())
        projected.append(np.concatenate(chunks))
    return projected


def scored(
    em: list, q: np.ndarray, mean: np.ndarray, delta: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    a, b = q[:, :2].T
    dim = em[0].shape[1]
    f, v = em[0] - mean[:dim], em[1] - mean[dim:]
    ff = np.einsum("ij,ij->i", f @ delta[:dim, :dim], f)
    vv = np.einsum("ij,ij->i", v @ delta[dim:, dim:], v)
    cross = f @ ((delta[:dim, dim:] + delta[dim:, :dim].T) / 2)
    ans = 0.5 * (ff[a] + vv[b]) + np.einsum("ij,ij->i", cross[a], v[b]) + bias
    ix = np.linspace(0, len(q) - 1, 7, dtype=int)
    np.testing.assert_allclose(
        ans[ix], score(em, q[ix], mean, delta, bias), atol=2e-10, rtol=0
    )
    return ans


def donor_groups(meta: list, allow: set) -> list:
    sides = [tuple(np.asarray(x) for x in rows) for rows in meta]
    names = sorted(set(sides[0][0]) & set(sides[1][0]) & allow)
    use = [
        n
        for n in names
        if all(len(np.unique(vids[who == n])) >= 2 for who, vids in sides)
    ]
    bank = []
    for who, video in sides:
        values = []
        for n in use:
            ix = np.flatnonzero(who == n)
            recordings, inv, counts = np.unique(
                video[ix], return_inverse=True, return_counts=True
            )
            values.append((ix, 1 / (len(recordings) * counts[inv])))
        bank.append(values)
    return bank


def make_parameters(
    full: State, track: str, ds: Media, groups: list, donor: list, dgroup: list
) -> tuple[State, list, list]:
    st = dict(full)
    if track == "standard":
        dims = REF["voice_block_dimensions"]
        number = [b[1] for b in REF["feature_blocks"]["voice_train"]].index("eres")
        sl = slice(sum(dims[:number]), sum(dims[: number + 1]))
        w = st["voice.net.0.weight"].clone()
        m = np.stack([ds.voice[ix, sl].mean(0) for _, ix in groups]).mean(0)
        st["voice.net.0.bias"] = st["voice.net.0.bias"] + w[:, sl] @ torch.from_numpy(m)
        w[:, sl] = 0
        st["voice.net.0.weight"] = w
    em = project(ds, st)
    gr = [
        (np.repeat(x, 2) if len(x) == 1 else x, np.repeat(y, 2) if len(y) == 1 else y)
        for x, y in groups
    ]
    stats = fit(em[0], em[1], gr, REF)
    delta, bias = -stats["precision_delta"], stats["constant"]
    if track == "same_gender":
        ext = [embed(x, st, s, REF) for s, x in zip(SIDES, donor, strict=True)]
        delta, bias = align(stats, em, ext, groups, dgroup)
    return st, [stats["mean"], delta, bias], em


def single_pairs(state: State, co: list, ds: Media, queries: np.ndarray) -> np.ndarray:
    # 每个原始输入以单条向量做独立双塔前向，重新计算完整的二次型。
    scores = []
    towers = [
        {k: state[f"{s}.net.{k}"].numpy().astype(float) for k in KEYS} for s in SIDES
    ]
    for a, b, _ in queries:
        pair = []
        for p, x in zip(towers, (ds.face[a], ds.voice[b]), strict=True):
            z = p["0.weight"] @ x.astype(float) + p["0.bias"]
            z = (z - p["1.running_mean"]) / np.sqrt(
                p["1.running_var"] + REF["bn_epsilon"]
            )
            z = np.maximum(0, z * p["1.weight"] + p["1.bias"])
            pair.append(z / max(float(np.linalg.norm(z)), 1e-12))
        centered = np.concatenate(pair) - co[0]
        scores.append(float(0.5 * centered @ co[1] @ centered + co[2]))
    return np.array(scores)
