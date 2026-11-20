"""把实际训练人物的末次梯度写入双塔，区分同性别群体与人物身份的监督。"""

from __future__ import annotations

import numpy as np
import torch
from media_common import CFG, HERE, REF, SIDES, read
from media_network import Matcher
from torch import Tensor
from torch.nn import functional as F

SPEC = read(HERE / "冻结前更新配置.json")


def views(model: Matcher) -> tuple[dict, dict]:
    params = {}
    bn = {}
    for side in SIDES:
        net = getattr(model, side).net
        for layer in (0, 1):
            for name in ("weight", "bias"):
                params[f"{side}{layer}{name}"] = getattr(net[layer], name)
        bn[side] = (
            net[1].running_mean.detach().clone(),
            torch.rsqrt(net[1].running_var.detach().clone() + net[1].eps),
        )
    return params, bn


def logits(
    params: dict, bn: dict, batch: list[Tensor], scale: Tensor, center: Tensor
) -> Tensor:
    embedding = []
    for side, x in zip(SIDES, batch, strict=True):
        y = F.linear(x, params[side + "0weight"], params[side + "0bias"])
        mean, inv = bn[side]
        embedding.append(
            F.normalize(
                F.relu(
                    (y - mean) * inv * params[side + "1weight"] + params[side + "1bias"]
                ),
                dim=-1,
            )
        )
    return scale * (embedding[0] @ embedding[1].T - center)


def balanced(s: Tensor) -> Tensor:
    diag = torch.eye(len(s), device=s.device, dtype=torch.bool)
    return 0.5 * (F.softplus(-s.diagonal()).mean() + F.softplus(s[~diag]).mean())


def pooled(s: Tensor, gender: Tensor) -> Tensor:
    # [身份，身份] 每行对全部同性别人物分配总量为 1 的正标签。
    match = gender[:, None] == gender[None, :]
    weight = match.to(s.dtype) / match.sum(dim=1, keepdim=True)
    n = len(s)
    return 0.5 * (
        (weight * F.softplus(-s)).sum() / n
        + ((1 - weight) * F.softplus(s)).sum() / (n * (n - 1))
    )


def rank_loss(s: Tensor, gender: Tensor) -> Tensor:
    # 每位脸端、声端人物等权，在同性别竞争者内比较本人和错配。
    negative = (gender[:, None] == gender[None, :]) & ~torch.eye(
        len(s), dtype=torch.bool, device=s.device
    )
    w = negative.to(s.dtype) / negative.sum(1, keepdim=True)
    pos = s.diagonal()[:, None]
    return 0.5 * (w * (F.softplus(s - pos) + F.softplus(s.T - pos))).sum(1).mean()


def trained_objective(
    params: dict,
    bn: dict,
    support: list[Tensor],
    query: list[Tensor],
    scale: Tensor,
    center: Tensor,
    method: str,
    lr: float,
    gender: Tensor,
) -> tuple[Tensor, list[Tensor]]:
    inner_rule, outer_rule = SPEC["objectives"][method]
    score = logits(params, bn, support, scale, center)
    inner = balanced(score)
    direction = inner - pooled(score, gender) if inner_rule == "identity" else inner
    gradients = torch.autograd.grad(
        direction, tuple(params.values()), create_graph=True, retain_graph=True
    )
    virtual = {
        k: v - lr * g for (k, v), g in zip(params.items(), gradients, strict=True)
    }

    def outer(p: dict) -> Tensor:
        value = logits(p, bn, query, scale, center)
        return rank_loss(value, gender) if outer_rule == "rank" else balanced(value)

    before, after = outer(params), outer(virtual)
    return 0.5 * (inner + after), [inner.detach(), before.detach(), after.detach()]


def norm(g: tuple[Tensor, ...]) -> Tensor:
    return torch.sqrt(torch.stack([v.square().sum() for v in g]).sum())


def endpoints(
    state: dict,
    temperature: dict,
    ext: list[Tensor],
    raw: list[Tensor],
    probes: dict,
    sex: Tensor,
    parent: str,
) -> tuple[dict, dict]:
    device = ext[0].device
    head = Matcher(ext[0].shape[-1], ext[1].shape[-1], state["proto"].shape[0], REF).to(
        device
    )
    head.load_state_dict(state)
    head.eval()
    params, bn = views(head)
    scale = torch.tensor(temperature["scale"], device=device)
    center = torch.tensor(temperature["center"], device=device)
    i = probes["wiki"][0]
    wrong = probes["wrong"][0]
    assert torch.equal(sex[i], sex[wrong]) and torch.all(i != wrong)
    batch = [x[i] for x in ext]
    gender = sex[i]
    score = logits(params, bn, batch, scale, center)
    full = balanced(score)
    expected = pooled(score, gender)
    permuted = balanced(logits(params, bn, [batch[0], ext[1][wrong]], scale, center))
    gradients = {
        k: torch.autograd.grad(l, tuple(params.values()), retain_graph=True)
        for k, l in (("full", full), ("pooled", expected), ("wrong", permuted))
    }
    gradients["identity"] = tuple(
        g - h for g, h in zip(gradients["full"], gradients["pooled"], strict=True)
    )
    norms = {k: norm(v) for k, v in gradients.items()}
    assert float(norms["identity"]) > 1e-8
    gradients["identity_norm"] = tuple(
        g * (norms["full"] / norms["identity"]) for g in gradients["identity"]
    )
    norms["identity_norm"] = norm(gradients["identity_norm"])
    np.testing.assert_allclose(
        float(norms["full"]), float(norms["identity_norm"]), rtol=2e-6
    )
    for g, h, i_grad in zip(
        gradients["full"], gradients["pooled"], gradients["identity"], strict=True
    ):
        torch.testing.assert_close(g, h + i_grad, atol=2e-6, rtol=2e-6)
    queries = [
        [x[indices[:, j]] for j, x in enumerate(raw)] for indices in probes["voxq"]
    ]
    result = {}
    measured = {}
    lr = CFG["lookahead_lr"]
    names = {
        f"{side}{layer}{key}": f"{side}.net.{layer}.{key}"
        for side in SIDES
        for layer in (0, 1)
        for key in ("weight", "bias")
    }
    with torch.no_grad():
        for name, change in SPEC["parents"][parent].items():
            virtual = (
                params
                if change == "keep"
                else {
                    k: v - lr * g
                    for (k, v), g in zip(params.items(), gradients[change], strict=True)
                }
            )
            kept = dict(state)
            for key, v in virtual.items():
                kept[names[key]] = v.detach().cpu()
            before = logits(virtual, bn, batch, scale, center)
            held = [logits(virtual, bn, q, scale, center) for q in queries]
            measured[name] = {
                "support_bce": float(balanced(before)),
                "support_rank": float(rank_loss(before, gender)),
                "probe_bce": [float(balanced(v)) for v in held],
                "probe_rank": [float(rank_loss(v, gender)) for v in held],
            }
            assert all(
                torch.equal(kept[k], v)
                for k, v in state.items()
                if k not in names.values()
            )
            result[name] = kept
    return result, {
        "scale": float(scale),
        "center": float(center),
        "gradient_norms": {k: float(v) for k, v in norms.items()},
        "objectives": measured,
    }
