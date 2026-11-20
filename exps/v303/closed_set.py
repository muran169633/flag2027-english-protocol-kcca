"""碎片合并的闭集转导分配（v302失败后的探索性修正）。

v302把每个上游人脸簇当一个身份；同一人被拆成两个脸簇时，语音后验被大簇
“解释掉”，碎片簇中的真匹配被打成低分。本版：
1. 锚点簇：拥有 >= min_edges 条候选边的人脸簇；
2. 锚点簇之间用语音共现亲和度 Rn(k,l) = R(k,l)/sqrt(R(k,k)R(l,l))，
   R = OH^T PV[u,u] OH，Rn >= merge_threshold 的锚点并查集合并为同一身份；
3. 非锚点小簇人脸按PF质量软分配到合并后身份；全零则均匀；
4. 语音身份后验 ∝ exp(beta * W)，W(i) = sum_{j!=q} w(v_q,u_j) M(f_j, i)，
   分数 = log sum_i M(f_q,i) p_i - log sum_i (1-M(f_q,i)) p_i。
所有统计严格排除查询自身边（合并结构若因删边改变，则对该查询单独重算）。
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _union(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    _, ids = np.unique(roots, return_inverse=True)
    return ids


def structure(
    labels: np.ndarray, counts: np.ndarray, r: np.ndarray, min_edges: int, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (锚点簇索引, 锚点->身份编号)。"""
    anchors = np.flatnonzero(counts >= min_edges)
    sub = r[np.ix_(anchors, anchors)]
    d = np.sqrt(np.clip(np.diag(sub), 1e-300, None))
    rn = sub / np.outer(d, d)
    a, b = np.triu_indices(len(anchors), 1)
    keep = rn[a, b] >= threshold
    return anchors, _union(len(anchors), list(zip(a[keep].tolist(), b[keep].tolist())))


def membership(
    labels: np.ndarray, pf: np.ndarray, anchors: np.ndarray, ident: np.ndarray
) -> np.ndarray:
    n_id = int(ident.max()) + 1
    cluster_to_id = np.full(labels.max() + 1, -1)
    cluster_to_id[anchors] = ident
    face_id = cluster_to_id[labels]
    m = np.zeros((len(labels), n_id))
    anchored = face_id >= 0
    m[np.flatnonzero(anchored), face_id[anchored]] = 1
    loose = np.flatnonzero(~anchored)
    if len(loose):
        anchor_faces = np.flatnonzero(anchored)
        mass = pf[np.ix_(loose, anchor_faces)] @ m[anchor_faces]
        total = mass.sum(1, keepdims=True)
        m[loose] = np.where(total > 1e-12, mass / np.maximum(total, 1e-300), 1 / n_id)
    return m


def _score(evidence: np.ndarray, mq: np.ndarray, beta: float) -> np.ndarray:
    logits = beta * evidence
    with np.errstate(divide="ignore"):
        lm, lc = np.log(mq), np.log(np.clip(1 - mq, 0, None))
    return logsumexp(logits + lm, axis=-1) - logsumexp(logits + lc, axis=-1)


def closed_merged(
    face: np.ndarray,
    pf: np.ndarray,
    pv: np.ndarray,
    edges: np.ndarray,
    queries: np.ndarray,
    prior: float,
    min_edges: int,
    threshold: float,
) -> tuple[np.ndarray, dict]:
    _, labels = np.unique(face, return_inverse=True)
    k = int(labels.max()) + 1
    ce = labels[edges[:, 0]]
    oh = np.zeros((len(edges), k))
    oh[np.arange(len(edges)), ce] = 1
    pe = pv[np.ix_(edges[:, 1], edges[:, 1])]
    rows_oh = pe @ oh  # E x K：每条边对各簇的语音质量
    r = oh.T @ rows_oh
    counts = oh.sum(0)
    anchors, ident = structure(labels, counts, r, min_edges, threshold)
    m = membership(labels, pf, anchors, ident)
    beta = np.log1p(prior * (ident.max() + 1) / (1 - prior))
    me = m[edges[:, 0]]
    lookup = {tuple(e): i for i, e in enumerate(edges.tolist())}
    own = np.array([lookup[tuple(q)] for q in queries.tolist()])
    w = pv[queries[:, 1]][:, edges[:, 1]]
    ev = w @ me
    ev -= w[np.arange(len(queries)), own][:, None] * me[own]
    scores = _score(ev, m[queries[:, 0]], beta)
    # 删除查询自身边后合并结构是否改变：只需检查涉及该边所在簇的条目和锚点资格。
    recomputed = 0
    signature = (tuple(anchors.tolist()), tuple(ident.tolist()))
    for qi, j in enumerate(own):
        c = ce[j]
        r2 = r.copy()
        r2[c, :] -= rows_oh[j]
        r2[:, c] -= rows_oh[j]
        r2[c, c] += pe[j, j]
        counts2 = counts.copy()
        counts2[c] -= 1
        a2, i2 = structure(labels, counts2, r2, min_edges, threshold)
        if (tuple(a2.tolist()), tuple(i2.tolist())) == signature:
            continue
        recomputed += 1
        m2 = membership(labels, pf, a2, i2)
        beta2 = np.log1p(prior * (i2.max() + 1) / (1 - prior))
        me2 = m2[edges[:, 0]]
        e2 = w[qi] @ me2 - w[qi, j] * me2[j]
        scores[qi] = _score(e2[None], m2[queries[qi, 0]][None], beta2)[0]
    info = {
        "face_clusters": k,
        "anchors": len(anchors),
        "identities": int(ident.max()) + 1,
        "merged_pairs": int(len(anchors) - (ident.max() + 1)),
        "loose_faces": int((m.max(1) < 1).sum()),
        "queries_recomputed_for_loo_structure": recomputed,
    }
    return scores, info


def physical(
    face: np.ndarray,
    pf: np.ndarray,
    pv: np.ndarray,
    edges: np.ndarray,
    query: np.ndarray,
    prior: float,
    min_edges: int,
    threshold: float,
) -> float:
    """字面删除查询边后，从头循环重算全部结构（独立实现）。"""
    _, labels = np.unique(face, return_inverse=True)
    k = int(labels.max()) + 1
    ctx = edges[~np.all(edges == query, axis=1)]
    if len(ctx) != len(edges) - 1:
        raise AssertionError("查询边不唯一")
    ce = labels[ctx[:, 0]]
    counts = np.bincount(ce, minlength=k)
    r = np.zeros((k, k))
    for a in range(k):
        ia = ctx[ce == a, 1]
        if len(ia) == 0:
            continue
        for b in range(k):
            ib = ctx[ce == b, 1]
            if len(ib):
                r[a, b] = pv[np.ix_(ia, ib)].sum()
    anchors = [c for c in range(k) if counts[c] >= min_edges]
    pairs = []
    for x in range(len(anchors)):
        for y in range(x + 1, len(anchors)):
            a, b = anchors[x], anchors[y]
            if r[a, b] / np.sqrt(r[a, a] * r[b, b]) >= threshold:
                pairs.append((x, y))
    ident = _union(len(anchors), pairs)
    n_id = int(ident.max()) + 1
    cid = {a: int(ident[i]) for i, a in enumerate(anchors)}

    def memb(f: int) -> np.ndarray:
        out = np.zeros(n_id)
        if labels[f] in cid:
            out[cid[labels[f]]] = 1
            return out
        for g in range(len(labels)):
            if labels[g] in cid:
                out[cid[labels[g]]] += pf[f, g]
        return out / out.sum() if out.sum() > 1e-12 else np.full(n_id, 1 / n_id)

    cache: dict[int, np.ndarray] = {}
    ev = np.zeros(n_id)
    for f, u in ctx:
        if f not in cache:
            cache[f] = memb(int(f))
        ev += pv[query[1], u] * cache[f]
    beta = np.log(1 + prior * n_id / (1 - prior))
    mq = memb(int(query[0]))
    logit = beta * ev

    def lse(values: list[float]) -> float:
        if not values:
            return -np.inf
        top = max(values)
        return top + float(np.log(sum(np.exp(v - top) for v in values)))

    pos = [logit[i] + np.log(mq[i]) for i in range(n_id) if mq[i] > 0]
    neg = [logit[i] + np.log(1 - mq[i]) for i in range(n_id) if mq[i] < 1]
    return lse(pos) - lse(neg)
