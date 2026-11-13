"""Partition strategies for block diffusion.

BD3-LM decomposes a sequence into *adjacent* windows:

    log p(x) = sum_b log p(x_b | x_<b),   x_b = positions [(b-1)L'+1, .., bL']

Its correctness argument is that the rate matrix acts only on the current block
while the prefix is a fixed condition, so every fact of the masked diffusion
framework holds block by block. That argument never mentions adjacency. The
chain rule is exact for *any* ordered partition, so the left-hand side does not
depend on the partition at all and every difference between partitions lives
entirely in the slack of the within-block diffusion bound.

The working hypothesis is mechanistic: positions that stay strongly dependent
given the prefix are the ones masked diffusion handles worst inside a block,
because it must commit to them under a factorized posterior. Splitting such
positions across blocks hands their dependence to the exact chain rule instead.
So partitions with low within-block coupling mass should give tighter bounds.

A partition is a list of index arrays, in generation order.

Strategies
----------
single         one block                     == MDLM
autoregressive blocks of size 1              == left-to-right autoregression
contiguous     adjacent windows              == BD3-LM exactly
random         size-matched random partition  control for "any non-adjacency"
burial         buried core first, then surface
sse            grouped by secondary structure element
anticluster    greedy coloring of the contact graph, minimising within-block
               contact mass                  predicted best by the hypothesis
cluster        spectral clustering of the contact graph, maximising within-block
               contact mass                  the opposite extreme

`cluster` matters as much as `anticluster`. Together they bracket the range of
coupling mass, which turns the main result from a bar chart into a regression
of bound slack on coupling mass, with contiguous and random landing in between.
"""

from __future__ import annotations

import numpy as np

STRATEGIES = (
    "single", "autoregressive", "contiguous", "random",
    "burial", "sse", "anticluster", "cluster",
)

Partition = list[np.ndarray]


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

def coupling_mass(partition: Partition, contacts: np.ndarray) -> float:
    """Fraction of true contacts that fall *inside* a block.

    This is the x-axis of the mechanism plot. 0 means every contact is handled
    by the exact chain rule across blocks; 1 means every contact sits inside a
    block and is approximated by the factorized posterior.
    """
    total = contacts.sum() / 2.0
    if total == 0:
        return 0.0
    inside = 0.0
    for block in partition:
        sub = contacts[np.ix_(block, block)]
        inside += sub.sum() / 2.0
    return float(inside / total)


def block_size_stats(partition: Partition) -> dict:
    sizes = np.array([len(b) for b in partition])
    return {
        "n_blocks": int(len(sizes)),
        "mean_block_size": float(sizes.mean()),
        "max_block_size": int(sizes.max()),
        "min_block_size": int(sizes.min()),
    }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _target_sizes(length: int, block_size: int) -> list[int]:
    n = max(1, int(np.ceil(length / block_size)))
    base, rem = divmod(length, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 50):
    """Minimal k-means so sklearn stays optional."""
    idx = rng.choice(len(x), size=k, replace=False)
    centers = x[idx].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                centers[j] = x[m].mean(0)
    return labels


def _from_labels(labels: np.ndarray, order: np.ndarray | None = None) -> Partition:
    uniq = np.unique(labels)
    if order is not None:
        uniq = uniq[np.argsort([order[u] for u in uniq])]
    return [np.where(labels == u)[0] for u in uniq if (labels == u).any()]


def _rebalance(partition: Partition, max_size: int) -> Partition:
    """Split any block larger than max_size so group sizes stay comparable.

    Without this, `cluster` produces one giant block and the comparison turns
    into a block-size comparison rather than a topology comparison.
    """
    out = []
    for block in partition:
        if len(block) <= max_size:
            out.append(block)
        else:
            n = int(np.ceil(len(block) / max_size))
            out.extend(np.array_split(block, n))
    return [b for b in out if len(b) > 0]


# --------------------------------------------------------------------------
# strategies
# --------------------------------------------------------------------------

def make_partition(
    strategy: str,
    record,
    block_size: int = 16,
    seed: int = 0,
    order_by: str = "natural",
) -> Partition:
    """Build a partition for one protein.

    `record` is a ProteinRecord. `block_size` is the target block size and is
    held fixed across strategies so that every comparison is size-matched.
    `order_by` controls generation order: natural, burial (buried first),
    size (largest first), or random.
    """
    L = record.length
    rng = np.random.default_rng(seed)
    idx = np.arange(L)

    if strategy == "single":
        part = [idx]
    elif strategy == "autoregressive":
        part = [np.array([i]) for i in range(L)]
    elif strategy == "contiguous":
        part = [np.array(b) for b in np.array_split(idx, len(_target_sizes(L, block_size)))]
    elif strategy == "random":
        perm = rng.permutation(L)
        part = [np.sort(b) for b in np.array_split(perm, len(_target_sizes(L, block_size)))]
    elif strategy == "burial":
        part = _burial_partition(record, block_size)
    elif strategy == "sse":
        part = _sse_partition(record, block_size)
    elif strategy == "anticluster":
        part = _anticluster_partition(record, block_size, rng)
    elif strategy == "cluster":
        part = _cluster_partition(record, block_size, rng)
    else:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")

    if strategy not in ("single", "autoregressive"):
        part = _rebalance(part, max_size=2 * block_size)
    return _apply_order(part, record, order_by, rng)


def _apply_order(part: Partition, record, order_by: str, rng) -> Partition:
    if order_by == "natural" or len(part) <= 1:
        return part
    if order_by == "burial":
        key = [record.rsa[b].mean() for b in part]       # buried (low RSA) first
        return [part[i] for i in np.argsort(key)]
    if order_by == "size":
        return [part[i] for i in np.argsort([-len(b) for b in part])]
    if order_by == "random":
        return [part[i] for i in rng.permutation(len(part))]
    raise ValueError(f"unknown order_by {order_by!r}")


def _burial_partition(record, block_size: int) -> Partition:
    """Buried core first, then surface, each chopped into blocks.

    This is the drug-discovery-shaped partition: the positions that determine
    the fold are committed before the ones that decorate it.
    """
    buried = np.where(record.buried())[0]
    exposed = np.where(~record.buried())[0]
    part = []
    for group in (buried, exposed):
        if len(group) == 0:
            continue
        n = max(1, int(np.ceil(len(group) / block_size)))
        part.extend(np.array_split(group, n))
    return part


def _sse_partition(record, block_size: int) -> Partition:
    """One block per contiguous secondary structure element.

    Helices and strands become units; loops are grouped together. Elements
    longer than the target size are split so the comparison stays size-matched.
    """
    part, start = [], 0
    sse = record.sse
    for i in range(1, len(sse) + 1):
        if i == len(sse) or sse[i] != sse[start]:
            part.append(np.arange(start, i))
            start = i
    return part


def _contact_graph(record) -> np.ndarray:
    return record.contacts.astype(float)


def _anticluster_partition(record, block_size: int, rng) -> Partition:
    """Greedy coloring that minimises within-block contact mass.

    Residues are processed in decreasing contact degree, the hardest to place
    first, and each is assigned to the block with the fewest contacts to its
    current members, subject to a size cap. This is a heuristic for balanced
    k-way max-cut, which is NP-hard; we do not claim optimality and the paper
    should say so.
    """
    A = _contact_graph(record)
    L = record.length
    n_blocks = max(1, int(np.ceil(L / block_size)))
    cap = int(np.ceil(L / n_blocks))

    blocks: list[list[int]] = [[] for _ in range(n_blocks)]
    order = np.argsort(-A.sum(1))
    for i in order:
        best, best_cost = None, None
        for b, members in enumerate(blocks):
            if len(members) >= cap:
                continue
            cost = A[i, members].sum() if members else 0.0
            # tie-break toward emptier blocks for balance
            cost += 1e-6 * len(members)
            if best_cost is None or cost < best_cost:
                best, best_cost = b, cost
        if best is None:
            best = int(np.argmin([len(m) for m in blocks]))
        blocks[best].append(int(i))
    return [np.sort(np.array(b)) for b in blocks if b]


def _cluster_partition(record, block_size: int, rng) -> Partition:
    """Spectral clustering of the contact graph: spatially compact blocks.

    This deliberately *maximises* within-block coupling and should, by the
    hypothesis, give the loosest bound. It is the control that makes the
    mechanism claim falsifiable.
    """
    A = _contact_graph(record)
    L = record.length
    n_blocks = max(1, int(np.ceil(L / block_size)))
    if n_blocks == 1:
        return [np.arange(L)]

    deg = A.sum(1)
    d_inv = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    lap = np.eye(L) - d_inv @ A @ d_inv
    vals, vecs = np.linalg.eigh(lap)
    embedding = vecs[:, 1 : n_blocks + 1]
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(norm, 1e-8)

    labels = _kmeans(embedding, n_blocks, rng)
    return _from_labels(labels)
