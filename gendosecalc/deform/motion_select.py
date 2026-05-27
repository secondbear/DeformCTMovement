"""Select N representative motion states from a dense timeseries.

Algorithm: **Partitioning Around Medoids (PAM)** — k-medoids clustering
on a 6-D feature vector ``[dx, dy, dz, rx·w, ry·w, rz·w]`` where
``w = rotation_weight_mm_per_deg`` converts degrees to a commensurate mm
scale.

Why k-medoids (not k-means)?
    - Medoids are real samples, so we can recover the original timestamp.
    - Repeated samples (motion dwell at same position) naturally increase the
      cost of *not* placing a medoid there, acting as implicit frequency
      weighting.

PAM implementation:
    A vendored O(k · N²) BUILD+SWAP PAM is used to avoid a mandatory
    ``scikit-learn-extra`` dependency.  For the typical prostate fraction
    with <50 000 samples this is fast (<1 s).  If ``scikit-learn-extra`` is
    installed and ``use_sklearn_extra=True`` is passed, it is preferred.

Output:
    A ``StateSelection`` with medoid indices into the original
    ``MotionSamples``, per-sample cluster assignments, per-cluster weights
    (counts), and the total within-cluster cost.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from gendosecalc.deform.models import MotionSamples, StateSelection

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _build_features(
    samples: MotionSamples,
    rotation_weight: float = 10.0,
) -> np.ndarray:
    """Return ``(N, 6)`` float64 feature matrix.

    Columns: ``[dx, dy, dz, rx·w, ry·w, rz·w]``.
    """
    t = samples.offsets_mm.astype(np.float64)       # (N, 3)
    r = samples.rotations_deg.astype(np.float64) * rotation_weight  # (N, 3)
    return np.hstack([t, r])  # (N, 6)


# ---------------------------------------------------------------------------
# Vendored PAM (Partitioning Around Medoids)
# ---------------------------------------------------------------------------

def _pam(
    X: np.ndarray,
    k: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Partitioning Around Medoids (BUILD + SWAP phases).

    Parameters:
        X: Feature matrix ``(N, d)``.
        k: Number of clusters.
        seed: Random seed for initial medoid selection.

    Returns:
        ``(medoid_indices, assignments, total_cost)`` where
        - ``medoid_indices``: ``(k,)`` int64 indices into X
        - ``assignments``: ``(N,)`` int64 — cluster for each sample
        - ``total_cost``: float — total within-cluster L2 distance
    """
    N = X.shape[0]
    if k >= N:
        medoid_indices = np.arange(N, dtype=np.int64)
        assignments = np.arange(N, dtype=np.int64)
        return medoid_indices, assignments, 0.0

    rng = np.random.default_rng(seed)

    # ----- BUILD phase: greedy initialisation -----
    # Start with the medoid that minimises total distance to all others
    # (a cheap approximation of the "most central" point)
    # We sample up to 5000 points for the distance matrix to keep BUILD fast
    sample_size = min(N, 5000)
    sample_idx = rng.choice(N, size=sample_size, replace=False)
    Xs = X[sample_idx]

    # Pairwise distances among the sample
    diff = Xs[:, np.newaxis, :] - Xs[np.newaxis, :, :]  # (S, S, d)
    D_sample = np.sqrt(np.sum(diff ** 2, axis=-1))  # (S, S)

    total_dist = D_sample.sum(axis=1)  # (S,)
    first_local = int(np.argmin(total_dist))
    medoids = [int(sample_idx[first_local])]

    for _ in range(1, k):
        # Each candidate adds the maximum reduction in total cost
        # Compute distance from each sample point to current medoid set
        D_to_medoids = _dist_to_medoids(X, X[medoids])  # (N,)
        # For each candidate, compute cost reduction
        best_gain = -np.inf
        best_cand = -1
        for cand in sample_idx:
            if cand in medoids:
                continue
            d_cand = np.linalg.norm(X - X[cand], axis=1)  # (N,)
            gain = np.sum(np.maximum(D_to_medoids - d_cand, 0))
            if gain > best_gain:
                best_gain = gain
                best_cand = int(cand)
        medoids.append(best_cand)

    medoids_arr = np.array(medoids, dtype=np.int64)

    # ----- SWAP phase -----
    improved = True
    while improved:
        improved = False
        # Distance matrix from every point to current medoids
        D = _dist_matrix(X, X[medoids_arr])  # (N, k)
        assignments = np.argmin(D, axis=1)   # (N,)
        cost = D[np.arange(N), assignments].sum()

        for m_idx in range(k):
            for cand in range(N):
                if cand in medoids_arr:
                    continue
                new_medoids = medoids_arr.copy()
                new_medoids[m_idx] = cand
                D_new = _dist_matrix(X, X[new_medoids])
                assign_new = np.argmin(D_new, axis=1)
                cost_new = D_new[np.arange(N), assign_new].sum()
                if cost_new < cost - 1e-9:
                    medoids_arr = new_medoids
                    cost = cost_new
                    assignments = assign_new
                    improved = True
                    break
            if improved:
                break

    return medoids_arr, assignments, float(cost)


def _dist_matrix(X: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    """Return ``(N, k)`` Euclidean distance matrix."""
    diff = X[:, np.newaxis, :] - medoids[np.newaxis, :, :]  # (N, k, d)
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _dist_to_medoids(X: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    """Return ``(N,)`` min distance from each point to the nearest medoid."""
    D = _dist_matrix(X, medoids)  # (N, k)
    return D.min(axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_representative_states(
    samples: MotionSamples,
    n_states: int,
    rotation_weight_mm_per_deg: float = 10.0,
    seed: int = 0,
    use_sklearn_extra: bool = False,
) -> StateSelection:
    """Select N representative motion states using weighted k-medoids.

    Parameters:
        samples: Full motion timeseries.
        n_states: Number of representative states to select.
        rotation_weight_mm_per_deg: Scale factor converting degrees to mm
            for the 6-D feature space (default 10 mm/deg).
        seed: Random seed for reproducibility.
        use_sklearn_extra: If True and ``sklearn_extra`` is installed, use
            its KMedoids implementation instead of the vendored PAM.

    Returns:
        A ``StateSelection`` with medoid indices, assignments, cluster weights,
        and total within-cluster cost.

    Raises:
        ValueError: If ``n_states < 1`` or ``n_states > len(samples)``.
    """
    n = len(samples)
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")
    if n_states > n:
        raise ValueError(
            f"n_states ({n_states}) cannot exceed number of samples ({n})"
        )

    X = _build_features(samples, rotation_weight=rotation_weight_mm_per_deg)

    if use_sklearn_extra:
        try:
            from sklearn_extra.cluster import KMedoids  # type: ignore[import]
            km = KMedoids(n_clusters=n_states, random_state=seed, method="pam")
            km.fit(X)
            medoid_indices = km.medoid_indices_.astype(np.int64)
            assignments = km.labels_.astype(np.int64)
            cost = float(km.inertia_)
            logger.info("Used sklearn_extra KMedoids for state selection")
        except ImportError:
            logger.warning(
                "sklearn_extra not found; falling back to vendored PAM"
            )
            medoid_indices, assignments, cost = _pam(X, k=n_states, seed=seed)
    else:
        medoid_indices, assignments, cost = _pam(X, k=n_states, seed=seed)

    # Compute cluster weights (sample counts)
    cluster_weights = np.bincount(assignments, minlength=n_states).astype(np.int64)

    # Per-cluster distance QA: mean and p95 from each medoid to its members
    cluster_mean_dist = np.zeros(n_states, dtype=np.float64)
    cluster_p95_dist = np.zeros(n_states, dtype=np.float64)
    for ci, med in enumerate(medoid_indices):
        member_mask = assignments == ci
        if member_mask.sum() > 0:
            dists = np.linalg.norm(X[member_mask] - X[med], axis=1)
            cluster_mean_dist[ci] = float(dists.mean())
            cluster_p95_dist[ci] = float(np.percentile(dists, 95))

    logger.info(
        "k-medoids converged: %d states, total cost=%.4f, "
        "cluster sizes [min=%d, max=%d, mean=%.1f]  "
        "within-cluster dist mean=%.2f p95=%.2f mm-eq",
        n_states,
        cost,
        int(cluster_weights.min()),
        int(cluster_weights.max()),
        float(cluster_weights.mean()),
        float(cluster_mean_dist.mean()),
        float(cluster_p95_dist.mean()),
    )

    return StateSelection(
        medoid_indices=medoid_indices,
        assignments=assignments,
        cluster_weights=cluster_weights,
        total_cost=cost,
        cluster_mean_dist_mm=cluster_mean_dist,
        cluster_p95_dist_mm=cluster_p95_dist,
    )
