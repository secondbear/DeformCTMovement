"""Tests for motion_select.py — k-medoids state selection."""

from __future__ import annotations

import numpy as np
import pytest

from gendosecalc.deform.models import MotionSamples, StateSelection
from gendosecalc.deform.motion_select import select_representative_states


def _make_samples(offsets: np.ndarray) -> MotionSamples:
    n = len(offsets)
    return MotionSamples(
        timestamps_ms=np.arange(n, dtype=np.int64) * 140,
        offsets_mm=np.asarray(offsets, dtype=np.float32),
        rotations_deg=np.zeros((n, 3), dtype=np.float32),
        has_rotations=False,
    )


class TestSelectRepresentativeStates:
    def test_basic_3_clusters(self) -> None:
        # 9 points in 3 clear clusters
        offsets = (
            [[0, 0, 0]] * 3
            + [[10, 0, 0]] * 3
            + [[0, 10, 0]] * 3
        )
        samples = _make_samples(offsets)
        sel = select_representative_states(samples, n_states=3, seed=0)
        assert isinstance(sel, StateSelection)
        assert len(sel.medoid_indices) == 3
        # Each cluster should have weight 3
        assert sorted(sel.cluster_weights.tolist()) == [3, 3, 3]

    def test_weight_conservation(self) -> None:
        """Sum of cluster weights must equal total number of samples."""
        n = 50
        offsets = np.random.default_rng(42).normal(size=(n, 3))
        samples = _make_samples(offsets)
        sel = select_representative_states(samples, n_states=5, seed=0)
        assert int(sel.cluster_weights.sum()) == n

    def test_medoids_are_real_samples(self) -> None:
        n = 30
        offsets = np.random.default_rng(7).normal(size=(n, 3))
        samples = _make_samples(offsets)
        sel = select_representative_states(samples, n_states=4, seed=0)
        assert all(0 <= int(idx) < n for idx in sel.medoid_indices)

    def test_n_states_equals_n(self) -> None:
        offsets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        samples = _make_samples(offsets)
        sel = select_representative_states(samples, n_states=3, seed=0)
        assert len(sel.medoid_indices) == 3

    def test_n_states_too_large_raises(self) -> None:
        samples = _make_samples([[1, 0, 0]])
        with pytest.raises(ValueError):
            select_representative_states(samples, n_states=5, seed=0)

    def test_n_states_zero_raises(self) -> None:
        samples = _make_samples([[1, 0, 0], [2, 0, 0]])
        with pytest.raises(ValueError):
            select_representative_states(samples, n_states=0, seed=0)

    def test_repeated_samples_cluster_weight(self) -> None:
        """Ten identical points should all end up in the same cluster."""
        offsets = [[5.0, 5.0, 5.0]] * 10 + [[100.0, 0.0, 0.0]] * 2
        samples = _make_samples(offsets)
        sel = select_representative_states(samples, n_states=2, seed=0)
        max_weight = int(sel.cluster_weights.max())
        assert max_weight == 10

    def test_reproducible_with_seed(self) -> None:
        n = 40
        offsets = np.random.default_rng(99).normal(size=(n, 3))
        samples = _make_samples(offsets)
        sel1 = select_representative_states(samples, n_states=5, seed=42)
        sel2 = select_representative_states(samples, n_states=5, seed=42)
        np.testing.assert_array_equal(sel1.medoid_indices, sel2.medoid_indices)

    def test_total_cost_non_negative(self) -> None:
        offsets = np.random.default_rng(1).normal(size=(20, 3))
        samples = _make_samples(offsets)
        sel = select_representative_states(samples, n_states=3, seed=0)
        assert sel.total_cost >= 0.0
