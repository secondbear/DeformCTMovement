"""Tests for bone_mask — HU segmentation and tissue weight map."""

from __future__ import annotations

import numpy as np
import pytest

from gendosecalc.deform.bone_mask import compute_bone_mask, compute_tissue_weight
from gendosecalc.deform.models import DeformationConfig


class TestComputeBoneMask:
    """Bone mask via HU thresholding."""

    def test_default_threshold(self, ct_array):
        mask = compute_bone_mask(ct_array)
        # Bone insert at 800 HU should be True; soft tissue at 0 HU should be False
        assert mask[2, 2, 2]  # bone region
        assert not mask[8, 8, 8]  # soft tissue centre

    def test_custom_threshold(self, ct_array):
        config = DeformationConfig(bone_threshold_hu=500.0)
        mask = compute_bone_mask(ct_array, config)
        assert mask[2, 2, 2]  # 800 > 500
        assert not mask[8, 8, 8]  # 0 < 500

    def test_all_bone_with_low_threshold(self, ct_array):
        config = DeformationConfig(bone_threshold_hu=-2000.0)
        mask = compute_bone_mask(ct_array, config)
        assert mask.all()

    def test_no_bone_with_high_threshold(self, ct_array):
        config = DeformationConfig(bone_threshold_hu=1000.0)
        mask = compute_bone_mask(ct_array, config)
        assert not mask.any()

    def test_shape_matches_input(self, ct_array):
        mask = compute_bone_mask(ct_array)
        assert mask.shape == ct_array.shape
        assert mask.dtype == bool


class TestComputeTissueWeight:
    """Tissue weight map: 0 in bone, 1 in soft tissue, smooth transition."""

    def test_bone_region_near_zero(self, ct_array, ct_spacing):
        mask = compute_bone_mask(ct_array)
        weight = compute_tissue_weight(mask, ct_spacing)
        # All actual bone voxels must be exactly 0 regardless of bone thickness
        assert weight[2, 2, 2] == 0.0

    def test_soft_tissue_region_near_one(self, ct_array, ct_spacing):
        mask = compute_bone_mask(ct_array)
        weight = compute_tissue_weight(mask, ct_spacing)
        # Deep inside soft tissue (centre), weight should be near 1
        assert weight[8, 8, 8] > 0.9

    def test_weight_bounded_zero_one(self, ct_array, ct_spacing):
        mask = compute_bone_mask(ct_array)
        weight = compute_tissue_weight(mask, ct_spacing)
        assert weight.min() >= 0.0
        assert weight.max() <= 1.0

    def test_smooth_transition(self, ct_array, ct_spacing):
        """Soft tissue voxels just outside bone should have intermediate weight."""
        mask = compute_bone_mask(ct_array)
        config = DeformationConfig(transition_width_mm=4.0)
        weight = compute_tissue_weight(mask, ct_spacing, config)
        # Bone voxels are always exactly 0 after the hard-set
        assert weight[4, 4, 4] == 0.0
        # Soft tissue 1 voxel outside the bone block should be intermediate
        boundary_values = weight[5, 5, 5]
        assert 0.1 < boundary_values < 0.95

    def test_shape_and_dtype(self, ct_array, ct_spacing):
        mask = compute_bone_mask(ct_array)
        weight = compute_tissue_weight(mask, ct_spacing)
        assert weight.shape == ct_array.shape
        assert weight.dtype == np.float32

    def test_uniform_bone_gives_zero_weight(self):
        """All-bone CT → weight ≈ 0 everywhere."""
        arr = np.full((10, 10, 10), 800, dtype=np.int16)
        spacing = np.array([1.0, 1.0, 1.0])
        mask = compute_bone_mask(arr)
        weight = compute_tissue_weight(mask, spacing)
        assert weight.max() < 0.05

    def test_uniform_tissue_gives_one_weight(self):
        """All-tissue CT → weight ≈ 1 everywhere."""
        arr = np.full((10, 10, 10), 0, dtype=np.int16)
        spacing = np.array([1.0, 1.0, 1.0])
        mask = compute_bone_mask(arr)
        weight = compute_tissue_weight(mask, spacing)
        assert weight.min() > 0.95
