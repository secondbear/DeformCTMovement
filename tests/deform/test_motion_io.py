"""Tests for motion_io.py — Synchrony XML and CSV ingestion."""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from gendosecalc.deform.motion_io import load_motion_csv, load_synchrony_xml
from gendosecalc.deform.models import MotionSamples

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8" standalone="no"?>
    <MotionData>
     <Version>3.0</Version>
     <DeliverySegment>
      <SegmentIndex>0</SegmentIndex>
      <RadiationResults>
       <ResultsIndex>0</ResultsIndex>
       <ModelData>
        <PotentialDifferenceTolerance>3.0</PotentialDifferenceTolerance>
        <MeasuredDifferenceTolerance>4.0</MeasuredDifferenceTolerance>
        <DataPoint>
         <Timestamp>1000000</Timestamp>
         <PotentialDiff>0.5</PotentialDiff>
         <TargetOffset><X>1.0</X><Y>-2.0</Y><Z>0.5</Z></TargetOffset>
         <MeasuredDifference>NaN</MeasuredDifference>
        </DataPoint>
        <DataPoint>
         <Timestamp>1000140</Timestamp>
         <PotentialDiff>0.8</PotentialDiff>
         <TargetOffset><X>2.0</X><Y>-1.5</Y><Z>0.3</Z></TargetOffset>
         <MeasuredDifference>NaN</MeasuredDifference>
        </DataPoint>
        <DataPoint>
         <Timestamp>1000280</Timestamp>
         <PotentialDiff>5.0</PotentialDiff>
         <TargetOffset><X>99.0</X><Y>99.0</Y><Z>99.0</Z></TargetOffset>
         <MeasuredDifference>NaN</MeasuredDifference>
        </DataPoint>
       </ModelData>
      </RadiationResults>
     </DeliverySegment>
    </MotionData>
""")

_MINIMAL_CSV_WITH_ROT = textwrap.dedent("""\
    timestamp_ms,dx,dy,dz,rx,ry,rz
    1000000,1.0,-2.0,0.5,0.1,-0.2,0.3
    1000140,2.0,-1.5,0.3,0.0,0.0,0.0
""")

_MINIMAL_CSV_NO_ROT = textwrap.dedent("""\
    timestamp_ms,dx,dy,dz
    1000000,1.0,-2.0,0.5
    1000140,2.0,-1.5,0.3
""")


@pytest.fixture()
def xml_file(tmp_path: Path) -> Path:
    f = tmp_path / "MotionData.xml"
    f.write_text(_MINIMAL_XML, encoding="utf-8")
    return f


@pytest.fixture()
def csv_with_rot(tmp_path: Path) -> Path:
    f = tmp_path / "motion.csv"
    f.write_text(_MINIMAL_CSV_WITH_ROT, encoding="utf-8")
    return f


@pytest.fixture()
def csv_no_rot(tmp_path: Path) -> Path:
    f = tmp_path / "motion_norot.csv"
    f.write_text(_MINIMAL_CSV_NO_ROT, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# XML tests
# ---------------------------------------------------------------------------

class TestLoadSynchronyXml:
    def test_loads_valid_samples(self, xml_file: Path) -> None:
        samples = load_synchrony_xml(xml_file, tolerance_mode="keep")
        assert isinstance(samples, MotionSamples)
        assert len(samples) == 3

    def test_timestamps_correct(self, xml_file: Path) -> None:
        samples = load_synchrony_xml(xml_file, tolerance_mode="keep")
        assert samples.timestamps_ms[0] == 1_000_000
        assert samples.timestamps_ms[1] == 1_000_140

    def test_offsets_correct(self, xml_file: Path) -> None:
        samples = load_synchrony_xml(xml_file, tolerance_mode="keep")
        np.testing.assert_allclose(samples.offsets_mm[0], [1.0, -2.0, 0.5], atol=1e-5)
        np.testing.assert_allclose(samples.offsets_mm[1], [2.0, -1.5, 0.3], atol=1e-5)

    def test_rotations_zero(self, xml_file: Path) -> None:
        samples = load_synchrony_xml(xml_file)
        assert not samples.has_rotations
        assert np.all(samples.rotations_deg == 0)

    def test_out_of_tolerance_dropped(self, xml_file: Path) -> None:
        samples = load_synchrony_xml(xml_file, tolerance_mode="drop")
        # Third point has PotentialDiff=5 > tolerance=3 → dropped
        assert len(samples) == 2

    def test_out_of_tolerance_warns_and_keeps(self, xml_file: Path) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = load_synchrony_xml(xml_file, tolerance_mode="warn")
        assert len(samples) == 3  # kept but warned

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_synchrony_xml(tmp_path / "nonexistent.xml")

    def test_real_example_data(self) -> None:
        """Smoke test on the real MotionData file if available."""
        real = Path(
            "exampledata/motion_063/MotionData_fraction01.xml"
        )
        if not real.exists():
            pytest.skip("Example data not available")
        samples = load_synchrony_xml(real, tolerance_mode="warn")
        assert len(samples) > 0
        assert samples.offsets_mm.shape[1] == 3


# ---------------------------------------------------------------------------
# CSV tests
# ---------------------------------------------------------------------------

class TestLoadMotionCsv:
    def test_with_rotations(self, csv_with_rot: Path) -> None:
        samples = load_motion_csv(csv_with_rot)
        assert len(samples) == 2
        assert samples.has_rotations
        np.testing.assert_allclose(samples.offsets_mm[0], [1.0, -2.0, 0.5], atol=1e-5)
        np.testing.assert_allclose(samples.rotations_deg[0], [0.1, -0.2, 0.3], atol=1e-5)

    def test_no_rotations_warns_zeros(self, csv_no_rot: Path) -> None:
        with pytest.warns(UserWarning, match="rotation"):
            samples = load_motion_csv(csv_no_rot)
        assert not samples.has_rotations
        assert np.all(samples.rotations_deg == 0)

    def test_timestamps(self, csv_with_rot: Path) -> None:
        samples = load_motion_csv(csv_with_rot)
        assert samples.timestamps_ms[0] == 1_000_000
        assert samples.timestamps_ms[1] == 1_000_140

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_motion_csv(tmp_path / "ghost.csv")

    def test_missing_required_columns_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_text("a,b,c\n1,2,3\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_motion_csv(bad)
