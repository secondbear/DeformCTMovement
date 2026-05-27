"""Tests for dicom_export.py — DICOM CT series export and private tags."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import UID

    _PYDICOM = True
except ImportError:
    _PYDICOM = False

pytestmark = pytest.mark.skipif(not _PYDICOM, reason="pydicom not installed")


# ---------------------------------------------------------------------------
# Minimal fake DICOM dataset factory
# ---------------------------------------------------------------------------

def _make_fake_ct_dataset(z: float = 0.0) -> "Dataset":
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()

    ds.is_implicit_VR = False
    ds.is_little_endian = True

    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = "1.2.3.4.5.6.7.8.9"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.PatientID = "SYNTHETIC"
    ds.PatientName = "SYNTHETIC^PATIENT"

    ds.Rows = 16
    ds.Columns = 16
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelSpacing = [2.0, 2.0]
    ds.SliceThickness = 2.0
    ds.ImagePositionPatient = [0.0, 0.0, z]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0

    arr = np.zeros((16, 16), dtype=np.int16)
    ds.PixelData = arr.tobytes()
    return ds


def _make_source_datasets(n: int = 4) -> "list[Dataset]":
    base_uid = pydicom.uid.generate_uid()
    datasets = []
    for i in range(n):
        ds = _make_fake_ct_dataset(z=float(i * 2))
        ds.SeriesInstanceUID = base_uid
        datasets.append(ds)
    return datasets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSaveCtSeries:
    def test_writes_correct_number_of_slices(self, tmp_path: Path) -> None:
        from gendosecalc.deform.dicom_export import save_ct_series
        from gendosecalc.deform.models import DeformationConfig

        n = 4
        vol = np.zeros((n, 16, 16), dtype=np.int16)
        sources = _make_source_datasets(n)
        config = DeformationConfig()

        save_ct_series(
            deformed_array=vol,
            source_datasets=sources,
            out_dir=tmp_path,
            state_index=0,
            epoch_ms=1_000_000_000,
            dvf_filename="state_000_dvf.mha",
            tx=1.0, ty=0.0, tz=0.0,
            rx=0.0, ry=0.0, rz=0.0,
            config=config,
            cluster_weight=5,
        )
        dcm_files = list(tmp_path.glob("CT.*.dcm"))
        assert len(dcm_files) == n

    def test_new_series_uid_differs_from_source(self, tmp_path: Path) -> None:
        from gendosecalc.deform.dicom_export import save_ct_series
        from gendosecalc.deform.models import DeformationConfig

        sources = _make_source_datasets(2)
        src_uid = sources[0].SeriesInstanceUID
        vol = np.zeros((2, 16, 16), dtype=np.int16)
        config = DeformationConfig()

        new_uid = save_ct_series(
            deformed_array=vol,
            source_datasets=sources,
            out_dir=tmp_path,
            state_index=1,
            epoch_ms=1_748_339_000_000,
            dvf_filename="state_001_dvf.mha",
            tx=0.0, ty=2.0, tz=0.0,
            rx=0.0, ry=0.0, rz=0.0,
            config=config,
            cluster_weight=3,
        )
        assert new_uid != str(src_uid)

    def test_content_date_matches_epoch(self, tmp_path: Path) -> None:
        from gendosecalc.deform.dicom_export import save_ct_series
        from gendosecalc.deform.models import DeformationConfig

        epoch_ms = 1_748_339_000_000  # 2025-05-27
        sources = _make_source_datasets(1)
        vol = np.zeros((1, 16, 16), dtype=np.int16)
        config = DeformationConfig()

        save_ct_series(
            deformed_array=vol,
            source_datasets=sources,
            out_dir=tmp_path,
            state_index=0,
            epoch_ms=epoch_ms,
            dvf_filename="state_000_dvf.mha",
            tx=0.0, ty=0.0, tz=0.0,
            rx=0.0, ry=0.0, rz=0.0,
            config=config,
            cluster_weight=1,
        )
        dcm = pydicom.dcmread(str(list(tmp_path.glob("CT.*.dcm"))[0]))
        dt = datetime.datetime.fromtimestamp(
            epoch_ms / 1000.0, tz=datetime.timezone.utc
        )
        assert dcm.ContentDate == dt.strftime("%Y%m%d")

    def test_frame_of_reference_preserved(self, tmp_path: Path) -> None:
        from gendosecalc.deform.dicom_export import save_ct_series
        from gendosecalc.deform.models import DeformationConfig

        sources = _make_source_datasets(1)
        fof_uid = "1.2.3.4.5.6.7.8.9"
        assert sources[0].FrameOfReferenceUID == fof_uid

        vol = np.zeros((1, 16, 16), dtype=np.int16)
        save_ct_series(
            deformed_array=vol,
            source_datasets=sources,
            out_dir=tmp_path,
            state_index=0,
            epoch_ms=1_000_000,
            dvf_filename="state_000_dvf.mha",
            tx=0.0, ty=0.0, tz=0.0,
            rx=0.0, ry=0.0, rz=0.0,
            config=DeformationConfig(),
            cluster_weight=1,
        )
        dcm = pydicom.dcmread(str(list(tmp_path.glob("CT.*.dcm"))[0]))
        assert dcm.FrameOfReferenceUID == fof_uid

    def test_pixel_data_correct(self, tmp_path: Path) -> None:
        from gendosecalc.deform.dicom_export import save_ct_series
        from gendosecalc.deform.models import DeformationConfig

        sources = _make_source_datasets(1)
        vol = np.full((1, 16, 16), fill_value=42, dtype=np.int16)
        save_ct_series(
            deformed_array=vol,
            source_datasets=sources,
            out_dir=tmp_path,
            state_index=0,
            epoch_ms=1_000_000,
            dvf_filename="state_000_dvf.mha",
            tx=0.0, ty=0.0, tz=0.0,
            rx=0.0, ry=0.0, rz=0.0,
            config=DeformationConfig(),
            cluster_weight=1,
        )
        dcm = pydicom.dcmread(str(list(tmp_path.glob("CT.*.dcm"))[0]))
        arr = dcm.pixel_array.astype(np.int16)
        assert arr[8, 8] == 42


class TestWriteRtdoseStateTags:
    def test_sets_content_date_and_time(self) -> None:
        from gendosecalc.deform.dicom_export import write_rtdose_state_tags
        from gendosecalc.deform.models import DeformationConfig, EnsembleManifestEntry

        entry = EnsembleManifestEntry(
            state_index=0,
            epoch_ms=1_748_339_000_000,
            iso_timestamp="2025-05-27T00:00:00+00:00",
            cluster_weight=5,
            tx=1.0, ty=0.0, tz=0.0,
            rx=0.0, ry=0.0, rz=0.0,
            ct_dir="state_000_ct",
            dvf_path="state_000_dvf.mha",
            deformed_series_instance_uid="1.2.3",
            source_ct_series_instance_uid="1.2.4",
        )
        ds = Dataset()
        write_rtdose_state_tags(ds, entry, config=DeformationConfig())

        expected_dt = datetime.datetime.fromtimestamp(
            1_748_339_000_000 / 1000.0, tz=datetime.timezone.utc
        )
        assert ds.ContentDate == expected_dt.strftime("%Y%m%d")
