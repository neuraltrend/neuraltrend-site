from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from tools.forward_record_backup_lib import create_backup, verify_backup


def test_forward_record_backup_round_trip(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "public").mkdir(parents=True)
    (source / "sandbox").mkdir(parents=True)
    (source / "public" / "BTC-USD.csv").write_text(
        "Date,Close,epoch_signal\n2026-07-27,100,1\n",
        encoding="utf-8",
    )
    (source / "sandbox" / "ETH-USD.csv").write_text(
        "Date,Close,epoch_signal\n2026-07-27,200,0\n",
        encoding="utf-8",
    )

    archive, created = create_backup(source, output, commit="abc123")
    verified = verify_backup(archive)

    assert archive.is_file()
    assert created["file_count"] == 2
    assert verified["file_count"] == 2
    assert len(verified["archive_sha256"]) == 64


def test_forward_record_backup_rejects_unsafe_archive(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = b"bad"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe archive path"):
        verify_backup(archive)


def test_forward_record_backup_rejects_output_inside_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="outside Forward Record storage"):
        create_backup(source, source / "backups")
