from pathlib import Path

import app as application


def _set_backup_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(application, "BACKUP_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(application, "BACKUP_STORAGE_EXPLICIT", True)
    monkeypatch.setattr(application, "BACKUP_RETENTION_COUNT", 2)


def test_admin_backups_requires_admin(authenticated_client):
    response = authenticated_client.get("/admin/backups")
    assert response.status_code == 404


def test_admin_backups_lists_and_downloads_managed_file(admin_client, monkeypatch, tmp_path):
    _set_backup_dir(monkeypatch, tmp_path)
    backup = tmp_path / "neuraltrend-postgres-20260728T000000Z.dump"
    backup.write_bytes(b"safe-test-backup")
    (tmp_path / (backup.name + ".sha256")).write_text("abc  file\n")
    page = admin_client.get("/admin/backups")
    assert page.status_code == 200
    assert backup.name.encode() not in page.data  # filenames are intentionally not displayed
    download = admin_client.get(f"/admin/backups/download/{backup.name}")
    assert download.status_code == 200
    assert download.data == b"safe-test-backup"
    assert "attachment" in download.headers["Content-Disposition"]


def test_admin_backup_rejects_path_traversal(admin_client, monkeypatch, tmp_path):
    _set_backup_dir(monkeypatch, tmp_path)
    response = admin_client.get("/admin/backups/download/..%2Fapp.py")
    assert response.status_code == 404


def test_admin_backup_create_database(admin_client, monkeypatch, tmp_path):
    _set_backup_dir(monkeypatch, tmp_path)
    def fake_create(directory, database_url):
        path = Path(directory) / "neuraltrend-postgres-20260728T000001Z.dump"
        path.write_bytes(b"dump")
        Path(str(path) + ".sha256").write_text("checksum\n")
        return path
    monkeypatch.setattr(application, "create_database_backup", fake_create)
    response = admin_client.post("/admin/backups/create", data={"kind": "database"})
    assert response.status_code == 302
    assert (tmp_path / "neuraltrend-postgres-20260728T000001Z.dump").is_file()


def test_admin_backup_delete_removes_archive_and_checksum(admin_client, monkeypatch, tmp_path):
    _set_backup_dir(monkeypatch, tmp_path)
    name = "neuraltrend-forward-record-20260728T000000Z.tar.gz"
    (tmp_path / name).write_bytes(b"archive")
    (tmp_path / (name + ".sha256")).write_text("checksum\n")
    response = admin_client.post(f"/admin/backups/delete/{name}")
    assert response.status_code == 302
    assert not (tmp_path / name).exists()
    assert not (tmp_path / (name + ".sha256")).exists()
