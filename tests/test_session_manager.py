from db.base.session_manager import SessionManager


def test_session_manager_creates_sqlite_parent_directory(tmp_path):
    db_path = tmp_path / "nested" / "smart_appointment.db"

    manager = SessionManager(f"sqlite:///{db_path.as_posix()}")
    manager.close()

    assert db_path.exists()
