from types import SimpleNamespace

from local_elections_jharkhand.build import compare_readers
from local_elections_jharkhand.extract import image_seats_2015 as ocr_seats


def test_packaged_ocr_helpers_import() -> None:
    assert compare_readers.parse is ocr_seats.parse


def test_ocr_seats_uses_isolated_uv_group(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ocr_seats.subprocess, "run", fake_run)
    ocr_seats.ocr_all([tmp_path / "page.png"], tmp_path)

    command, kwargs = calls[0]
    assert command[:5] == ["uv", "run", "--group", "surya", "python"]
    assert kwargs == {"cwd": ocr_seats.ROOT}
