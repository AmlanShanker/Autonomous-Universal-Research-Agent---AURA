import hashlib

import pytest

from tools.implementations import DatasetDownloadTool


class FakeResponse:
    """
    Minimal fake requests.Response used for deterministic tests.
    """

    def __init__(
        self,
        content: bytes,
    ):
        self.content = content

    def raise_for_status(self):
        pass

    def iter_content(
        self,
        chunk_size: int = 1024 * 1024,
    ):
        for index in range(
            0,
            len(self.content),
            chunk_size,
        ):
            yield self.content[
                index:index + chunk_size
            ]


def test_successful_download(
    monkeypatch,
    tmp_path,
):
    content = b"AURA dataset test content."

    def fake_get(
        *args,
        **kwargs,
    ):
        return FakeResponse(
            content
        )

    monkeypatch.setattr(
        "tools.implementations.requests.get",
        fake_get,
    )

    tool = DatasetDownloadTool()

    tool.DOWNLOAD_DIRECTORY = (
        tmp_path
    )

    result = tool.execute(
        source_url="https://example.com/data.txt",
        destination_path="data.txt",
    )

    destination = (
        tmp_path
        / "data.txt"
    )

    expected_checksum = hashlib.sha256(
        content
    ).hexdigest()

    assert result["status"] == "success"

    assert result["download_status"] == "success"

    assert result["size_bytes"] == len(
        content
    )

    assert result["sha256"] == (
        expected_checksum
    )

    assert result["checksum_verified"] is None

    assert destination.exists()

    assert destination.read_bytes() == content


def test_successful_download_with_checksum(
    monkeypatch,
    tmp_path,
):
    content = b"Checksum validation test."

    expected_checksum = hashlib.sha256(
        content
    ).hexdigest()

    def fake_get(
        *args,
        **kwargs,
    ):
        return FakeResponse(
            content
        )

    monkeypatch.setattr(
        "tools.implementations.requests.get",
        fake_get,
    )

    tool = DatasetDownloadTool()

    tool.DOWNLOAD_DIRECTORY = (
        tmp_path
    )

    result = tool.execute(
        source_url="https://example.com/data.txt",
        destination_path="checked.txt",
        checksum=expected_checksum,
    )

    assert result["status"] == "success"

    assert result["checksum_verified"] is True

    assert (
        tmp_path
        / "checked.txt"
    ).exists()


def test_invalid_checksum_is_rejected_before_download(
    monkeypatch,
    tmp_path,
):
    download_called = False

    def fake_get(
        *args,
        **kwargs,
    ):
        nonlocal download_called

        download_called = True

        return FakeResponse(
            b"should not download"
        )

    monkeypatch.setattr(
        "tools.implementations.requests.get",
        fake_get,
    )

    tool = DatasetDownloadTool()

    tool.DOWNLOAD_DIRECTORY = (
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="SHA-256 checksum",
    ):
        tool.execute(
            source_url="https://example.com/data.txt",
            destination_path="invalid.txt",
            checksum="invalid",
        )

    assert download_called is False

    assert not (
        tmp_path
        / "invalid.txt"
    ).exists()


def test_checksum_mismatch_removes_file(
    monkeypatch,
    tmp_path,
):
    content = b"Original content."

    incorrect_checksum = hashlib.sha256(
        b"Different content."
    ).hexdigest()

    def fake_get(
        *args,
        **kwargs,
    ):
        return FakeResponse(
            content
        )

    monkeypatch.setattr(
        "tools.implementations.requests.get",
        fake_get,
    )

    tool = DatasetDownloadTool()

    tool.DOWNLOAD_DIRECTORY = (
        tmp_path
    )

    with pytest.raises(
        RuntimeError,
        match="checksum",
    ):
        tool.execute(
            source_url="https://example.com/data.txt",
            destination_path="mismatch.txt",
            checksum=incorrect_checksum,
        )

    assert not (
        tmp_path
        / "mismatch.txt"
    ).exists()


def test_path_traversal_is_rejected(
    tmp_path,
):
    tool = DatasetDownloadTool()

    tool.DOWNLOAD_DIRECTORY = (
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="inside storage/downloads",
    ):
        tool.execute(
            source_url="https://example.com/data.txt",
            destination_path="../outside.txt",
        )


def test_non_http_url_is_rejected(
    tmp_path,
):
    tool = DatasetDownloadTool()

    tool.DOWNLOAD_DIRECTORY = (
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="HTTP or HTTPS",
    ):
        tool.execute(
            source_url="file:///secret/data.txt",
            destination_path="data.txt",
        )
