import hashlib
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import requests

from tools.base import ResearchTool


class DatasetDownloadTool(ResearchTool):
    """
    Download datasets from HTTP/HTTPS sources.

    Files are restricted to AURA's controlled
    storage/downloads directory.

    Downloaded files are never executed by this tool.
    """

    DOWNLOAD_DIRECTORY = (
        Path("storage")
        / "downloads"
    )

    MAX_FILE_SIZE = 250 * 1024 * 1024
    CONNECT_TIMEOUT_SECONDS = 5
    READ_TIMEOUT_SECONDS = 30

    @property
    def name(self) -> str:
        return "dataset_download"

    @property
    def description(self) -> str:
        return (
            "Download a dataset from an HTTP or HTTPS "
            "URL into AURA's controlled storage directory."
        )

    def execute(
        self,
        source_url: str,
        destination_path: str,
        checksum: str | None = None,
    ) -> dict:
        """
        Download a dataset.

        Args:
            source_url:
                HTTP or HTTPS URL.

            destination_path:
                Relative path inside
                storage/downloads/.

            checksum:
                Optional expected SHA-256 checksum.

        Returns:
            Structured download result.
        """

        self._validate_url(
            source_url
        )

        destination = (
            self._resolve_destination(
                destination_path
            )
        )

        expected_checksum = (
            self._validate_checksum(
                checksum
            )
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            response = requests.get(
                source_url,
                stream=True,
                timeout=(self.CONNECT_TIMEOUT_SECONDS, self.READ_TIMEOUT_SECONDS),
                allow_redirects=False,
            )

            if 300 <= getattr(response, "status_code", 200) < 400:
                raise RuntimeError("Dataset download redirects are not permitted.")

            content_length = getattr(response, "headers", {}).get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("Dataset response has an invalid Content-Length.") from exc
                if declared_size < 0 or declared_size > self.MAX_FILE_SIZE:
                    raise RuntimeError("Dataset exceeds the maximum allowed download size.")

            response.raise_for_status()

            total_bytes = 0

            sha256 = hashlib.sha256()

            with destination.open(
                "wb"
            ) as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if not chunk:
                        continue

                    total_bytes += len(
                        chunk
                    )

                    if (
                        total_bytes
                        > self.MAX_FILE_SIZE
                    ):

                        raise RuntimeError(
                            "Dataset exceeds the maximum "
                            "allowed download size."
                        )

                    file.write(
                        chunk
                    )

                    sha256.update(
                        chunk
                    )

        except requests.exceptions.RequestException as exc:

            if destination.exists():
                destination.unlink()

            raise RuntimeError(
                f"Dataset download failed: {exc}"
            ) from exc

        except Exception:

            if destination.exists():
                destination.unlink()

            raise

        calculated_checksum = (
            sha256.hexdigest()
        )

        checksum_verified = None

        if expected_checksum is not None:

            checksum_verified = (
                calculated_checksum
                == expected_checksum
            )

            if not checksum_verified:

                if destination.exists():
                    destination.unlink()

                raise RuntimeError(
                    "Downloaded file checksum "
                    "does not match the expected SHA-256."
                )

        return {
            "status": "success",
            "source_url": source_url,
            "local_path": str(
                destination
            ),
            "download_status": "success",
            "size_bytes": total_bytes,
            "sha256": calculated_checksum,
            "checksum_verified": checksum_verified,
        }

    def _validate_url(
        self,
        source_url: str,
    ) -> None:
        """
        Ensure only HTTP/HTTPS URLs are accepted.
        """

        if not source_url:
            raise ValueError(
                "Source URL cannot be empty."
            )

        parsed = urlparse(
            source_url
        )

        if parsed.scheme not in {
            "http",
            "https",
        }:

            raise ValueError(
                "Dataset source must use "
                "HTTP or HTTPS."
            )

        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Dataset source URL is invalid.")

        allowed_port = 80 if parsed.scheme == "http" else 443
        if parsed.port not in (None, allowed_port):
            raise ValueError("Dataset source URL uses a disallowed port.")

        hostname = parsed.hostname.rstrip(".").lower()
        if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(
            (".localhost", ".local", ".internal", ".test")
        ):
            raise ValueError("Dataset source host is not publicly routable.")

        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("Dataset source host must use a public IP address.")

    def _validate_checksum(
        self,
        checksum: str | None,
    ) -> str | None:
        """
        Validate and normalize an optional SHA-256 checksum.

        Validation happens before any network request or
        file creation.
        """

        if checksum is None:
            return None

        expected_checksum = (
            checksum.strip().lower()
        )

        if (
            len(expected_checksum)
            != 64
        ):

            raise ValueError(
                "SHA-256 checksum must contain "
                "exactly 64 hexadecimal characters."
            )

        if any(
            character not in "0123456789abcdef"
            for character in expected_checksum
        ):

            raise ValueError(
                "SHA-256 checksum must contain "
                "only hexadecimal characters."
            )

        return expected_checksum

    def _resolve_destination(
        self,
        destination_path: str,
    ) -> Path:
        """
        Resolve a destination path while preventing
        path traversal outside storage/downloads/.
        """

        if not destination_path:
            raise ValueError(
                "Destination path cannot be empty."
            )

        root = (
            self.DOWNLOAD_DIRECTORY
            .resolve()
        )

        destination = (
            root
            / destination_path
        ).resolve()

        try:
            destination.relative_to(
                root
            )

        except ValueError as exc:

            raise ValueError(
                "Destination path must remain "
                "inside storage/downloads/."
            ) from exc

        return destination
