import requests
import xml.etree.ElementTree as ET

from tools.base import ResearchTool
from agents.evidence_quality import deduplicate_sources, normalize_source


class LiteratureSearchTool(ResearchTool):
    """
    Search scholarly literature using the arXiv API.
    """

    ARXIV_URL = "https://export.arxiv.org/api/query"
    MAX_QUERY_LENGTH = 2_000
    MAX_RESPONSE_BYTES = 5 * 1024 * 1024
    CONNECT_TIMEOUT_SECONDS = 5
    READ_TIMEOUT_SECONDS = 20

    @property
    def name(self) -> str:
        return "literature_search"

    @property
    def description(self) -> str:
        return (
            "Search scholarly literature on arXiv and "
            "return relevant research papers."
        )

    def execute(
        self,
        query: str,
        max_results: int = 5,
    ) -> dict:
        """
        Search arXiv for research papers.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Literature search query cannot be empty.")
        query = query.strip()
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValueError(f"Literature search query exceeds {self.MAX_QUERY_LENGTH} characters.")
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise ValueError("max_results must be an integer.")
        if not 1 <= max_results <= 25:
            raise ValueError("max_results must be between 1 and 25.")

        params = {
            "search_query": f"all:{query.strip()}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        headers = {
            "User-Agent": "AURA Research Agent/1.0"
        }

        print("Searching arXiv...")

        try:
            response = requests.get(
                self.ARXIV_URL,
                params=params,
                headers=headers,
                timeout=(self.CONNECT_TIMEOUT_SECONDS, self.READ_TIMEOUT_SECONDS),
                stream=True,
            )

            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > self.MAX_RESPONSE_BYTES:
                raise RuntimeError("arXiv response exceeds the configured size limit.")
            response_chunks = []
            response_size = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                response_size += len(chunk)
                if response_size > self.MAX_RESPONSE_BYTES:
                    raise RuntimeError("arXiv response exceeds the configured size limit.")
                response_chunks.append(chunk)
            response_content = b"".join(response_chunks)

        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"arXiv literature search failed: {exc}"
            ) from exc

        try:
            root = ET.fromstring(
                response_content
            )

        except ET.ParseError as exc:
            raise RuntimeError(
                "Failed to parse arXiv response."
            ) from exc

        namespace = {
            "atom": "http://www.w3.org/2005/Atom"
        }

        results = []

        for entry in root.findall(
            "atom:entry",
            namespace,
        ):

            title = entry.find(
                "atom:title",
                namespace,
            )

            published = entry.find(
                "atom:published",
                namespace,
            )

            summary = entry.find(
                "atom:summary",
                namespace,
            )

            entry_id = entry.find(
                "atom:id",
                namespace,
            )

            authors = []

            for author in entry.findall(
                "atom:author",
                namespace,
            ):

                name = author.find(
                    "atom:name",
                    namespace,
                )

                if name is not None:
                    authors.append(
                        name.text.strip()
                    )

            results.append(
                {
                    "title": (
                        title.text.strip()
                        if title is not None
                        else None
                    ),
                    "publication_year": (
                        published.text[:4]
                        if published is not None
                        and published.text
                        else None
                    ),
                    "authors": authors,
                    "abstract": (
                        summary.text.strip()
                        if summary is not None
                        else None
                    ),
                    "url": (
                        entry_id.text.strip()
                        if entry_id is not None
                        else None
                    ),
                }
            )

        normalized = deduplicate_sources([
            normalize_source(item, query=query, provider="arxiv")
            for item in results
        ])
        enriched = [
            {
                **item.model_dump(mode="json"),
                "title": item.title,
                "authors": item.authors,
                "abstract": item.abstract,
                "url": item.url,
                "publication_year": item.publication_year,
            }
            for item in normalized
        ]
        return {
            "query": query,
            "source": "arXiv",
            "results": enriched,
            "count": len(enriched),
            "raw_count": len(results),
            "deduplicated_count": len(results) - len(enriched),
        }
