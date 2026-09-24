import requests
import xml.etree.ElementTree as ET

from tools.base import ResearchTool


class LiteratureSearchTool(ResearchTool):
    """
    Search scholarly literature using the arXiv API.
    """

    ARXIV_URL = "https://export.arxiv.org/api/query"

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

        if not query or not query.strip():
            raise ValueError(
                "Literature search query cannot be empty."
            )

        max_results = max(
            1,
            min(max_results, 25),
        )

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
                timeout=60,
            )

            response.raise_for_status()

        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"arXiv literature search failed: {exc}"
            ) from exc

        try:
            root = ET.fromstring(
                response.content
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

        return {
            "query": query,
            "source": "arXiv",
            "results": results,
            "count": len(results),
        }
