import hashlib
import re
from urllib.parse import urlparse, urlunparse

from storage.models.evidence import ResearchSource


_DOI_RE = re.compile(r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?(10\.\d{4,9}/[^\s]+)", re.I)
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+)", re.I)


def normalize_source(record: dict, query: str = "", provider: str = "unknown") -> ResearchSource:
    """Normalize one scholarly record and attach explainable provenance."""
    raw_url = str(record.get("url") or record.get("id") or "").strip()
    doi_value = str(record.get("doi") or "").strip()
    doi_match = _DOI_RE.search(doi_value or raw_url)
    doi = doi_match.group(1).rstrip(".,;") if doi_match else None
    arxiv_match = _ARXIV_RE.search(raw_url)
    arxiv_id = str(record.get("arxiv_id") or (arxiv_match.group(1) if arxiv_match else "")).strip() or None
    if arxiv_id:
        arxiv_id = re.sub(r"\.pdf$", "", arxiv_id, flags=re.I)
    if doi:
        canonical = "doi:" + doi.lower()
    elif arxiv_id:
        canonical = "arxiv:" + arxiv_id.lower()
    else:
        parsed = urlparse(raw_url)
        normalized_url = urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", parsed.query, ""))
        canonical = normalized_url or "title:" + " ".join(str(record.get("title", "")).lower().split())
    source_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    title = " ".join(str(record.get("title") or "Untitled source").split())
    authors = [str(a).strip() for a in record.get("authors", []) if str(a).strip()]
    abstract = str(record.get("abstract") or "").strip() or None
    reasons = []
    score = 0.0
    if doi:
        score += 0.35
        reasons.append("Persistent DOI identifier is present (+0.35).")
    elif arxiv_id:
        score += 0.25
        reasons.append("Recognized arXiv identifier is present (+0.25).")
    if raw_url.lower().startswith("https://"):
        score += 0.15
        reasons.append("HTTPS source URL is present (+0.15).")
    if title != "Untitled source":
        score += 0.15
        reasons.append("Title metadata is present (+0.15).")
    if authors:
        score += 0.10
        reasons.append("Author metadata is present (+0.10).")
    if abstract:
        score += 0.15
        reasons.append("Abstract metadata is present (+0.15).")
    if record.get("publication_year"):
        score += 0.10
        reasons.append("Publication-year metadata is present (+0.10).")
    score = min(1.0, round(score, 4))
    if not reasons:
        reasons.append("Insufficient metadata for a positive heuristic score.")
    return ResearchSource(
        source_id=source_id,
        canonical_id=canonical,
        title=title,
        authors=authors,
        abstract=abstract,
        url=raw_url,
        doi=doi,
        arxiv_id=arxiv_id,
        publisher=record.get("publisher"),
        publication_year=int(record["publication_year"]) if str(record.get("publication_year", "")).isdigit() else None,
        source_provider=provider,
        retrieved_query=query or None,
        credibility_score=score,
        credibility_reasons=reasons,
        metadata={"scoring_method": "metadata_completeness_heuristic_v1"},
    )


def deduplicate_sources(records: list[ResearchSource]) -> list[ResearchSource]:
    """Deduplicate by DOI, arXiv ID, or normalized URL; retain richer metadata."""
    selected: dict[str, ResearchSource] = {}
    for source in records:
        current = selected.get(source.canonical_id)
        richness = lambda item: sum(bool(value) for value in (item.title, item.authors, item.abstract, item.doi, item.arxiv_id, item.publication_year))
        if current is None or richness(source) > richness(current):
            selected[source.canonical_id] = source
    return sorted(selected.values(), key=lambda item: item.canonical_id)
