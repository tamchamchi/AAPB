#!/usr/bin/env python3
"""Search Semantic Scholar for prior work related to an AAPB research claim.

The API key is read from SEMANTIC_SCHOLAR_API_KEY and is never written to output.
This is a retrieval aid, not an automated novelty verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

FIELDS = "title,year,authors,abstract,url,venue,citationCount,externalIds"
API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def search(query: str, api_key: str, limit: int, retries: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {"query": query, "limit": limit, "fields": FIELDS}
    )
    request = urllib.request.Request(
        f"{API_URL}?{params}",
        headers={"x-api-key": api_key, "User-Agent": "AAPB-novelty-review/1.0"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            return payload.get("data", [])
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < retries:
                time.sleep(min(2**attempt * 2, 16))
                continue
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Semantic Scholar returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
                continue
            raise RuntimeError(f"Could not reach Semantic Scholar: {error.reason}") from error
    return []


def render_markdown(claim: str, results: list[tuple[str, list[dict]]]) -> str:
    lines = [
        f"# Semantic Scholar novelty search — {date.today().isoformat()}",
        "",
        f"**Claim searched:** {claim}",
        "",
        "> Search results are leads for human review. Search ranking and absence of a match do not establish novelty.",
        "",
    ]
    seen: set[str] = set()
    for query, papers in results:
        lines.extend([f"## Query: {query}", ""])
        fresh = []
        for paper in papers:
            paper_id = paper.get("paperId") or paper.get("url") or paper.get("title")
            if paper_id not in seen:
                seen.add(paper_id)
                fresh.append(paper)
        if not fresh:
            lines.extend(["No new records for this query.", ""])
            continue
        for paper in fresh:
            authors = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:6])
            if len(paper.get("authors", [])) > 6:
                authors += ", et al."
            lines.extend(
                [
                    f"### [{paper.get('title', 'Untitled')}]({paper.get('url', '')})",
                    f"- {authors} ({paper.get('year', 'n.d.')}). {paper.get('venue') or 'Venue not listed'}. "
                    f"Citations reported by Semantic Scholar: {paper.get('citationCount', 'n/a')}.",
                    f"- IDs: `{json.dumps(paper.get('externalIds') or {}, ensure_ascii=False)}`",
                    f"- Abstract: {paper.get('abstract') or 'No abstract indexed.'}",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", required=True, help="The precise proposed contribution to check")
    parser.add_argument(
        "--query",
        action="append",
        help="A focused Semantic Scholar query; repeat to run several (default: claim plus two expansions)",
    )
    parser.add_argument("--limit", type=int, default=10, help="Results per query (1–100; default: 10)")
    parser.add_argument("--retries", type=int, default=4, help="Retries for rate limits/network errors")
    parser.add_argument("--out", type=Path, help="Markdown path (default: output/novelty_<date>.md)")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        parser.error("Set SEMANTIC_SCHOLAR_API_KEY in the environment; the key is never stored in the repo.")

    queries = args.query or [
        args.claim,
        f"{args.claim} text-to-image diffusion",
        f"{args.claim} classifier-free guidance",
    ]
    results = []
    try:
        for index, query in enumerate(queries):
            if index:
                time.sleep(1.1)
            results.append((query, search(query, api_key, args.limit, args.retries)))
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    output = args.out or Path("output") / f"novelty_{date.today().isoformat()}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(args.claim, results), encoding="utf-8")
    print(f"Wrote {output} ({sum(len(papers) for _, papers in results)} retrieved records).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
