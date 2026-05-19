"""Dry-run publish tool from Dry-Run architecture doc."""

from __future__ import annotations

import datetime
import hashlib
from typing import List


def publish_post(content: str, hashtags: List[str], dry_run: bool = True) -> str:
    """Publish a social media post. dry_run=True previews only."""
    ts = datetime.datetime.now().isoformat()
    tags = " ".join(f"#{h.lstrip('#')}" for h in hashtags)
    full = f"{content}\n\n{tags}".strip()
    if dry_run:
        return f"[DRY RUN @ {ts}] Would publish:\n---\n{full}\n---"
    post_id = hashlib.md5(full.encode()).hexdigest()[:8]
    return f"[LIVE @ {ts}] Published id={post_id}"
