"""Dry-run approval gate tests."""

from arch_chat.tools.mock_publish import publish_post


def test_publish_post_dry_run_vs_live():
    preview = publish_post("Hello", ["test"], dry_run=True)
    live = publish_post("Hello", ["test"], dry_run=False)
    assert "DRY RUN" in preview
    assert "LIVE" in live


def test_dry_run_gate_blocks_live_without_approval():
    approved = False
    preview = publish_post("Hello", ["test"], dry_run=True)
    if approved:
        result = publish_post("Hello", ["test"], dry_run=False)
    else:
        result = f"Not approved. Preview only:\n\n{preview}"
    assert "Not approved" in result
    assert "DRY RUN" in result
    assert "LIVE" not in result


def test_dry_run_gate_allows_live_with_approval():
    approved = True
    preview = publish_post("Hello", ["test"], dry_run=True)
    if approved:
        result = publish_post("Hello", ["test"], dry_run=False)
    else:
        result = preview
    assert "LIVE" in result
