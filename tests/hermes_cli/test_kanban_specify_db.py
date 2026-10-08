"""Tests for kb.specify_triage_task — the DB-layer atomic promotion
from the triage column to todo. LLM-free by design."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        triage=True,
    )


def _park_in_triage(conn, tid, *, kind="capability", recurrences=2):
    """Force the block-loop breaker's end state directly: the card sits in
    ``triage`` with the loop marker at/over the limit. Written straight to the
    columns so the test does not have to drive two block/unblock cycles
    through ``claim_task`` + ``block_task``."""
    conn.execute(
        "UPDATE tasks SET status = 'triage', block_kind = ?, block_recurrences = ? WHERE id = ?",
        (kind, recurrences, tid),
    )


def test_specify_promotes_triage_to_todo(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="rough idea")
        assert kb.get_task(conn, tid).status == "triage"
    with kbc.connect() as conn:
        ok = kb.specify_triage_task(
            conn,
            tid,
            title="Refined: rough idea",
            body="**Goal**\nDo the thing.",
            author="specifier-bot",
        )
    assert ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    # No parents → recompute_ready should have flipped it past todo to ready.
    assert task.status == "ready"
    assert task.title == "Refined: rough idea"
    assert "**Goal**" in (task.body or "")


# --- block-loop park guard (kanban t_8e40219e) ---------------------------------
# A card the unblock-loop breaker routed to triage is parked FOR A HUMAN.
# Promotion must refuse it, or the decomposer resurrects it every tick and
# BLOCK_RECURRENCE_LIMIT never latches (t_5ca79764 looped 4x).

def test_specify_refuses_block_loop_parked_card(kanban_home):
    """(a) at-limit card is skipped and stays in triage, fields untouched."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="needs a human")
        _park_in_triage(conn, tid)
        ok = kb.specify_triage_task(
            conn, tid, title="Refined", body="b", author="decomposer",
        )
    assert ok is False
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        events = [e.kind for e in kb.list_events(conn, tid)]
        comments = kb.list_comments(conn, tid)
    assert task.status == "triage"
    assert task.title == "needs a human", "refusal must not write the LLM's fields"
    assert "specified" not in events
    assert comments == [], "refusal must not leave a 'promoted' audit comment"


def test_specify_promotes_below_limit_blocked_card(kanban_home):
    """(b) below-limit card still promotes — the breaker lets these through."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="first pass")
        _park_in_triage(conn, tid, kind="transient", recurrences=kb.BLOCK_RECURRENCE_LIMIT - 1)
        ok = kb.specify_triage_task(conn, tid, title="Refined", body="b", author="specifier")
    assert ok is True
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"


def test_specify_promotes_after_human_unblock_resets_recurrences(kanban_home):
    """(c) a human-released card (marker cleared) promotes normally."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="human released")
        _park_in_triage(conn, tid)
        assert kb.specify_triage_task(conn, tid, title="still parked", body="b") is False
        # What a human does: drag it out of triage (the dashboard's direct
        # status write clears nothing by itself), then reset the marker.
        conn.execute(
            "UPDATE tasks SET block_kind = NULL, block_recurrences = 0 WHERE id = ?", (tid,),
        )
        ok = kb.specify_triage_task(conn, tid, title="Refined", body="b", author="specifier")
    assert ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.title == "Refined"


def test_specify_still_promotes_untouched_triage_card(kanban_home):
    """A fresh triage card (block_kind IS NULL) is unaffected by the guard."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="fresh")
        assert kb.triage_park_reason(conn, tid) is None
        ok = kb.specify_triage_task(conn, tid, title="Refined", body="b")
    assert ok is True


def test_triage_park_reason_reports_and_expires(kanban_home):
    """The reason is human-readable, and absent once the card leaves triage."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="x")
        assert kb.triage_park_reason(conn, tid) is None, "untried card is not parked"
        _park_in_triage(conn, tid)
        reason = kb.triage_park_reason(conn, tid)
        assert reason and "block-loop detection" in reason
        assert str(kb.BLOCK_RECURRENCE_LIMIT) in reason
        assert "awaiting human" in reason
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (tid,))
        assert kb.triage_park_reason(conn, tid) is None
    with kbc.connect() as conn:
        assert kb.triage_park_reason(conn, "t_does_not_exist") is None


def test_decompose_triage_task_refuses_block_loop_parked_root(kanban_home):
    """The fan-out path had the same defect: it flipped root triage -> todo and
    inserted fresh children every tick. It must refuse before any child row."""
    from hermes_cli.kanban_db_graph import decompose_triage_task

    with kbc.connect() as conn:
        tid = _create_triage(conn, title="parked root")
        _park_in_triage(conn, tid)
        before = kb.list_tasks(conn, limit=1000)
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "child", "body": "b", "assignee": "researcher", "parents": []}],
            author="decomposer",
        )
        assert child_ids is None
        assert kb.get_task(conn, tid).status == "triage"
        after = kb.list_tasks(conn, limit=1000)
        assert len(after) == len(before), "no child may be inserted for a parked root"
        assert kb.triage_park_reason(conn, tid) is not None, "still parked after refusal"


def test_decompose_triage_task_promotes_below_limit_root(kanban_home):
    """Regression guard: the fan-out path still works below the limit."""
    from hermes_cli.kanban_db_graph import decompose_triage_task

    with kbc.connect() as conn:
        tid = _create_triage(conn, title="live root")
        _park_in_triage(conn, tid, kind="transient", recurrences=1)
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "child", "body": "b", "assignee": "researcher", "parents": []}],
        )
        assert child_ids is not None and len(child_ids) == 1
        assert kb.get_task(conn, tid).status in ("todo", "ready")


def test_specify_rejects_blank_title(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="rough")
    with kbc.connect() as conn, pytest.raises(ValueError):
        kb.specify_triage_task(conn, tid, title="   ", body="ok")


def test_specify_records_audit_comment_only_when_author_given(kanban_home):
    # With author → comment added.
    with kbc.connect() as conn:
        tid1 = _create_triage(conn, title="a")
        kb.specify_triage_task(
            conn, tid1, title="A-spec", body="b", author="ace"
        )
        comments1 = kb.list_comments(conn, tid1)
    assert len(comments1) == 1
    assert "Specified" in comments1[0].body
    assert comments1[0].author == "ace"

    # Without author → no comment (silent).
    with kbc.connect() as conn:
        tid2 = _create_triage(conn, title="b")
        kb.specify_triage_task(conn, tid2, title="B-spec", body="b")
        comments2 = kb.list_comments(conn, tid2)
    assert comments2 == []


