"""Final session titles reach Telegram groups through the real callback and scheduler.

Only the external bot transport is replaced; no live Telegram mutations are made.
"""

import asyncio
from types import SimpleNamespace

import pytest

from agent.secret_scope import is_multiplex_active, set_multiplex_active
from agent.title_generator import apply_instant_title, auto_title_session
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from plugins.platforms.telegram.adapter import TelegramAdapter


class RecordingBot:
    def __init__(self):
        self.renames = []
        self.replies = []
        self.called = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.error = None

    async def set_chat_title(self, *, chat_id, title):
        self.renames.append((chat_id, title, get_hermes_home()))
        self.called.set()
        await self.release.wait()
        if self.error:
            raise self.error
        return True

    async def send_message(self, **kwargs):
        self.replies.append(kwargs)
        return SimpleNamespace(message_id=len(self.replies))

    async def send_chat_action(self, **kwargs):
        return True


def _adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={"rich_messages": False}))
    adapter._bot = RecordingBot()
    return adapter


def _attach(runner, source, session_id):
    agent = SimpleNamespace(session_id=session_id)
    ctx = TurnContext(source=source)
    TurnRunner(runner, ctx)._attach_session_title_callback(agent, ctx)
    return getattr(agent, "_on_session_title", lambda *_: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_type,is_forum,project", [
    ("group", False, False), ("supergroup", False, False),
    ("supergroup", False, True), ("supergroup", True, False),
])
async def test_stored_final_title_renames_originating_group(tmp_path, raw_type, is_forum, project):
    homes = {name: tmp_path / name for name in ("default", "second")}
    for home in homes.values():
        home.mkdir()
    adapters = {name: _adapter() for name in homes}
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapters["default"]}
    runner._profile_adapters = {"second": {Platform.TELEGRAM: adapters["second"]}}
    runner._gateway_loop = asyncio.get_running_loop()
    active = is_multiplex_active()
    set_multiplex_active(True)
    try:
        # A -> B -> A, identical group names and IDs, distinct transport owners/homes.
        for index, profile in enumerate(("default", "second", "default")):
            adapter = adapters[profile]
            bot = adapter._bot
            bot.called.clear()
            adapter._owner_profile = profile
            with _profile_runtime_scope(homes[profile], prepared_secret_scope={}):
                source = adapter.build_source(
                    chat_id="-101", chat_name="Same project" if project else "Same group",
                    chat_type=adapter._normalize_chat_type(raw_type, is_forum=is_forum),
                    thread_id="42" if is_forum else None,
                )
                callback = _attach(runner, source, f"session-{index}")
                db = SessionDB(homes[profile] / "state.db")
                try:
                    db.create_session(f"session-{index}", source="telegram")
                    assert db.set_auto_title(f"session-{index}", f"Investigate café latency {index}", source="llm")
                    title = db.get_session_title(f"session-{index}")
                finally:
                    db.close()
                # Reopen persistence before firing the callback; titles are not regenerated.
                db = SessionDB(homes[profile] / "state.db")
                try:
                    assert db.get_session_title(f"session-{index}") == title
                    await asyncio.to_thread(callback, title, "llm")
                    await asyncio.wait_for(bot.called.wait(), timeout=2)
                finally:
                    db.close()
                assert bot.renames[-1] == (-101, title, homes[profile])
        assert len(adapters["default"]._bot.renames) == 2
        assert len(adapters["second"]._bot.renames) == 1
    finally:
        set_multiplex_active(active)


@pytest.mark.asyncio
async def test_filtered_titles_and_rejection_do_not_interrupt_replies(tmp_path, caplog):
    adapter = _adapter()
    bot = adapter._bot
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {"missing": {}}
    runner._gateway_loop = asyncio.get_running_loop()
    source = adapter.build_source(chat_id="-101", chat_type="group")
    callback = _attach(runner, source, "group-session")
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("group-session", source="telegram")
        await asyncio.to_thread(apply_instant_title, db, "group-session", "Investigate request latency", callback)
        # A declined upgrade produces only a derived title, never a group rename.
        await asyncio.to_thread(
            auto_title_session, db, "group-session", "Investigate request latency",
            title_callback=callback, runtime_validator=lambda: False,
        )
        assert db.get_session_title_source("group-session") == "derived"
        await asyncio.to_thread(callback, "Manual title", "user")
        for platform, chat_type in ((Platform.TELEGRAM, "dm"), (Platform.TELEGRAM, "channel"),
                                    (Platform.DISCORD, "group"), (Platform.SLACK, "group")):
            excluded = SessionSource(platform=platform, chat_id="-101", chat_type=chat_type)
            await asyncio.to_thread(_attach(runner, excluded, "excluded"), "Do not rename", "llm")
        unavailable = SessionSource(platform=Platform.TELEGRAM, chat_id="-101", chat_type="group", profile="missing")
        await asyncio.to_thread(_attach(runner, unavailable, "unavailable"), "Do not borrow default", "llm")

        # The transport sees exact text, including whitespace and an over-limit title.
        title = "  Café  " + "x" * 129
        bot.release.clear()
        bot.error = ValueError("sensitive transport details must not be logged")
        await asyncio.to_thread(callback, title, "llm")
        await asyncio.wait_for(bot.called.wait(), timeout=2)
        assert [(chat, text) for chat, text, _home in bot.renames] == [(-101, title)]
        result = await asyncio.wait_for(adapter.send("-101", "Reply while rename waits"), timeout=2)
        assert result.success
        bot.release.set()
        # A loop barrier gives the already-scheduled rename its error-handling turn.
        await asyncio.sleep(0)
        result = await asyncio.wait_for(adapter.send("-101", "Reply after rejection"), timeout=2)
        assert result.success
        assert len(bot.replies) == 2
        assert "Telegram group title rename rejected" in caplog.text
        assert "ValueError" in caplog.text
        assert "sensitive transport details" not in caplog.text
    finally:
        bot.release.set()
        db.close()
