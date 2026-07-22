from nanobot.session.manager import SessionManager


def test_raw_transcript_survives_live_trim(tmp_path):
    manager = SessionManager(workspace=tmp_path)
    session = manager.get_or_create("cli:topic")
    session.add_message("user", "first")
    session.add_message("assistant", "answer")
    manager.save(session)
    session.messages = session.messages[-1:]
    manager.save(session)
    assert [m["content"] for m in manager.display_history(session.key, session.messages)] == [
        "first",
        "answer",
    ]


def test_legacy_tail_is_seeded_once(tmp_path):
    manager = SessionManager(workspace=tmp_path)
    session = manager.get_or_create("cli:legacy")
    session.messages = [{"role": "user", "content": "retained"}]
    manager.save(session)
    manager.save(session)
    assert len(manager.display_history(session.key, session.messages)) == 1


def test_delete_removes_raw_transcript(tmp_path):
    manager = SessionManager(workspace=tmp_path)
    session = manager.get_or_create("cli:delete")
    session.add_message("user", "remove")
    manager.save(session)
    path = manager.transcripts.path_for(session.key)
    assert path.exists()
    assert manager.delete_session(session.key)
    assert not path.exists()


def test_display_history_pages_by_complete_user_turns(tmp_path):
    manager = SessionManager(workspace=tmp_path)
    session = manager.get_or_create("cli:paged")
    for turn in range(5):
        session.add_message("user", f"question-{turn}")
        session.add_message("assistant", "", tool_calls=[{
            "id": f"call-{turn}",
            "function": {"name": "read_file", "arguments": "{}"},
        }])
        session.add_message("tool", f"result-{turn}", tool_call_id=f"call-{turn}")
        session.add_message("assistant", f"answer-{turn}")
    manager.save(session)

    newest = manager.display_history_page(session.key, session.messages, turn_limit=2)
    assert [m["content"] for m in newest.messages if m["role"] == "user"] == [
        "question-3",
        "question-4",
    ]
    assert newest.has_older
    assert newest.before_offset is not None
    assert newest.messages[-1]["content"] == "answer-4"

    middle = manager.display_history_page(
        session.key,
        [],
        before_offset=newest.before_offset,
        turn_limit=2,
    )
    assert [m["content"] for m in middle.messages if m["role"] == "user"] == [
        "question-1",
        "question-2",
    ]
    assert middle.has_older

    oldest = manager.display_history_page(
        session.key,
        [],
        before_offset=middle.before_offset,
        turn_limit=2,
    )
    assert [m["content"] for m in oldest.messages if m["role"] == "user"] == [
        "question-0"
    ]
    assert not oldest.has_older
    assert oldest.before_offset is None


def test_turn_page_does_not_split_large_multiline_turn(tmp_path):
    manager = SessionManager(workspace=tmp_path)
    session = manager.get_or_create("cli:large-page")
    session.add_message("user", "old")
    session.add_message("assistant", "old-answer")
    session.add_message("user", "new")
    session.add_message("assistant", "line\n" * 40000)
    manager.save(session)

    page = manager.display_history_page(session.key, session.messages, turn_limit=1)

    assert page.messages[0]["content"] == "new"
    assert page.messages[-1]["content"].startswith("line\n")
    assert page.has_older


def test_existing_transcript_page_does_not_rescan_archive_for_sync(tmp_path, monkeypatch):
    manager = SessionManager(workspace=tmp_path)
    session = manager.get_or_create("cli:no-rescan")
    session.add_message("user", "question")
    session.add_message("assistant", "answer")
    manager.save(session)
    manager.transcripts.known.clear()

    def fail_sync(*args, **kwargs):
        raise AssertionError("existing transcript must not be fully scanned by sync")

    monkeypatch.setattr(manager.transcripts, "sync", fail_sync)
    page = manager.display_history_page(session.key, session.messages, turn_limit=30)

    assert [message["content"] for message in page.messages] == ["question", "answer"]
