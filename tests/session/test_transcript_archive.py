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
