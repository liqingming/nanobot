from nanobot.session.resume_state import is_ambiguous_resume_request


def test_ambiguous_resume_request_recognizes_unspecified_continuations() -> None:
    assert is_ambiguous_resume_request("继续")
    assert is_ambiguous_resume_request("继续中断任务")
    assert is_ambiguous_resume_request("请恢复上次未完成的工作。")
    assert is_ambiguous_resume_request("接着之前的计划")


def test_ambiguous_resume_request_does_not_capture_named_objectives() -> None:
    assert not is_ambiguous_resume_request("继续分析 BinaryAddressableProvider")
    assert not is_ambiguous_resume_request("恢复登录模块重构")
    assert not is_ambiguous_resume_request("把文档里的四处类名修正")
    assert not is_ambiguous_resume_request("")
