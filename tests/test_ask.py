import json
from datetime import datetime
from types import SimpleNamespace

from app import ask


def test_ask_appends_response_metadata_to_json(monkeypatch, tmp_path):
    responses = iter(
        [
            SimpleNamespace(
                text="first answer",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=7,
                ),
            ),
            SimpleNamespace(text="second answer", usage_metadata=None),
        ]
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: next(responses))
    )
    log_path = tmp_path / "ask_history.json"
    monkeypatch.setattr(ask, "USAGE_LOG_PATH", log_path)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.genai, "Client", lambda api_key: client)

    assert ask.ask("first question") == "first answer"
    assert ask.ask("second question") == "second answer"

    entries = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(entries) == 2
    first, second = entries
    assert first["model"] == ask.MODEL
    assert first["input_tokens"] == 12
    assert first["output_tokens"] == 7
    assert first["response_time_seconds"] >= 0
    assert datetime.fromisoformat(first["answered_at"]).utcoffset() is not None
    assert second["input_tokens"] is None
    assert second["output_tokens"] is None