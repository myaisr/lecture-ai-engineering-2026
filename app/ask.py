"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
USAGE_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "ask_history.json"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = perf_counter()
    response = client.models.generate_content(model=MODEL, contents=question)
    response_time_seconds = perf_counter() - started_at
    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[usage] 入力: {usage.prompt_token_count} tokens / "
            f"出力: {usage.candidates_token_count} tokens"
        )

    entry = {
        "answered_at": datetime.now().astimezone().isoformat(),
        "model": MODEL,
        "response_time_seconds": round(response_time_seconds, 3),
        "input_tokens": usage.prompt_token_count if usage is not None else None,
        "output_tokens": usage.candidates_token_count if usage is not None else None,
    }
    if USAGE_LOG_PATH.exists():
        entries = json.loads(USAGE_LOG_PATH.read_text(encoding="utf-8"))
    else:
        entries = []
    entries.append(entry)
    USAGE_LOG_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
