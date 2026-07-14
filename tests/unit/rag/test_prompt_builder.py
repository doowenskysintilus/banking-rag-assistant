from backend.llm.client import ChatTurn
from backend.rag.prompt_builder import PromptBuilder


def test_builds_a_single_user_message_with_no_history():
    messages = PromptBuilder().build(
        "What is the fee?", "CONTEXT:\nsome context", history=None
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "CONTEXT:\nsome context" in messages[0]["content"]
    assert "QUESTION:\nWhat is the fee?" in messages[0]["content"]


def test_prepends_history_turns_before_the_new_question():
    history = [
        ChatTurn(role="user", content="Hi"),
        ChatTurn(role="assistant", content="Hello, how can I help?"),
    ]
    messages = PromptBuilder().build(
        "Follow-up question", "CONTEXT:\nctx", history=history
    )

    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "Hi"}
    assert messages[1] == {"role": "assistant", "content": "Hello, how can I help?"}
    assert messages[2]["role"] == "user"
    assert "Follow-up question" in messages[2]["content"]


def test_empty_history_list_behaves_like_none():
    with_none = PromptBuilder().build("q", "ctx", history=None)
    with_empty = PromptBuilder().build("q", "ctx", history=[])
    assert with_none == with_empty
