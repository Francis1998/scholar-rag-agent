# Paper Chat Memory Guide

![Paper chat memory demo](../assets/paper-chat-memory.gif)

`PaperChatMemory` stores multi-turn scholarly chat turns in SQLite, keyed by
`session_id`, with optional `document_ids` / `chunk_ids` provenance. Inspired by
LocalGPT / PrivateGPT academic chat memory and PaperQA multi-turn paper Q&A.
Distinct from the agent event log and from retrieval gates. Local memory for
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 paper-chat pipelines (not a
DOI connector).

## Usage

```python
from storage.paper_chat_memory import PaperChatMemory

memory = PaperChatMemory("paper_chat.db")
memory.append_turn("s1", "user", "What dataset is used?", document_ids=["paper-a"])
memory.append_turn(
    "s1", "assistant", "The authors use MIMIC-III.", document_ids=["paper-a"], chunk_ids=["c12"]
)
prompt_block = memory.format_context("s1", max_chars=2000)
```

`get_turns(..., limit=N)` returns the newest N turns in chronological order.
`clear_session` deletes one session without affecting others.
