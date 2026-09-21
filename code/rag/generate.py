"""Grounded generation over retrieved policy chunks, with citations."""

from __future__ import annotations

import os

import anthropic

from eval.instrumentation import instrument_call

MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = (
    "You are an insurance coverage assistant. Answer ONLY using the provided "
    "policy excerpts below. Every factual claim must cite its source as "
    "[Insurer Product DocType, page N]. If the excerpts do not contain enough "
    "information to answer, say so explicitly -- never guess or use outside "
    "knowledge of insurance policies."
)


_client_instance: anthropic.Anthropic | None = None


def _client() -> anthropic.Anthropic:
    # Built once and reused -- a fresh client per call (as this used to do)
    # means a fresh HTTP connection pool per call too, which adds up when
    # generate_answer() runs in a loop (eval scripts, batch runs).
    global _client_instance
    if _client_instance is None:
        # Identity-linked API keys (workspace-scoped orgs) require the workspace
        # acting on this request to be named explicitly -- the SDK only attaches
        # this automatically on the WIF/Bearer-token credential path, not for a
        # plain api_key= client, so it has to go on as a header here.
        headers = {}
        workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        if workspace_id:
            headers["anthropic-workspace-id"] = workspace_id
        _client_instance = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=headers)
    return _client_instance


def build_context(retrieved_chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{c['insurer']} {c['product']} {c['doc_type']}, page {c['page']}]\n{c['text']}"
        for c in retrieved_chunks
    )


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return "I don't have enough information to answer that -- no matching policy text was found."

    context = build_context(retrieved_chunks)
    with instrument_call("retrieve_policy") as rec:
        response = _client().messages.create(
            model=MODEL,
            max_tokens=800,
            system=[
                {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
            ],
            messages=[
                {"role": "user", "content": f"Policy excerpts:\n\n{context}\n\nQuestion: {question}"}
            ],
        )
        rec["response"] = response
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise RuntimeError(
            f"No text block in response (stop_reason={response.stop_reason!r}) -- "
            f"likely a refusal or truncation before any text was generated."
        )
    return text
