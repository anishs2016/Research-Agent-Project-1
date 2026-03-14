import os
import json
from typing import Generator

import anthropic
from tavily import TavilyClient
from dotenv import load_dotenv


load_dotenv()
load_dotenv("venv/.env", override=False)

MODEL = "claude-sonnet-4-20250514"

def get_system_prompt() -> str:
    from datetime import date
    today = date.today().strftime("%B %d, %Y")
    return f"""Today's date is {today}. You are a thorough research assistant. When given a question:
1. Identify what specific information you need to find
2. Use the web_search tool to find relevant, up-to-date information
3. Search multiple times with different queries if needed to get comprehensive coverage
4. Synthesize a clear, well-structured answer with inline citations

Always cite your sources with URLs. If the first search isn't sufficient, keep searching until you have enough information to give a complete answer.
Always use the current date in your search queries. Never search for outdated years like 2024 — always search for the most recent content relative to today."""

SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current information. "
        "Use specific, targeted queries to find relevant results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The specific search query to look up",
            }
        },
        "required": ["query"],
    },
}


def _make_anthropic_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. "
            "Add it to your .env file as: ANTHROPIC_API_KEY=your_key_here"
        )
    return anthropic.Anthropic(api_key=api_key)


def _make_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY not found. "
            "Add it to your .env file as: TAVILY_API_KEY=your_key_here"
        )
    return TavilyClient(api_key=api_key)


def _run_search(query: str, days: int = 7) -> str:
    """Execute a Tavily search limited to recent content and return formatted results."""
    try:
        tavily = _make_tavily_client()
        response = tavily.search(
            query=query,
            max_results=5,
            days=days,
            include_answer=True,
            include_raw_content=False,
        )

        parts = []
        if response.get("answer"):
            parts.append(f"Quick answer: {response['answer']}\n")

        for i, result in enumerate(response.get("results", []), 1):
            parts.append(
                f"[{i}] {result.get('title', 'Untitled')}\n"
                f"URL: {result.get('url', '')}\n"
                f"Excerpt: {result.get('content', '')}"
            )

        return "\n\n".join(parts) if parts else "No results found for this query."
    except Exception as e:
        return f"Search error: {e}"


MAX_ITERATIONS = 10


def research_agent(question: str, days: int = 7) -> Generator[dict, None, None]:
    """
    Multi-step research agent. Loops until it has enough info to answer.

    Yields dicts with one of these shapes:
      {"type": "search",     "query": str}     — agent is about to search
      {"type": "text_delta", "text":  str}     — streaming answer text chunk
      {"type": "warning",    "message": str}   — non-fatal notice (e.g. iteration cap)
      {"type": "error",      "message": str}   — something went wrong
    """
    try:
        client = _make_anthropic_client()
    except ValueError as e:
        yield {"type": "error", "message": str(e)}
        return

    messages: list[dict] = [{"role": "user", "content": question}]
    finished = False

    for _ in range(MAX_ITERATIONS):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=4096,
                system=get_system_prompt(),
                tools=[SEARCH_TOOL],
                messages=messages,
            ) as stream:
                # Stream any text Claude emits in this turn
                for text in stream.text_stream:
                    yield {"type": "text_delta", "text": text}

                response = stream.get_final_message()

        except anthropic.AuthenticationError:
            yield {"type": "error", "message": "Invalid ANTHROPIC_API_KEY. Check your .env file."}
            return
        except anthropic.RateLimitError:
            yield {"type": "error", "message": "Anthropic rate limit reached. Please wait and try again."}
            return
        except anthropic.APIConnectionError:
            yield {"type": "error", "message": "Could not connect to the Anthropic API. Check your internet connection."}
            return
        except anthropic.APIStatusError as e:
            yield {"type": "error", "message": f"Anthropic API error ({e.status_code}): {e.message}"}
            return

        # Append the full assistant turn (preserves tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            finished = True
            break

        if response.stop_reason != "tool_use":
            finished = True
            break

        # Execute every tool call Claude requested
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "web_search":
                query = block.input.get("query", "")
                yield {"type": "search", "query": query}
                result = _run_search(query, days=days)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        if not tool_results:
            finished = True
            break

        messages.append({"role": "user", "content": tool_results})

    if not finished:
        yield {
            "type": "warning",
            "message": f"Reached the {MAX_ITERATIONS}-iteration search limit — the answer may be incomplete. Try re-running.",
        }
