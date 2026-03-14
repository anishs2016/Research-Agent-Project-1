import os
from typing import Generator

import anthropic
import yfinance as yf
from dotenv import load_dotenv

from agent import SEARCH_TOOL, MAX_ITERATIONS, _run_search, _make_anthropic_client

load_dotenv()
load_dotenv("venv/.env", override=False)

MODEL = "claude-sonnet-4-20250514"

# Base system prompt — recent price data is appended dynamically per call
def _get_system_base() -> str:
    from datetime import date
    today = date.today().strftime("%B %d, %Y")
    return f"""Today's date is {today}. You are a financial analyst. Your job is to predict near-term stock performance based on recent news AND the actual price data provided to you.

When given a ticker and company name:
1. Search for recent earnings, revenue, and guidance
2. Search for analyst upgrades, downgrades, and price target changes
3. Search for product launches, partnerships, contracts, or layoffs
4. Search for any legal, regulatory, or macro risks facing the company
5. Search for competitor news that could affect this stock

You will be given the last 10 days of closing prices in this system prompt. Use those numbers to ground your price target in reality.

Produce your analysis in exactly this format:

## Verdict: [BULLISH 📈 / BEARISH 📉 / NEUTRAL ➡️]
**Confidence:** [High / Medium / Low]

### Price Target (1–4 weeks): $X–$Y
Base this range on the current price, recent trend, and news sentiment.

### Key Drivers
- Driver 1 (cite source)
- Driver 2 (cite source)
- Driver 3 (cite source)

### Near-Term Outlook (1–4 weeks)
2–3 sentences on likely price direction and why.

### Risks That Could Change This
- Risk 1
- Risk 2

### Sources
- [Title](URL)

---
⚠️ *AI-generated analysis based on news sentiment and price trend only — not financial advice.*"""


def _fetch_price_data(ticker: str) -> dict:
    """
    Fetch the last 90 days of price history via yfinance.

    Returns a dict with:
      current_price    float
      change_5d_pct    float
      hist_30d         dict[str, float]  — date string → close price (for UI chart)
      price_context    str               — formatted last-10-day block for LLM
    On failure, returns {"error": str}.
    """
    try:
        hist = yf.Ticker(ticker).history(period="90d")
        if hist.empty:
            return {"error": f"No price data returned for '{ticker}'. Check the ticker symbol."}

        closes = hist["Close"]
        current_price = float(closes.iloc[-1])
        ref_price = float(closes.iloc[-6]) if len(closes) >= 6 else float(closes.iloc[0])
        change_5d_pct = (current_price - ref_price) / ref_price * 100

        last_10 = closes.tail(10)
        price_context = "\n".join(
            f"  {date.strftime('%Y-%m-%d')}: ${float(price):.2f}"
            for date, price in last_10.items()
        )

        hist_30d = {
            date.strftime("%Y-%m-%d"): round(float(price), 2)
            for date, price in closes.tail(30).items()
        }

        return {
            "current_price": round(current_price, 2),
            "change_5d_pct": round(change_5d_pct, 2),
            "hist_30d": hist_30d,
            "price_context": price_context,
        }
    except Exception as e:
        return {"error": str(e)}


def stocks_agent(ticker: str, company: str, days: int = 7) -> Generator[dict, None, None]:
    """
    Research and predict stock performance from recent news + real price data.

    Yields dicts:
      {"type": "price_data",  "current_price": float, "change_5d_pct": float,
                               "hist_30d": dict[str, float], "ticker": str}
      {"type": "search",      "query": str}
      {"type": "text_delta",  "text": str}
      {"type": "warning",     "message": str}
      {"type": "error",       "message": str}
    """
    # ── 1. Fetch price data before touching the LLM ───────────────────────────
    price_data = _fetch_price_data(ticker)

    if "error" in price_data:
        yield {"type": "warning", "message": f"Price data unavailable: {price_data['error']}. Proceeding with news analysis only."}
        price_section = ""
    else:
        yield {
            "type": "price_data",
            "current_price": price_data["current_price"],
            "change_5d_pct": price_data["change_5d_pct"],
            "hist_30d": price_data["hist_30d"],
            "ticker": ticker,
        }
        price_section = (
            f"\n\nRecent closing prices for {ticker} (last 10 trading days):\n"
            f"{price_data['price_context']}\n"
            f"Current price: ${price_data['current_price']:.2f} "
            f"({'▲' if price_data['change_5d_pct'] >= 0 else '▼'}"
            f"{abs(price_data['change_5d_pct']):.2f}% vs 5 days ago)"
        )

    # ── 2. Build system prompt with price context ─────────────────────────────
    system = _get_system_base() + price_section

    # ── 3. Run agentic loop ───────────────────────────────────────────────────
    try:
        client = _make_anthropic_client()
    except ValueError as e:
        yield {"type": "error", "message": str(e)}
        return

    prompt = (
        f"Analyze {company} (ticker: {ticker}) and predict its near-term stock performance "
        f"based on news from the last {days} days."
    )
    messages: list[dict] = [{"role": "user", "content": prompt}]
    finished = False

    for _ in range(MAX_ITERATIONS):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=4096,
                system=system,
                tools=[SEARCH_TOOL],
                messages=messages,
            ) as stream:
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
            yield {"type": "error", "message": "Could not connect to the Anthropic API."}
            return
        except anthropic.APIStatusError as e:
            yield {"type": "error", "message": f"Anthropic API error ({e.status_code}): {e.message}"}
            return

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            finished = True
            break
        if response.stop_reason != "tool_use":
            finished = True
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "web_search":
                query = block.input.get("query", "")
                yield {"type": "search", "query": query}
                result = _run_search(query, days=days)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )

        if not tool_results:
            finished = True
            break
        messages.append({"role": "user", "content": tool_results})

    if not finished:
        yield {
            "type": "warning",
            "message": f"Reached the {MAX_ITERATIONS}-iteration search limit — the analysis may be incomplete. Try re-running.",
        }
