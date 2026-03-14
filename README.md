# 🔍 Research Agent

An agentic AI research assistant that searches the web and synthesizes cited answers in real time.

Ask any question → the agent breaks it into search queries → fetches live results → streams back a cited answer.

## Demo
[add a screenshot or GIF here]

## Tech Stack
- **Claude API** (claude-sonnet) — reasoning & tool calling
- **Tavily** — real-time web search
- **Streamlit** — UI & streaming

## How it works
1. User asks a question
2. Claude decides what to search for
3. Tavily fetches live web results
4. Claude synthesizes a cited answer
5. Repeats until it has enough information

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API keys to `.env`:
ANTHROPIC_API_KEY=your_key
TAVILY_API_KEY=your_key
4. Run: `streamlit run app.py`

## Requirements
- Python 3.8+
- Anthropic API key
- Tavily API key
