import pandas as pd
import streamlit as st

from agent import research_agent
from stocks import stocks_agent

st.set_page_config(
    page_title="Research & Stock Agent",
    page_icon="🔍",
    layout="centered",
)

tab_research, tab_stocks = st.tabs(["🔍 Research", "📈 Stock Predictor"])


# ── Research tab ──────────────────────────────────────────────────────────────
with tab_research:
    st.title("🔍 Research Agent")
    st.caption(
        "Ask any question. The agent will search the web multiple times if needed, "
        "then synthesize a cited answer."
    )

    question = st.text_area(
        "Your question",
        placeholder="e.g. What are the latest developments in fusion energy?",
        height=100,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        r_submitted = st.button("Research →", type="primary", use_container_width=True)
    with col2:
        r_days = st.selectbox(
            "Recency", [7, 14, 30],
            format_func=lambda d: f"Last {d}d",
            label_visibility="collapsed",
            key="r_days",
        )

    if r_submitted:
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.expander("🔎 Search activity", expanded=True):
                search_log = st.empty()

            st.divider()
            answer_placeholder = st.empty()

            searches: list[str] = []
            full_answer = ""
            had_error = False

            for event in research_agent(question, days=r_days):
                match event["type"]:
                    case "search":
                        searches.append(event["query"])
                        search_log.markdown("\n".join(f"- 🔍 {q}" for q in searches))

                    case "text_delta":
                        full_answer += event["text"]
                        answer_placeholder.markdown(full_answer)

                    case "warning":
                        st.warning(event["message"])

                    case "error":
                        st.error(f"**Error:** {event['message']}", icon="🚨")
                        had_error = True
                        break

            if not had_error:
                if searches:
                    search_log.markdown(
                        "\n".join(f"- ✅ {q}" for q in searches)
                        + f"\n\n*{len(searches)} search(es) completed.*"
                    )
                if not full_answer:
                    st.info("The agent finished but produced no text output.")


# ── Stock Predictor tab ───────────────────────────────────────────────────────
with tab_stocks:
    st.title("📈 Stock Predictor")
    st.caption(
        "Enter a ticker to get an AI prediction grounded in real price data "
        "and recent news — not financial advice."
    )

    col_ticker, col_company = st.columns([1, 2])
    with col_ticker:
        ticker = st.text_input("Ticker", placeholder="AAPL").strip().upper()
    with col_company:
        company = st.text_input("Company name", placeholder="Apple Inc.").strip()

    col_btn, col_days = st.columns([3, 1])
    with col_btn:
        s_submitted = st.button("Analyze →", type="primary", use_container_width=True)
    with col_days:
        s_days = st.selectbox(
            "Recency", [7, 14, 30],
            format_func=lambda d: f"Last {d}d",
            label_visibility="collapsed",
            key="s_days",
        )

    if s_submitted:
        if not ticker or not company:
            st.warning("Please enter both a ticker symbol and a company name.")
        else:
            # Placeholders set up before the loop so layout order is guaranteed
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()

            with st.expander("🔎 Search activity", expanded=True):
                search_log = st.empty()

            st.divider()
            verdict_placeholder = st.empty()
            result_placeholder = st.empty()

            searches: list[str] = []
            full_text = ""
            had_error = False

            for event in stocks_agent(ticker, company, days=s_days):
                match event["type"]:
                    case "price_data":
                        current = event["current_price"]
                        change = event["change_5d_pct"]
                        arrow = "▲" if change >= 0 else "▼"

                        with metrics_placeholder.container():
                            m1, m2 = st.columns(2)
                            m1.metric("Current Price", f"${current:,.2f}")
                            m2.metric("5-Day Change", f"{arrow} {abs(change):.2f}%",
                                      delta=f"{change:+.2f}%",
                                      delta_color="normal")

                        df = pd.DataFrame(
                            {"Close ($)": event["hist_30d"]}
                        )
                        chart_placeholder.line_chart(df, use_container_width=True)

                    case "search":
                        searches.append(event["query"])
                        search_log.markdown("\n".join(f"- 🔍 {q}" for q in searches))

                    case "text_delta":
                        full_text += event["text"]
                        result_placeholder.markdown(full_text)

                    case "warning":
                        st.warning(event["message"])

                    case "error":
                        st.error(f"**Error:** {event['message']}", icon="🚨")
                        had_error = True
                        break

            if not had_error:
                # Render verdict banner once after streaming is complete
                upper = full_text.upper()
                if "BULLISH" in upper:
                    verdict_placeholder.success("Verdict: **BULLISH 📈**")
                elif "BEARISH" in upper:
                    verdict_placeholder.error("Verdict: **BEARISH 📉**")
                elif "NEUTRAL" in upper:
                    verdict_placeholder.warning("Verdict: **NEUTRAL ➡️**")

                if searches:
                    search_log.markdown(
                        "\n".join(f"- ✅ {q}" for q in searches)
                        + f"\n\n*{len(searches)} search(es) completed.*"
                    )
                if not full_text:
                    st.info("The agent finished but produced no text output.")
