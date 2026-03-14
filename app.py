import streamlit as st
from agent import research_agent

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔍",
    layout="centered",
)

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

submitted = st.button("Research →", type="primary", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        # ── Search activity log ──────────────────────────────────────────────
        with st.expander("🔎 Search activity", expanded=True):
            search_log = st.empty()

        # ── Answer ───────────────────────────────────────────────────────────
        st.divider()
        answer_placeholder = st.empty()

        searches: list[str] = []
        full_answer = ""
        had_error = False

        for event in research_agent(question):
            match event["type"]:
                case "search":
                    searches.append(event["query"])
                    search_log.markdown(
                        "\n".join(f"- 🔍 {q}" for q in searches)
                    )

                case "text_delta":
                    full_answer += event["text"]
                    answer_placeholder.markdown(full_answer)

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
