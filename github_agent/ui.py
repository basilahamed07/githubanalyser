import json
import logging

import streamlit as st

from graph import build_graph_from_llm_config, run_query, run_query_for_connection
from mcp_connection import get_tools_for_connection_sync, get_tools_sync
from runtime_context import resolve_git_connection


class StreamlitLogHandler(logging.Handler):
    """Store logs in Streamlit session state."""

    def emit(self, record):
        if "logs" not in st.session_state:
            st.session_state.logs = []
        st.session_state.logs.append(self.format(record))


log_handler = StreamlitLogHandler()
log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
logging.basicConfig(
    level=logging.INFO,
    handlers=[log_handler, logging.StreamHandler()],
)


def format_currency(value):
    if value is None:
        return "N/A"
    return f"${value:,.6f}"


def build_metrics_caption(run_result: dict) -> str:
    totals = run_result.get("usage", {}).get("totals", {})
    cost = run_result.get("cost", {}).get("estimated_cost_usd")
    return (
        f"Input: {int(totals.get('input_tokens', 0)):,} | "
        f"Output: {int(totals.get('output_tokens', 0)):,} | "
        f"Total: {int(totals.get('total_tokens', 0)):,} | "
        f"Est. Cost: {format_currency(cost)}"
    )


st.set_page_config(page_title="GitHub Repo Agent", page_icon="🤖", layout="wide")
st.title("🤖 GitHub Repository Intelligence Agent")
st.caption("Powered by LangGraph + Azure OpenAI/Groq + GitHub MCP")


with st.sidebar:
    st.header("Configuration")

    st.subheader("LLM")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["azure_openai", "groq"],
        format_func=lambda provider: "Azure OpenAI" if provider == "azure_openai" else "Groq",
    )

    if llm_provider == "azure_openai":
        azure_endpoint = st.text_input("Azure Endpoint", value="https://YOUR_RESOURCE.openai.azure.com/")
        azure_api_key = st.text_input("Azure API Key", value="", type="password")
        deployment_name = st.text_input("Deployment Name", value="gpt-4o")
        model_name = st.text_input("Model Name", value="gpt-4o")
        api_version = st.text_input("API Version", value="2024-02-01")
        groq_api_key = ""
    else:
        groq_api_key = st.text_input("Groq API Key", value="", type="password")
        model_name = st.text_input("Groq Model Name", value="llama-3.3-70b-versatile")
        azure_endpoint = ""
        azure_api_key = ""
        deployment_name = ""
        api_version = ""

    st.divider()

    st.subheader("GitHub MCP Server")
    mcp_url = st.text_input("MCP Server URL", value="https://api.githubcopilot.com/mcp/")
    mcp_api_key = st.text_input("MCP API Key / GitHub Token", value="", type="password")
    git_connection_payload = st.text_area(
        "GitConnection Row JSON",
        value="",
        help="Optional. Paste a GitConnection row or dict from your DB. repo_url and access_token will be resolved automatically.",
    )

    st.divider()

    st.subheader("LangSmith")
    enable_langsmith = st.checkbox("Enable LangSmith tracing", value=False)
    langsmith_project = st.text_input("LangSmith Project", value="github-repo-agent")
    langsmith_api_url = st.text_input("LangSmith API URL", value="https://api.smith.langchain.com")
    langsmith_web_url = st.text_input("LangSmith Web URL", value="https://smith.langchain.com")
    langsmith_api_key = st.text_input("LangSmith API Key", value="", type="password")

    st.divider()

    st.subheader("Repository")
    repo_owner = st.text_input("Repo Owner", value="")
    repo_name = st.text_input("Repo Name", value="")

    connect_btn = st.button("Connect & Load Tools", type="primary")


for key, default in [
    ("tools_loaded", False),
    ("graph", None),
    ("chat_history", []),
    ("logs", []),
    ("tools", []),
    ("last_run_metrics", None),
    ("connected_model_name", None),
    ("connected_llm_provider", None),
    ("connected_git_connection", None),
    ("connected_repo_context", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


if connect_btn:
    git_connection = None
    repo_context = None

    llm_config = {
        "llm_provider": llm_provider,
        "model_name": model_name,
    }
    if llm_provider == "azure_openai":
        llm_config.update(
            {
                "azure_endpoint": azure_endpoint,
                "azure_api_key": azure_api_key,
                "deployment_name": deployment_name,
                "api_version": api_version,
            }
        )
        missing_llm = not all([azure_endpoint, azure_api_key, deployment_name, model_name, api_version])
        llm_error = "Please fill all Azure OpenAI settings."
    else:
        llm_config.update({"groq_api_key": groq_api_key})
        missing_llm = not all([groq_api_key, model_name])
        llm_error = "Please fill all Groq settings."

    if git_connection_payload.strip():
        try:
            git_connection = json.loads(git_connection_payload)
            repo_context = resolve_git_connection(
                git_connection,
                github_token=mcp_api_key or None,
                mcp_url=mcp_url or None,
            )
        except Exception as exc:
            st.sidebar.error(f"Invalid GitConnection payload: {exc}")

    missing_mcp = not git_connection and not all([mcp_url, mcp_api_key])

    if missing_llm:
        st.sidebar.error(llm_error)
    elif missing_mcp:
        st.sidebar.error("Please fill MCP URL and token, or provide a GitConnection JSON payload.")
    else:
        st.session_state.logs = []
        with st.sidebar:
            with st.spinner("Connecting to MCP server..."):
                try:
                    if git_connection:
                        tools = get_tools_for_connection_sync(
                            git_connection,
                            mcp_url=mcp_url or None,
                            github_token=mcp_api_key or None,
                        )
                    else:
                        tools = get_tools_sync(mcp_url, mcp_api_key)

                    graph = build_graph_from_llm_config(llm_config, all_tools=tools)

                    st.session_state.tools_loaded = True
                    st.session_state.graph = graph
                    st.session_state.tools = tools
                    st.session_state.last_run_metrics = None
                    st.session_state.connected_model_name = model_name
                    st.session_state.connected_llm_provider = llm_provider
                    st.session_state.connected_git_connection = git_connection
                    st.session_state.connected_repo_context = repo_context.safe_metadata() if repo_context else None

                    st.success(f"Connected! {len(tools)} tools loaded.")
                    if repo_context:
                        st.caption(f"Repo resolved from GitConnection: {repo_context.repo_full_name}")
                    with st.expander("Tools loaded"):
                        st.write([t.name for t in tools])
                except Exception as exc:
                    st.error(f"Connection failed: {exc}")
                    logging.error(f"[CONNECT] Failed: {exc}")


col_chat, col_status, col_logs = st.columns([2, 1, 1])

with col_chat:
    st.subheader("Ask Anything About the Repo")

    examples = [
        "What tech stack is used?",
        "List all API endpoints",
    ]

    st.caption("Quick queries:")
    ex_cols = st.columns(3)
    selected_example = None
    for i, ex in enumerate(examples):
        with ex_cols[i % 3]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                selected_example = ex

    st.divider()

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["query"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            if chat.get("metrics"):
                st.caption(build_metrics_caption(chat["metrics"]))
            if chat.get("trace_url"):
                st.markdown(f"[Open LangSmith trace]({chat['trace_url']})")

    user_input = st.chat_input("Ask anything about the repository...")
    if selected_example:
        user_input = selected_example

    if user_input:
        if not st.session_state.tools_loaded:
            st.error("Please connect to MCP server first!")
        elif not st.session_state.connected_git_connection and (not repo_owner or not repo_name):
            st.error("Please enter Repo Owner and Repo Name in the sidebar!")
        elif enable_langsmith and not langsmith_api_key:
            st.error("Please enter a LangSmith API key or disable tracing.")
        else:
            st.session_state.logs = []
            active_model_name = st.session_state.connected_model_name or model_name
            tracing_config = {
                "enabled": enable_langsmith,
                "project_name": langsmith_project,
                "api_url": langsmith_api_url,
                "web_url": langsmith_web_url,
                "api_key": langsmith_api_key,
            }

            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Agent thinking and executing..."):
                    try:
                        if st.session_state.connected_git_connection:
                            run_result = run_query_for_connection(
                                graph=st.session_state.graph,
                                user_query=user_input,
                                git_connection=st.session_state.connected_git_connection,
                                model_name=active_model_name,
                                tracing_config=tracing_config,
                                github_token=mcp_api_key or None,
                                mcp_url=mcp_url or None,
                            )
                        else:
                            run_result = run_query(
                                graph=st.session_state.graph,
                                user_query=user_input,
                                repo_owner=repo_owner,
                                repo_name=repo_name,
                                model_name=active_model_name,
                                tracing_config=tracing_config,
                            )

                        answer = run_result["answer"]
                        st.markdown(answer)
                        st.caption(build_metrics_caption(run_result))
                        if run_result.get("trace_url"):
                            st.markdown(f"[Open LangSmith trace]({run_result['trace_url']})")

                        st.session_state.last_run_metrics = run_result
                        st.session_state.chat_history.append(
                            {
                                "query": user_input,
                                "answer": answer,
                                "metrics": run_result,
                                "trace_url": run_result.get("trace_url"),
                            }
                        )
                    except Exception as exc:
                        st.error(f"Agent error: {exc}")
                        logging.error(f"[UI] Agent error: {exc}", exc_info=True)

with col_status:
    st.subheader("Agent Status")

    if st.session_state.tools_loaded:
        st.success("Connected")
        st.metric("Tools Loaded", len(st.session_state.tools))
        st.metric("Queries Run", len(st.session_state.chat_history))
        if st.session_state.connected_llm_provider:
            st.caption(f"Provider: {st.session_state.connected_llm_provider}")
        if st.session_state.connected_model_name:
            st.caption(f"Model: {st.session_state.connected_model_name}")
        if st.session_state.connected_repo_context:
            st.caption(f"Repo: {st.session_state.connected_repo_context['repo_full_name']}")
            if st.session_state.connected_repo_context.get("github_username"):
                st.caption(f"GitHub User: {st.session_state.connected_repo_context['github_username']}")

        last_run_metrics = st.session_state.get("last_run_metrics")
        if last_run_metrics:
            totals = last_run_metrics.get("usage", {}).get("totals", {})
            st.divider()
            st.subheader("Last Run")
            st.metric("Input Tokens", f"{int(totals.get('input_tokens', 0)):,}")
            st.metric("Output Tokens", f"{int(totals.get('output_tokens', 0)):,}")
            st.metric("Total Tokens", f"{int(totals.get('total_tokens', 0)):,}")
            st.metric("Est. Cost", format_currency(last_run_metrics.get("cost", {}).get("estimated_cost_usd")))
            if last_run_metrics.get("trace_project"):
                st.caption(f"LangSmith Project: {last_run_metrics['trace_project']}")
            if last_run_metrics.get("trace_url"):
                st.markdown(f"[Open LangSmith trace]({last_run_metrics['trace_url']})")

        st.subheader("Nodes")
        for node in [
            "Repo Analyst",
            "Code Analyst",
            "PR Node",
            "Issues Node",
            "Team Node",
            "Security Node",
            "Copilot Node",
            "Write Ops",
            "Meta Node",
        ]:
            st.caption(node)
    else:
        st.warning("Not Connected")

    if st.session_state.chat_history:
        st.divider()
        st.subheader("History")
        for chat in reversed(st.session_state.chat_history[-5:]):
            st.caption(f"Q: {chat['query'][:40]}...")
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.session_state.last_run_metrics = None
            st.rerun()

with col_logs:
    st.subheader("Live Agent Logs")

    if st.button("Refresh Logs"):
        st.rerun()

    if st.button("Clear Logs"):
        st.session_state.logs = []
        st.rerun()

    logs = st.session_state.get("logs", [])
    if logs:
        st.text_area(
            label="Logs",
            value="\n".join(logs),
            height=600,
            label_visibility="collapsed",
        )
    else:
        st.info("Logs will appear here when agent runs.")
