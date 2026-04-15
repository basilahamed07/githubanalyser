import logging
import asyncio
from functools import partial
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client
from langgraph.graph import StateGraph, END

from agent_state import AgentState
from agent_nodes import (
    intent_classifier_node, orchestrator_node,
    repo_analyst_node, code_analyst_node, pr_node,
    issues_node, team_node, security_node, copilot_node,
    write_ops_node, meta_node, synthesizer_node, get_llm
)
from runtime_context import resolve_git_connection, resolve_llm_config
from tracking import estimate_usage_cost, summarize_usage_metadata

logger = logging.getLogger(__name__)


def router_edge(state: dict) -> str:
    if state.get("needs_clarification"):
        logger.info("[ROUTER] → clarify (needs clarification)")
        return "synthesizer"
    if state.get("is_complete"):
        logger.info("[ROUTER] → synthesizer (is_complete=True)")
        return "synthesizer"

    next_node = state.get("next_node", "synthesizer")
    valid_nodes = [
        "repo_analyst", "code_analyst", "pr_node",
        "issues_node", "team_node", "security_node",
        "copilot_node", "write_ops", "meta_node", "synthesizer"
    ]

    if next_node not in valid_nodes:
        logger.warning(f"[ROUTER] Invalid next_node '{next_node}' → defaulting to synthesizer")
        return "synthesizer"

    logger.info(f"[ROUTER] → {next_node}")
    return next_node


def build_graph(
    azure_endpoint=None,
    azure_api_key=None,
    deployment_name=None,
    api_version=None,
    model_name=None,
    all_tools=None,
    llm_provider="azure_openai",
    groq_api_key=None,
):
    logger.info("[GRAPH] Building LangGraph agent...")

    tracked_model_name = model_name or deployment_name
    llm = get_llm(
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        deployment_name=deployment_name,
        api_version=api_version,
        model_name=tracked_model_name,
        llm_provider=llm_provider,
        groq_api_key=groq_api_key,
    )

    # Wrap async nodes with partial for dependency injection
    intent_node    = partial(intent_classifier_node, llm=llm)
    orch_node      = partial(orchestrator_node, llm=llm)
    repo_node      = partial(repo_analyst_node, llm=llm, all_tools=all_tools)
    code_node      = partial(code_analyst_node, llm=llm, all_tools=all_tools)
    pr_node_fn     = partial(pr_node, llm=llm, all_tools=all_tools)
    issues_node_fn = partial(issues_node, llm=llm, all_tools=all_tools)
    team_node_fn   = partial(team_node, llm=llm, all_tools=all_tools)
    sec_node       = partial(security_node, llm=llm, all_tools=all_tools)
    cop_node       = partial(copilot_node, llm=llm, all_tools=all_tools)
    write_node     = partial(write_ops_node, llm=llm, all_tools=all_tools)
    meta_node_fn   = partial(meta_node, llm=llm, all_tools=all_tools)
    synth_node     = partial(synthesizer_node, llm=llm)

    graph = StateGraph(AgentState)

    graph.add_node("intent_classifier", intent_node)
    graph.add_node("orchestrator",      orch_node)
    graph.add_node("repo_analyst",      repo_node)
    graph.add_node("code_analyst",      code_node)
    graph.add_node("pr_node",           pr_node_fn)
    graph.add_node("issues_node",       issues_node_fn)
    graph.add_node("team_node",         team_node_fn)
    graph.add_node("security_node",     sec_node)
    graph.add_node("copilot_node",      cop_node)
    graph.add_node("write_ops",         write_node)
    graph.add_node("meta_node",         meta_node_fn)
    graph.add_node("synthesizer",       synth_node)

    graph.set_entry_point("intent_classifier")
    graph.add_edge("intent_classifier", "orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        router_edge,
        {
            "repo_analyst":  "repo_analyst",
            "code_analyst":  "code_analyst",
            "pr_node":       "pr_node",
            "issues_node":   "issues_node",
            "team_node":     "team_node",
            "security_node": "security_node",
            "copilot_node":  "copilot_node",
            "write_ops":     "write_ops",
            "meta_node":     "meta_node",
            "synthesizer":   "synthesizer",
        }
    )

    # All specialist nodes loop back to orchestrator
    for node_name in [
        "repo_analyst", "code_analyst", "pr_node",
        "issues_node", "team_node", "security_node",
        "copilot_node", "write_ops", "meta_node"
    ]:
        graph.add_edge(node_name, "orchestrator")

    graph.add_edge("synthesizer", END)

    compiled = graph.compile()
    setattr(compiled, "_github_agent_model_name", tracked_model_name)
    setattr(compiled, "_github_agent_llm_provider", llm_provider)
    logger.info("[GRAPH] Graph compiled successfully")
    return compiled


def build_graph_from_llm_config(llm_config, all_tools):
    resolved = resolve_llm_config(llm_config)
    return build_graph(
        llm_provider=resolved.llm_provider,
        model_name=resolved.model_name,
        azure_endpoint=resolved.azure_endpoint,
        azure_api_key=resolved.azure_api_key,
        deployment_name=resolved.deployment_name,
        api_version=resolved.api_version,
        groq_api_key=resolved.groq_api_key,
        all_tools=all_tools,
    )


def _build_langsmith_tracer(tracing_config: dict | None) -> LangChainTracer | None:
    if not tracing_config or not tracing_config.get("enabled"):
        return None

    client = Client(
        api_url=tracing_config.get("api_url"),
        api_key=tracing_config.get("api_key"),
        web_url=tracing_config.get("web_url"),
    )
    return LangChainTracer(
        project_name=tracing_config.get("project_name"),
        client=client,
    )


async def run_query_async(
    graph,
    user_query,
    repo_owner,
    repo_name,
    model_name,
    tracing_config=None,
    repo_context=None,
) -> dict:
    logger.info(f"[RUN] Starting query: '{user_query}' on {repo_owner}/{repo_name}")

    initial_state = {
        "user_query": user_query,
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "messages": [],
        "intermediate_results": {},
        "tool_calls_made": [],
        "loop_count": 0,
        "is_complete": False,
        "needs_clarification": False,
        "plan": [],
        "current_step": 0,
        "next_node": "",
        "intent": "",
        "domain": [],
        "final_answer": "",
        "error": None
    }
    if repo_context is not None:
        initial_state.update({
            "git_id": repo_context.git_id,
            "application_id": repo_context.application_id,
            "customer_id": repo_context.customer_id,
            "group_id": repo_context.group_id,
            "repo_url": repo_context.repo_url,
            "default_branch": repo_context.default_branch,
            "branch": repo_context.branch,
            "github_username": repo_context.username,
        })

    graph_config = {
        "run_name": "github_repo_agent_query",
        "tags": ["streamlit", "github-agent", f"model:{model_name}"],
        "metadata": {
            "repo_owner": repo_owner,
            "repo_name": repo_name,
            "model_name": model_name,
            "interface": "streamlit",
        },
    }
    if repo_context is not None:
        graph_config["metadata"].update(repo_context.safe_metadata())

    tracer = _build_langsmith_tracer(tracing_config)
    if tracer is not None:
        graph_config["callbacks"] = [tracer]
        logger.info(
            f"[RUN] LangSmith tracing enabled for project={tracing_config.get('project_name')}"
        )

    with get_usage_metadata_callback() as usage_callback:
        result = await graph.ainvoke(initial_state, config=graph_config)

    usage_summary = summarize_usage_metadata(usage_callback.usage_metadata)
    cost_summary = estimate_usage_cost(usage_summary, model_name)

    trace_url = None
    if tracer is not None:
        try:
            tracer.wait_for_futures()
            trace_url = tracer.get_run_url()
            logger.info(f"[RUN] LangSmith trace available at: {trace_url}")
        except Exception as exc:
            logger.warning(f"[RUN] Unable to fetch LangSmith run URL: {exc}")

    answer = result.get("final_answer", "No answer generated.")
    logger.info(
        f"[RUN] Query complete. Answer length: {len(answer)} | total_tokens={usage_summary['totals']['total_tokens']}"
    )
    return {
        "answer": answer,
        "usage": usage_summary,
        "cost": cost_summary,
        "trace_url": trace_url,
        "trace_project": tracing_config.get("project_name") if tracing_config else None,
    }


def run_query(graph, user_query, repo_owner, repo_name, model_name, tracing_config=None) -> dict:
    """Sync wrapper - runs async graph in event loop."""
    return asyncio.run(
        run_query_async(
            graph,
            user_query,
            repo_owner,
            repo_name,
            model_name,
            tracing_config=tracing_config,
        )
    )


async def run_query_for_connection_async(
    graph,
    user_query,
    git_connection,
    model_name=None,
    tracing_config=None,
    github_token=None,
    mcp_url=None,
) -> dict:
    repo_context = resolve_git_connection(
        git_connection,
        github_token=github_token,
        mcp_url=mcp_url,
    )
    tracked_model_name = model_name or getattr(graph, "_github_agent_model_name", None)
    if not tracked_model_name:
        raise ValueError("model_name is required when the graph does not expose one")

    return await run_query_async(
        graph=graph,
        user_query=user_query,
        repo_owner=repo_context.repo_owner,
        repo_name=repo_context.repo_name,
        model_name=tracked_model_name,
        tracing_config=tracing_config,
        repo_context=repo_context,
    )


def run_query_for_connection(
    graph,
    user_query,
    git_connection,
    model_name=None,
    tracing_config=None,
    github_token=None,
    mcp_url=None,
) -> dict:
    return asyncio.run(
        run_query_for_connection_async(
            graph=graph,
            user_query=user_query,
            git_connection=git_connection,
            model_name=model_name,
            tracing_config=tracing_config,
            github_token=github_token,
            mcp_url=mcp_url,
        )
    )
